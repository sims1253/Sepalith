#!/usr/bin/env python3
"""S2 confirmation quality leg: full sft_v3 authority + fresh disjoint noop draw.

Registered statistics live in docs/validation/2026-09-10-s2-preregistration.json
(-2pp fallback band; point >= -0.5pp). Design: docs/research/2026-09-10-s2-noninferiority-design.md
sections 3-4. Successor to b4_quant_quality.py; adoption stays with the lead.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import eval_scenarios as scenario
import eval_noop_fp as noop
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write

ARMS = ['Q8_0', 'Q4_K_M_imatrix']
PORT = 18478
SCENARIO_CAP = 250          # cap raise: rename contributes all 202 rows
CORPUS_SEED = 20260910      # NEW corpus seed (!= pilot 20260820)
BUILD_SEED = 20260911       # NEW noop build seed (!= pilot 7)
FRESH_CORPUS_N = 58         # fresh corpus functions -> >=250 cases (54 gave 242 on the live corpus;
                           # the registered target is the CASE count, not the function count)
AUTHORED_N = 24             # pilot AUTHORED sources excluded from the draw
Z_ONE_SIDED_95 = 1.6448536269514722
PREREG = Path(__file__).resolve().parents[2] / 'docs/validation/2026-09-10-s2-preregistration.json'
COHORT_SHAPE = dict(scenario_total=307, rename_total=202, fresh_scenario_ids=52,
                    fresh_corpus_n=FRESH_CORPUS_N, min_noop_cases=250,
                    pilot_scenario=255, pilot_noop=258, pilot_corpus=30)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def lower_bound(result):
    """One-sided 95% lower bound for the paired difference d = p_cand - p_base.

    Design section 1: SE = sqrt((p10 + p01 - d_hat^2)/n) with p10 = losses/n,
    p01 = gains/n, d_hat = (gains - losses)/n. Returns pp-scaled fields.
    """
    n = result['n']
    p10, p01 = result['losses'] / n, result['gains'] / n
    d = p01 - p10
    se = math.sqrt(max(0.0, p10 + p01 - d * d) / n)
    return dict(n=n, delta_pp=100 * d, se_pp=100 * se,
                lb_pp=100 * (d - Z_ONE_SIDED_95 * se), z=Z_ONE_SIDED_95)


def paired(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError('Invalid paired cohort')
    loss = sum(a and not b for a, b in zip(base, candidate))
    gain = sum(b and not a for a, b in zip(base, candidate))
    d = loss + gain
    p = min(1, 2 * sum(math.comb(d, k) for k in range(min(loss, gain) + 1)) / 2 ** d) if d else 1
    return dict(n=len(base), baseline_positive=sum(base), candidate_positive=sum(candidate),
                losses=loss, gains=gain, delta_pp=100 * (gain - loss) / len(base),
                mcnemar_exact_two_sided=p)


def registered_margins(path):
    """Read the registered exact band from the preregistration json (never assume it)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    text = doc['registered']
    lb = re.search(r'LB\s*>=\s*-(\d+(?:\.\d+)?)\s*pp', text)
    point = re.search(r'point estimate\s*>=\s*-(\d+(?:\.\d+)?)\s*pp', text)
    valid = re.search(r'valid\s*>=\s*-(\d+(?:\.\d+)?)\s*pp', doc.get('rationale', ''))
    if not (lb and point and valid):
        raise ValueError('Registered S2 band not found in ' + str(path))
    return dict(exact_lb_margin_pp=-float(lb.group(1)), exact_point_margin_pp=-float(point.group(1)),
                valid_lb_margin_pp=-float(valid.group(1)), registered=text,
                source=str(path), sha256=sha256_bytes(path.read_bytes()))


def load_pilot(assets, shape=COHORT_SHAPE):
    rows = read_rows(assets / 'cases.jsonl')
    selection = json.loads((assets / 'selection.json').read_text())
    scenario_rows = [r for r in rows if r['kind'] == 'scenario']
    noop_rows = [r for r in rows if r['kind'] == 'noop']
    if len(scenario_rows) != shape['pilot_scenario'] or len(noop_rows) != shape['pilot_noop']:
        raise ValueError('Require the frozen pilot cohort: '
                         f'{shape["pilot_scenario"]} scenario + {shape["pilot_noop"]} noop')
    if len({(r['kind'], r['id']) for r in rows}) != len(rows):
        raise ValueError('Pilot cohort identity drift')
    if selection.get('corpus_seed') == CORPUS_SEED or selection.get('noop_seed') == BUILD_SEED:
        raise ValueError('Confirmation seeds must differ from the pilot seeds')
    functions = selection.get('corpus_functions') or []
    keys = {(f['rel_path'], f['fn']) for f in functions}
    if len(keys) != shape['pilot_corpus']:
        raise ValueError(f'Require the {shape["pilot_corpus"]} recorded pilot corpus functions')
    return dict(scenario_ids={r['id'] for r in scenario_rows},
                noop_ids={(r['case_kind'], r['id']) for r in noop_rows},
                function_keys=keys)


def build_scenario_rows(shape):
    selected, report = scenario.load_heldout(SCENARIO_CAP)
    rows = []
    for family in scenario.FAMILIES:
        for r in selected.get(family, []):
            scenario.scenarios.validate_example(r['scenario'] if 'scenario' in r else r)
            rows.append(dict(kind='scenario', id=hashlib.sha1(r['_prompt'].encode()).hexdigest()[:12],
                             prompt=r['_prompt'], scenario=r['scenario'] if 'scenario' in r else r))
    total = sum(f['scored'] for f in report.values())
    if (set(report) != set(scenario.FAMILIES) or total != shape['scenario_total']
            or report['rename_propagation']['scored'] != shape['rename_total']):
        raise ValueError(f'Scenario authority changed: {report}')
    return rows, report


def draw_fresh_corpus(original, want, seed, pilot_keys):
    """Draw `want` corpus functions with a NEW seed, skipping pilot functions."""
    drawn = original(want + len(pilot_keys), seed)
    fresh = [f for f in drawn if (f['rel_path'], f['fn']) not in pilot_keys]
    if len(fresh) < want:
        raise ValueError(f'Corpus universe exhausted: {len(fresh)} fresh < {want}')
    return fresh[:want]


def build_noop_rows(pilot, shape):
    """Fresh disjoint noop cohort: new seeds, pilot functions skipped, AUTHORED excluded."""
    raw_files = {}
    original_read = Path.read_bytes
    original_corpus = noop.corpus_functions
    drawn = []

    def recorded_read(path):
        value = original_read(path)
        if path.is_relative_to(noop.NORMALIZED):
            raw_files[str(path)] = value
        return value

    def draw(want):
        kept = draw_fresh_corpus(original_corpus, want, CORPUS_SEED, pilot['function_keys'])
        drawn.extend(kept)
        return kept

    Path.read_bytes = recorded_read
    noop.corpus_functions = draw
    try:
        made = noop.build_cases(shape['fresh_corpus_n'], BUILD_SEED)
    finally:
        Path.read_bytes = original_read
        noop.corpus_functions = original_corpus
    corpus_cases = [c for c in made if not c.rel_path.startswith('authored/')]
    rows = []
    for c in corpus_cases:
        prompt, truncated = noop.build_prompt(c.lines, c.cursor_line, c.cursor_char, c.rel_path)
        rows.append(dict(kind='noop', case_kind=c.kind, id=c.id, prompt=prompt, truncated_prefix=truncated,
                         **{key: getattr(c, key) for key in c.__slots__ if key not in ('id', 'kind')}))
    overlap = pilot['noop_ids'] & {(r['case_kind'], r['id']) for r in rows}
    if overlap:
        raise ValueError('Fresh noop cohort collides with pilot (kind,id): ' + str(sorted(overlap)[:5]))
    if len(rows) < shape['min_noop_cases']:
        raise ValueError(f'Fresh noop draw below target: {len(rows)} < {shape["min_noop_cases"]}')
    return rows, drawn, raw_files


def prepare(run, assets, prereg=PREREG, shape=COHORT_SHAPE):
    pilot = load_pilot(assets, shape)
    margins = registered_margins(prereg)
    for arm in ARMS:
        if not (assets / (arm + '.gguf')).is_file():
            raise ValueError('Missing frozen model: ' + arm)
    sources = assets / 'selection-sources'
    scenario.SCEN_DIR = sources
    scenario.HOLDOUT_REF = sources / 'holdout.jsonl'
    noop.FINISH_SRC = sources / 'finish_block_sample.jsonl'
    scenario_rows, report = build_scenario_rows(shape)
    new_ids = {r['id'] for r in scenario_rows}
    if not pilot['scenario_ids'] <= new_ids:
        raise ValueError('Pilot scenario rows are not a subset of the authority')
    replication_ids = pilot['scenario_ids'] & new_ids
    fresh_ids = new_ids - pilot['scenario_ids']
    if len(fresh_ids) != shape['fresh_scenario_ids']:
        raise ValueError(f'Fresh scenario subset changed: {len(fresh_ids)} != {shape["fresh_scenario_ids"]}')
    noop_rows, drawn, raw_files = build_noop_rows(pilot, shape)
    rows = scenario_rows + noop_rows
    if len({(r['kind'], r['id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate case identity in the confirmation cohort')
    (run / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    source_record = []
    for i, (path, value) in enumerate(sorted(raw_files.items())):
        name = f'normalized-{i:04d}.R'
        (run / name).write_bytes(value)
        source_record.append(dict(source=path, file=name, sha256=sha256_bytes(value)))
    frozen = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(sources.glob('*.jsonl'))}
    selection = dict(scenarios=report, noop_cases=len(noop_rows), noop_seed=BUILD_SEED,
        corpus_seed=CORPUS_SEED, corpus_functions=drawn, authored_excluded=AUTHORED_N,
        disjointness=dict(pilot_corpus_functions_skipped=len(pilot['function_keys']),
            case_overlap=[], scenario_pilot_ids_included=len(replication_ids),
            scenario_fresh_ids=len(fresh_ids), noop_pilot_overlap=[]),
        source_reads=source_record, frozen_source_hashes=frozen,
        boundary='Fresh disjoint confirmation cohort frozen once; pilot cohort is never resumed')
    write(run / 'selection.json', selection)
    write(run / 'prepared.json', dict(arms=ARMS, case_ids=[[r['kind'], r['id']] for r in rows],
        expected_rows=2 * len(rows), ctx=8192, max_seconds=5370,
        replication_ids=sorted(replication_ids), fresh_ids=sorted(fresh_ids),
        preregistration=dict(source=margins['source'], sha256=margins['sha256']),
        scope='S2 confirmation: full sft_v3 eval authority + fresh disjoint noop draw; not causal proof of imatrix benefit',
        gates=f'Registered: exact one-sided 95% LB >= {margins["exact_lb_margin_pp"]}pp AND point >= '
              f'{margins["exact_point_margin_pp"]}pp; fresh subset point >= -2.0pp; valid LB >= '
              f'{margins["valid_lb_margin_pp"]}pp; noop fresh losses <= gains; replication 1:1',
        calibration='Existing imatrix covers only 8x512 chunks; no release-grade coverage claim'))
    print(json.dumps(dict(scenarios=len(scenario_rows), noop=len(noop_rows), rows=len(rows))))


def complete(port, prompt, limit, stops):
    body = json.dumps(dict(prompt=prompt, max_tokens=limit, temperature=0, stop=stops,
                           stream=False)).encode()
    import urllib.request
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    return data, time.monotonic() - started


def score_row(r, data):
    choice = data['choices'][0]
    rec = dict(arm=None, kind=r['kind'], id=r['id'], response=data, at=time.time())
    if r['kind'] == 'scenario':
        pred = scenario.parse_pred('zeta2', choice['text'])
        gt = [l.rstrip() for l in r['scenario']['region_new']]
        while gt and not gt[-1]:
            gt.pop()
        passed, kind, reason = scenario.validator_verdict(r['scenario'], pred)
        rec.update(exact=pred == gt, valid_pass=passed, fail_kind=kind, valid_reason=reason,
                   family=r['scenario']['family'])
    else:
        pred = noop.parse_prediction(choice['text'])
        rec.update(proposal=bool(pred), expectation=r['expectation'], cls=r['cls'])
    return rec


def measure(run, assets):
    p = json.loads((run / 'prepared.json').read_text())
    rows = read_rows(run / 'cases.jsonl')
    counts = {}
    server = None

    def expire(*_):
        raise DeadlineExceeded('S2 quality exceeded its execution bound')

    old = signal.signal(signal.SIGALRM, expire)
    deadline = time.monotonic() + p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run / 'requests.jsonl').open('x') as out:
            for arm in p['arms']:
                server = GpuServer(assets / (arm + '.gguf'), PORT, ['-lv', '4', '--seed', '20260905'],
                    ctx=p['ctx'], server=Path(os.environ['S1_RUNTIME']) / 'llama-server',
                    foreground=True, log_path=run / ('server-' + arm + '.log'))
                try:
                    server.start(ready_timeout=180)
                    check_offload(server.log_path.read_text())
                    current = {}
                    for r in rows:
                        key = (r['kind'], r['id'])
                        n = tokenize(server.port, r['prompt'])
                        limit = 640 if r['kind'] == 'scenario' else 320
                        if n + limit + 16 > p['ctx']:
                            raise ValueError('Prompt exceeds context: ' + str(key))
                        if key in counts and n != counts[key]:
                            raise ValueError('Cross-arm tokenizer mismatch')
                        current[key] = n
                    if not counts:
                        counts = current
                    with (run / 'token-audit.jsonl').open('a') as audit:
                        for (kind, identity), n in current.items():
                            audit.write(json.dumps(dict(arm=arm, kind=kind, id=identity, tokens=n)) + '\n')
                    for r in rows:
                        key = (r['kind'], r['id'])
                        limit = 640 if r['kind'] == 'scenario' else 320
                        stops = [scenario.STOP] if r['kind'] == 'scenario' else noop.EXT_STOPS
                        data, wall = complete(server.port, r['prompt'], limit, stops)
                        rec = score_row(r, data)
                        rec.update(arm=arm, tokenized_prompt_tokens=current[key],
                                   served_prompt_tokens=data.get('usage', {}).get('prompt_tokens'),
                                   wall_s=wall)
                        choice = data['choices'][0]
                        if rec['served_prompt_tokens'] != current[key] or choice.get('finish_reason') not in ('stop', 'length'):
                            write(run / 'invalid-response.json', rec)
                            raise ValueError('Invalid quality response')
                        out.write(json.dumps(rec, allow_nan=False) + '\n')
                        out.flush()
                    print(json.dumps(dict(arm=arm, rows=len(rows))), flush=True)
                finally:
                    signal.alarm(0)
                    server.stop()
                    server = None
                    signal.alarm(max(1, math.ceil(deadline - time.monotonic())))
    finally:
        signal.alarm(0)
        try:
            if server is not None:
                server.stop()
        finally:
            signal.signal(signal.SIGALRM, old)


def pilot_verdicts(assets, arms, replication_ids):
    path = assets / 'pilot_requests.jsonl'
    if not path.is_file():
        raise ValueError('Missing pilot requests for the replication-integrity subset')
    verdicts = {arm: {} for arm in arms}
    for r in read_rows(path):
        if r.get('kind') == 'scenario' and r.get('arm') in verdicts:
            verdicts[r['arm']][r['id']] = (bool(r.get('exact')), bool(r.get('valid_pass')))
    for arm in arms:
        missing = replication_ids - set(verdicts[arm])
        if missing:
            raise ValueError(f'Pilot replication requests incomplete for {arm}: {len(missing)} missing')
    return verdicts


def evaluate(run, assets, prereg=PREREG):
    p = json.loads((run / 'prepared.json').read_text())
    rows = read_rows(run / 'requests.jsonl')
    if [(r['arm'], r['kind'], r['id']) for r in rows] != [(a, k, i) for a in p['arms'] for k, i in p['case_ids']]:
        raise ValueError('Incomplete, duplicate or reordered quality coverage')
    if any(r.get('error') or r['served_prompt_tokens'] != r['tokenized_prompt_tokens'] for r in rows):
        raise ValueError('Invalid quality request')
    frozen = p.get('preregistration') or {}
    source = Path(frozen.get('source') or prereg)
    margins = registered_margins(source)
    if frozen.get('sha256') and frozen['sha256'] != margins['sha256']:
        raise ValueError('Preregistration changed after cohort freeze')
    replication_ids = set(p['replication_ids'])
    fresh_ids = set(p['fresh_ids'])
    by_arm = {arm: [r for r in rows if r['arm'] == arm] for arm in p['arms']}
    summary = {}
    for arm, rs in by_arm.items():
        sc = [r for r in rs if r['kind'] == 'scenario']
        no = [r for r in rs if r['kind'] == 'noop' and r['expectation'] == 'no_proposal']
        summary[arm] = dict(scenarios=len(sc), exact=sum(r['exact'] for r in sc),
            valid=sum(r['valid_pass'] for r in sc), noop_scored=len(no),
            noop_false_suggestions=sum(r['proposal'] for r in no),
            generation_limit_rows=sum(r['response']['choices'][0]['finish_reason'] == 'length' for r in rs))
        summary[arm]['scenario_by_family'] = {family: dict(
            n=sum(r['family'] == family for r in sc),
            exact=sum(r['exact'] for r in sc if r['family'] == family),
            valid=sum(r['valid_pass'] for r in sc if r['family'] == family))
            for family in sorted({r['family'] for r in sc})}
        summary[arm]['noop_by_class'] = {cls: dict(
            n=sum(r['cls'] == cls for r in rs if r['kind'] == 'noop'),
            proposals=sum(r['proposal'] for r in rs if r['kind'] == 'noop' and r['cls'] == cls))
            for cls in sorted({r['cls'] for r in rs if r['kind'] == 'noop'})}
    base = by_arm[p['arms'][0]]
    candidate = by_arm[p['arms'][1]]
    base_sc = [r for r in base if r['kind'] == 'scenario']
    cand_sc = [r for r in candidate if r['kind'] == 'scenario']
    base_no = [r for r in base if r['kind'] == 'noop' and r['expectation'] == 'no_proposal']
    cand_no = [r for r in candidate if r['kind'] == 'noop' and r['expectation'] == 'no_proposal']
    exact = paired([r['exact'] for r in base_sc], [r['exact'] for r in cand_sc])
    valid = paired([r['valid_pass'] for r in base_sc], [r['valid_pass'] for r in cand_sc])
    noop_restraint = paired([not r['proposal'] for r in base_no], [not r['proposal'] for r in cand_no])
    comparisons = {p['arms'][1]: dict(exact=exact, valid=valid, noop_restraint=noop_restraint,
        exact_lower_bound=lower_bound(exact), valid_lower_bound=lower_bound(valid),
        noop_rule_of_three_pp=300 / len(base_no) if base_no else None)}
    comparisons[p['arms'][1]]['exact_by_family'] = {}
    for family in sorted({r['family'] for r in base_sc}):
        b = [r['exact'] for r in base_sc if r['family'] == family]
        c = [r['exact'] for r in cand_sc if r['family'] == family]
        result = paired(b, c)
        result['lower_bound'] = lower_bound(result)
        comparisons[p['arms'][1]]['exact_by_family'][family] = result
    pilot = pilot_verdicts(assets, p['arms'], replication_ids)
    replication = {}
    for arm, rs in by_arm.items():
        sc = {r['id']: r for r in rs if r['kind'] == 'scenario'}
        exact_flips = sorted(i for i in replication_ids if sc[i]['exact'] != pilot[arm][i][0])
        valid_flips = sorted(i for i in replication_ids if sc[i]['valid_pass'] != pilot[arm][i][1])
        replication[arm] = dict(n=len(replication_ids), exact_flips=exact_flips, valid_flips=valid_flips,
                                verdict_match=not exact_flips and not valid_flips)
    fresh_base = [r for r in base_sc if r['id'] in fresh_ids]
    fresh_cand = [r for r in cand_sc if r['id'] in fresh_ids]
    fresh52 = paired([r['exact'] for r in fresh_base], [r['exact'] for r in fresh_cand])
    fresh52['lower_bound'] = lower_bound(fresh52)
    gates = dict(
        exact_lb_ge_registered=comparisons[p['arms'][1]]['exact_lower_bound']['lb_pp'] >= margins['exact_lb_margin_pp'],
        exact_point_ge_registered=exact['delta_pp'] >= margins['exact_point_margin_pp'],
        fresh_subset_point_ge_minus_2pp=fresh52['delta_pp'] >= -2.0,
        valid_lb_ge_registered=comparisons[p['arms'][1]]['valid_lower_bound']['lb_pp'] >= margins['valid_lb_margin_pp'],
        noop_fresh_losses_le_gains=noop_restraint['losses'] <= noop_restraint['gains'],
        replication_1to1=all(v['verdict_match'] for v in replication.values()))
    gates['all_quality_gates_pass'] = all(gates.values())
    write(run / 'evaluation.json', dict(rows=len(rows), arms=summary, paired=comparisons,
        subsets=dict(replication=replication, fresh52=fresh52,
                     note='Fresh-only coverage is family-degenerate (rename); the registered analysis pools the authority'),
        registered_gates=gates, registered_band=margins,
        inference_boundary='Paired p-values are unadjusted descriptive tests; the noninferiority decision uses the registered one-sided lower bound. Package dependence remains.'))
    write(run / 'verdict.json', dict(verdict='S2-CONFIRMATION-QUALITY-MEASURED', adoption='NOT-ASSESSED',
        boundary='Quality leg only. Intent judge and CPU-cycle legs gate separately; promotion is the lead decision. '
                 'Existing imatrix coverage is only 8x512 chunks. Any replication flip is runtime drift: investigate before unblinding.'))


def smoke(run, assets):
    """CPU-only wiring smoke on synthetic tiny data; no models, no GPU, no network."""
    na = dict(family='na_rm_propagation', package='pkg', path='R/a.R', prefix=['f <- function(x) {'],
        region_old=['  mean(x)'], region_new=['  mean(x, na.rm = TRUE)'], cursor_idx=2, note='smoke',
        event_diff='User edited "R/a.R":\n\n```diff\n@@ -2 +2 @@\n-  mean(x)\n+  mean(x, na.rm = TRUE)\n```')
    ren = dict(family='rename_propagation', package='pkg', path='R/b.R', prefix=['# header'],
        region_old=['  total <- sum(x)'], region_new=['  total <- sum(value)'], cursor_idx=2, note='smoke',
        event_diff='User edited "R/b.R":\n\n```diff\n@@ -2 +2 @@\n-  total <- sum(x)\n+  total <- sum(value)\n```')
    pipe = dict(family='pipe_rewrite', package='pkg', path='R/c.R', prefix=['# c'],
        region_old=['  x %>% head()'], region_new=['  x |> head()'], cursor_idx=0, note='smoke',
        event_diff='User edited "R/c.R":\n\n```diff\n@@ -1 +1 @@\n-  x %>% head()\n+  x |> head()\n```')
    heldout = [dict(_prompt='P:' + ex['family'], scenario=ex) for ex in (na, ren, pipe)]
    report = {fam: dict(held_out=0, scored=0, capped_away=0) for fam in scenario.FAMILIES}
    for ex in (na, ren, pipe):
        report[ex['family']] = dict(held_out=1, scored=1, capped_away=0)
    (assets / 'selection-sources').mkdir(parents=True, exist_ok=True)
    (assets / 'selection-sources' / 'holdout.jsonl').write_text('{}\n')
    for fam in report:
        (assets / 'selection-sources' / (fam + '.jsonl')).write_text('{}\n')

    def fake_load_heldout(cap):
        return {fam: [r for r in heldout if r['scenario']['family'] == fam]
                for fam in scenario.FAMILIES}, report

    def fake_corpus(want, seed=None):
        assert seed == CORPUS_SEED
        return [dict(rel_path='synthetic/pilot.R', lines=['pilot <- function(x) {', '  x', '}'],
                     fn='pilot', r0=0, r1=2, package='synthetic'),
                dict(rel_path='synthetic/one.R', lines=['one <- function(x) {', '  x', '  y', '}'],
                     fn='one', r0=0, r1=3, package='synthetic'),
                dict(rel_path='synthetic/two.R', lines=['two <- function(x) {', '  x', '  y', '}'],
                     fn='two', r0=0, r1=3, package='synthetic')][:want]

    smoke_ids = [sha256_bytes(r['_prompt'].encode())[:12] for r in heldout]
    for arm in ARMS:
        (assets / (arm + '.gguf')).write_bytes(b'smoke')
    pilot_cases = [dict(kind='scenario', id=smoke_ids[0], prompt='P:' + na['family'], scenario=na),
                   dict(kind='scenario', id=smoke_ids[1], prompt='P:' + ren['family'], scenario=ren),
                   dict(kind='noop', case_kind='after_close_brace', id='pilotnoop', prompt='PN',
                        cls='a_after_close_brace', expectation='no_proposal')]
    (assets / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pilot_cases))
    (assets / 'selection.json').write_text(json.dumps(dict(corpus_seed=20260820, noop_seed=7,
        corpus_functions=[dict(rel_path='synthetic/pilot.R', fn='pilot')])))
    shape = dict(scenario_total=3, rename_total=1, fresh_scenario_ids=1, fresh_corpus_n=2, min_noop_cases=2,
                 pilot_scenario=2, pilot_noop=1, pilot_corpus=1)
    saved = (scenario.load_heldout, noop.corpus_functions, GpuServer, check_offload, tokenize, complete)

    class FakeServer:
        def __init__(self, model, port, flags, **kw):
            self.port, self.model = port, str(model)
            self.log_path = run / 'server.log'
            self.log_path.write_text('smoke server log\n')
        def start(self, ready_timeout=0):
            return True
        def stop(self):
            return None

    tokens = {r['_prompt']: 40 + i for i, r in enumerate(heldout)}
    texts = {}
    for r in heldout:
        gt = [l.rstrip() for l in r['scenario']['region_new']]
        texts[r['_prompt']] = '\n'.join(gt) + '\n>>>>>>> UPDATED'

    def fake_complete(port, prompt, limit, stops):
        data = dict(usage=dict(prompt_tokens=tokens.get(prompt, 60)),
                    choices=[dict(finish_reason='stop', text=texts.get(prompt, ''))])
        return data, 0.01

    def fake_tokenize(port, prompt):
        return tokens.get(prompt, 60)

    try:
        scenario.load_heldout = fake_load_heldout
        noop.corpus_functions = fake_corpus
        globals()['GpuServer'] = FakeServer
        globals()['check_offload'] = lambda log: None
        globals()['tokenize'] = fake_tokenize
        globals()['complete'] = fake_complete
        prepare(run, assets, shape=shape)
        pilot_requests = []
        for arm in ARMS:
            for sid in smoke_ids[:2]:
                pilot_requests.append(dict(arm=arm, kind='scenario', id=sid, exact=True, valid_pass=True))
        (assets / 'pilot_requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pilot_requests))
        saved_runtime = os.environ.get('S1_RUNTIME')
        os.environ['S1_RUNTIME'] = str(assets)
        try:
            measure(run, assets)
        finally:
            if saved_runtime is None:
                os.environ.pop('S1_RUNTIME', None)
            else:
                os.environ['S1_RUNTIME'] = saved_runtime
        evaluate(run, assets)
    finally:
        scenario.load_heldout, noop.corpus_functions = saved[0], saved[1]
        globals()['GpuServer'], globals()['check_offload'] = saved[2], saved[3]
        globals()['tokenize'], globals()['complete'] = saved[4], saved[5]
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert gates['all_quality_gates_pass'], gates
    assert (run / 'cases.jsonl').is_file() and (run / 'selection.json').is_file()
    print('SMOKE-OK ' + json.dumps(gates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'measure', 'evaluate', 'smoke'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prereg', type=Path, default=PREREG)
    args = parser.parse_args()
    if args.action == 'evaluate':
        evaluate(args.run, args.assets, args.prereg)
    elif args.action == 'smoke':
        smoke(args.run, args.assets)
    else:
        globals()[args.action](args.run, args.assets)
