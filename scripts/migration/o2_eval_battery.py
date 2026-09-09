#!/usr/bin/env python3
"""O2 eval battery: filtered vs unfiltered GRPO arms on the frozen 513-case
cohort; paired McNemar control-vs-filtered. WINNER-FILTER condition 2 leg
(condition 1 already met: first50 psg 0.660 vs 0.3175)."""
import argparse, json, math, os, signal, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import eval_scenarios as scenario
import eval_noop_fp as noop
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write

ARMS = ['control', 'filtered']
ALPHA = {'control': None, 'filtered': None}


def cases(assets):
    return [json.loads(l) for l in (assets / 'cases.jsonl').read_text().splitlines()]


def prepare(run, assets):
    rows = cases(assets)
    if len(rows) != 513 or len({(r['kind'], r['id']) for r in rows}) != 513:
        raise ValueError('Require 255 scenario and 258 unique no-op cases')
    if sum(r['kind'] == 'scenario' for r in rows) != 255:
        raise ValueError('Scenario cohort changed')
    for r in rows:
        if r['kind'] == 'scenario':
            scenario.scenarios.validate_example(r['scenario'])
        elif r['kind'] != 'noop':
            raise ValueError('Unknown case kind')
    import hashlib
    def digest(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                h.update(b)
            return h.hexdigest()
    import hashlib as _h2
    def _dg(p):
        hh = _h2.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                hh.update(b)
        return hh.hexdigest()
    model_sha = {arm: _dg(assets / (arm + '_Q8_0.gguf')) for arm in ARMS}
    noop_split = {}
    for r in rows:
        if r['kind'] == 'noop':
            noop_split[r['expectation']] = noop_split.get(r['expectation'], 0) + 1
    if noop_split.get('no_proposal') != 204:
        raise ValueError('Expected 204 no_proposal noop cases, got %r' % noop_split)
    # cross-arm GGUF byte-size parity as a light identity check
    sizes = {(assets / (arm + '_Q8_0.gguf')).stat().st_size for arm in ARMS}
    if len(sizes) != 1:
        raise ValueError('arm GGUF sizes differ: %r' % sizes)
    write(run / 'prepared.json', dict(
        arms=ARMS, model_sha256=model_sha, cases_sha256=digest(assets / 'cases.jsonl'),
        noop_expectation_split=noop_split,
        case_ids=[[r['kind'], r['id']] for r in rows],
        expected_rows=513 * len(ARMS), ctx=8192, max_seconds=3600,
        scope='WiSE-FT alpha dose-response on the resolved b4 base pick; '
              'within-run paired comparisons vs the b4 anchor only',
        gates='Pre-registered: report exact/valid/noopFP per alpha with McNemar vs b4. '
              'Adoption (runbook B10: any alpha dominating the product axis) is a separate '
              'review decision, never automatic.',
        rationale='PFT1 found b4 LoRA degraded base general-text BPB +20.3% '
                  '(domain-asymmetric forgetting); alpha<1 may recover out-of-domain quality. '
                  'Also pre-registers the RL-phase guardrail instrument '
                  '(blend RL checkpoints back toward SFT init).'))


def measure(run, assets):
    p = json.loads((run / 'prepared.json').read_text()); rows = cases(assets); counts = {}; server = None
    def expire(*_):
        raise DeadlineExceeded('B10 battery exceeded 90 minute bound')
    old = signal.signal(signal.SIGALRM, expire); deadline = time.monotonic() + p['max_seconds']; signal.alarm(p['max_seconds'])
    try:
        with (run / 'requests.jsonl').open('x') as out:
            import hashlib as _h
            def _digest(path):
                h = _h.sha256()
                with open(path, 'rb') as f:
                    for b in iter(lambda: f.read(8 << 20), b''):
                        h.update(b)
                return h.hexdigest()
            for arm in p['arms']:
                model_path = assets / (arm + '_Q8_0.gguf')
                if p.get('model_sha256') and _digest(model_path) != p['model_sha256'][arm]:
                    raise ValueError('Model hash mismatch before serving: ' + arm)
                server = GpuServer(model_path, 18478, ['-lv', '4', '--seed', '20260905'], ctx=p['ctx'],
                                   server=Path(os.environ['S1_RUNTIME']) / 'llama-server', foreground=True, log_path=run / ('server-' + arm + '.log'))
                try:
                    server.start(ready_timeout=180); check_offload(server.log_path.read_text())
                    current = {}
                    for r in rows:
                        key = (r['kind'], r['id']); n = tokenize(server.port, r['prompt']); limit = 640 if r['kind'] == 'scenario' else 320
                        if n + limit + 16 > p['ctx']:
                            raise ValueError('Prompt exceeds context: ' + str(key))
                        if key in counts and n != counts[key]:
                            raise ValueError('Cross-arm tokenization mismatch')
                        current[key] = n
                    if not counts: counts = current
                    with (run / 'token-audit.jsonl').open('a') as audit:
                        for (kind, identity), n in current.items():
                            audit.write(json.dumps(dict(arm=arm, kind=kind, id=identity, tokens=n)) + '\n')
                    for r in rows:
                        key = (r['kind'], r['id']); limit = 640 if r['kind'] == 'scenario' else 320
                        stops = [scenario.STOP] if r['kind'] == 'scenario' else noop.EXT_STOPS
                        body = json.dumps(dict(prompt=r['prompt'], max_tokens=limit, temperature=0, stop=stops, stream=False)).encode()
                        req = urllib.request.Request('http://127.0.0.1:%d/v1/completions' % server.port, data=body, headers={'Content-Type': 'application/json'})
                        started = time.monotonic()
                        with urllib.request.urlopen(req, timeout=120) as response:
                            data = json.load(response)
                        rec = dict(arm=arm, kind=r['kind'], id=r['id'], tokenized_prompt_tokens=current[key],
                                   served_prompt_tokens=data.get('usage', {}).get('prompt_tokens'), response=data, wall_s=time.monotonic() - started, at=time.time())
                        choice = data['choices'][0]
                        if rec['served_prompt_tokens'] != current[key] or choice.get('finish_reason') not in ('stop', 'length'):
                            write(run / 'invalid-response.json', rec); raise ValueError('Invalid quality response')
                        if r['kind'] == 'scenario':
                            pred = scenario.parse_pred('zeta2', choice['text']); gt = [l.rstrip() for l in r['scenario']['region_new']]
                            while gt and not gt[-1]:
                                gt.pop()
                            passed, kind, reason = scenario.validator_verdict(r['scenario'], pred)
                            rec.update(exact=pred == gt, valid_pass=passed, fail_kind=kind, valid_reason=reason, family=r['scenario']['family'])
                        else:
                            pred = noop.parse_prediction(choice['text']); rec.update(proposal=bool(pred), expectation=r['expectation'], cls=r['cls'])
                        out.write(json.dumps(rec, allow_nan=False) + '\n'); out.flush()
                    print(json.dumps(dict(arm=arm, rows=len(rows))), flush=True)
                finally:
                    signal.alarm(0); server.stop(); server = None; signal.alarm(max(1, math.ceil(deadline - time.monotonic())))
    finally:
        signal.alarm(0)
        try:
            if server is not None: server.stop()
        finally:
            signal.signal(signal.SIGALRM, old)


def paired(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError('Invalid paired cohort')
    loss = sum(a and not b for a, b in zip(base, candidate)); gain = sum(b and not a for a, b in zip(base, candidate)); d = loss + gain
    p = min(1, 2 * sum(math.comb(d, k) for k in range(min(loss, gain) + 1)) / 2 ** d) if d else 1
    return dict(n=len(base), baseline_positive=sum(base), candidate_positive=sum(candidate), losses=loss, gains=gain,
                delta_pp=100 * (gain - loss) / len(base), mcnemar_exact_two_sided=p)


def evaluate(run):
    p = json.loads((run / 'prepared.json').read_text()); rows = [json.loads(l) for l in (run / 'requests.jsonl').read_text().splitlines()]
    if [(r['arm'], r['kind'], r['id']) for r in rows] != [(a, k, i) for a in p['arms'] for k, i in p['case_ids']]:
        raise ValueError('Incomplete, duplicate or reordered coverage')
    if any(r.get('error') or r['served_prompt_tokens'] != r['tokenized_prompt_tokens'] for r in rows):
        raise ValueError('Invalid request')
    summary = {}; comparisons = {}
    anchor = p['arms'][0]
    for arm in p['arms']:
        rs = [r for r in rows if r['arm'] == arm]; sc = [r for r in rs if r['kind'] == 'scenario']; no = [r for r in rs if r['kind'] == 'noop' and r['expectation'] == 'no_proposal']
        summary[arm] = dict( scenarios=len(sc), exact=sum(r['exact'] for r in sc), valid=sum(r['valid_pass'] for r in sc),
                            noop_scored=len(no), noop_false_suggestions=sum(r['proposal'] for r in no),
                            generation_limit_rows=sum(r['response']['choices'][0]['finish_reason'] == 'length' for r in rs))
        summary[arm]['scenario_by_family'] = {family: dict(n=sum(r['family'] == family for r in sc),
                                                           exact=sum(r['exact'] for r in sc if r['family'] == family), valid=sum(r['valid_pass'] for r in sc if r['family'] == family))
                                              for family in sorted({r.get('family', 'unknown') for r in sc})} if all('family' in r for r in sc) else {}
        if arm == anchor:
            continue
        a_sc, a_no = [r for r in rows if r['arm'] == anchor and r['kind'] == 'scenario'], [r for r in rows if r['arm'] == anchor and r['kind'] == 'noop' and r['expectation'] == 'no_proposal']
        comparisons[arm] = dict(exact=paired([r['exact'] for r in a_sc], [r['exact'] for r in sc]),
                                valid=paired([r['valid_pass'] for r in a_sc], [r['valid_pass'] for r in sc]),
                                noop_restraint=paired([not r['proposal'] for r in a_no], [not r['proposal'] for r in no]))
    def wilson(k, n, z=1.96):
        if n == 0:
            return None
        ph = k / n
        d = 1 + z * z / n
        c = ph + z * z / (2 * n)
        hw = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5)
        return [round((c - hw) / d, 4), round((c + hw) / d, 4)]
    arm_rows = {a: sum(1 for r in rows if r['arm'] == a) for a in p['arms']}
    for arm in p['arms']:
        n_rows = summary[arm]['scenarios']
        total_rows = arm_rows[arm]
        summary[arm]['exact_wilson95'] = wilson(summary[arm]['exact'], n_rows)
        summary[arm]['valid_wilson95'] = wilson(summary[arm]['valid'], n_rows)
        summary[arm]['length_rate'] = round(summary[arm]['generation_limit_rows'] / max(1, total_rows), 4)
    dose = [dict(arm=arm, exact=summary[arm]['exact'], valid=summary[arm]['valid'],
                 noop_false=summary[arm]['noop_false_suggestions'],
                 length_rate=summary[arm]['length_rate']) for arm in p['arms']]
    write(run / 'evaluation.json', dict(rows=len(rows), dose_response=dose, arms=summary, paired_vs_control=comparisons,
                                        statistics_note='unadjusted exact-McNemar tests (2 arms x 3 metrics), exploratory: '
                                                       'no significance voting, no p>0.05-equivalence reading, no noninferiority claim '
                                                       '(no pre-specified margin). Product-axis dominance (review-time definition): '
                                                       'an arm dominates iff exact>=anchor AND valid>=anchor AND noopFP<=anchor AND '
                                                       'strictly better on at least one of the three.',
                                        inference_boundary='Within-run paired tests vs the fresh b4 anchor only; unadjusted descriptive '
                                                           'p-values; Q8 exports from frozen f16 parents; length-rate artifacts '
                                                           'flagged per arm.'))
    write(run / 'verdict.json', dict(verdict='WISE-FT-DOSE-RESPONSE-MEASURED', adoption='NOT-ASSESSED',
                                     boundary='Runbook B10 adoption (any alpha dominating the product axis) is a review decision. '
                                              'The general-domain recovery question (BPB forgetting probe) is a separate leg. '
                                              'This run does not blend any RL artifact and does not modify banked b4 assets.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['prepare', 'measure', 'evaluate']); p.add_argument('--run', type=Path, required=True); p.add_argument('--assets', type=Path)
    a = p.parse_args(); evaluate(a.run) if a.action == 'evaluate' else globals()[a.action](a.run, a.assets)
