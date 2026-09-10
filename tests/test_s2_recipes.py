"""Unit tests + CPU mock smokes for the S2 confirmation recipes (quality, intent, timing)."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts' / 'migration'))
import s2_quality as q
import s2_intent as i
import s2_timing as t


# --------------------------------------------------------------------------
# registered band / LB formula
# --------------------------------------------------------------------------

def test_lb_formula_matches_the_design():
    # Design section 1: SE = sqrt((p10 + p01 - d_hat^2)/n), one-sided 95% LB.
    result = q.paired([True] * 6 + [False] * 4, [True] * 4 + [False] * 6)
    # 6 losses, 0 gains? base: 6 True then 4 False; cand: 4 True then 6 False
    # losses = base and not cand among first 6 = 2? compute directly instead:
    n = result['n']
    p10, p01 = result['losses'] / n, result['gains'] / n
    d = p01 - p10
    expected_se = math.sqrt((p10 + p01 - d * d) / n)
    lb = q.lower_bound(result)
    assert lb['delta_pp'] == pytest.approx(result['delta_pp'])
    assert lb['se_pp'] == pytest.approx(100 * expected_se)
    assert lb['lb_pp'] == pytest.approx(100 * (d - q.Z_ONE_SIDED_95 * expected_se))


def test_lb_zero_discordance_is_zero_width():
    result = q.paired([True, False, True], [True, False, True])
    lb = q.lower_bound(result)
    assert lb['se_pp'] == 0 and lb['lb_pp'] == lb['delta_pp'] == 0


def test_lb_sign_follows_losses_and_gains():
    loss = q.lower_bound(q.paired([True] * 10, [True] * 5 + [False] * 5))
    gain = q.lower_bound(q.paired([True] * 5 + [False] * 5, [True] * 10))
    assert loss['delta_pp'] == pytest.approx(-50) and gain['delta_pp'] == pytest.approx(50)
    assert loss['lb_pp'] < loss['delta_pp'] < 0 < gain['lb_pp'] < gain['delta_pp']


def test_registered_margins_parse_from_the_frozen_preregistration():
    margins = q.registered_margins(q.PREREG)
    assert margins['exact_lb_margin_pp'] == -2.0
    assert margins['exact_point_margin_pp'] == -0.5
    assert margins['valid_lb_margin_pp'] == -2.0
    assert margins['sha256']


def test_registered_margins_reject_unregistered_text(tmp_path):
    path = tmp_path / 'prereg.json'
    path.write_text(json.dumps(dict(registered='we will decide later', rationale='')))
    with pytest.raises(ValueError, match='not found'):
        q.registered_margins(path)
    with pytest.raises(ValueError, match='intent gate'):
        i.registered_margins(path)
    with pytest.raises(ValueError, match='cycle gates'):
        t.registered_margins(path)


def test_timing_registered_margins_parse():
    margins = t.registered_margins(t.PREREG)
    assert margins['speedup_min'] == 1.15
    assert margins['drift_max_pct'] == 3.0


# --------------------------------------------------------------------------
# cohort selection logic (mock corpus)
# --------------------------------------------------------------------------

def corpus_entry(name):
    return dict(rel_path=f'synthetic/{name}.R', fn=name, package='synthetic',
                lines=[f'{name} <- function(x) {{', '  x', '  y', '  z', '  w', '}'],
                r0=0, r1=5)


def test_draw_fresh_corpus_skips_pilot_functions_and_slices():
    pilot_keys = {('synthetic/pilot.R', 'pilot')}
    universe = [corpus_entry('pilot')] + [corpus_entry(f'f{i}') for i in range(6)]
    seen = {}

    def original(want, seed):
        seen['want'], seen['seed'] = want, seed
        return universe[:want]

    fresh = q.draw_fresh_corpus(original, 4, q.CORPUS_SEED, pilot_keys)
    assert [f['fn'] for f in fresh] == ['f0', 'f1', 'f2', 'f3']
    assert seen['seed'] == q.CORPUS_SEED != 20260820
    assert seen['want'] == 4 + 1  # pilot key count inflates the request


def test_draw_fresh_corpus_raises_when_universe_exhausted():
    def original(want, seed):
        return [corpus_entry('only')]

    with pytest.raises(ValueError, match='exhausted'):
        q.draw_fresh_corpus(original, 2, q.CORPUS_SEED, set())


def event_diff(path, old, new):
    return f'User edited "{path}":\n\n```diff\n@@ -1 +1 @@\n-{old}\n+{new}\n```'


def na_rm_ex(old='  mean(x)', new='  mean(x, na.rm = TRUE)', path='R/a.R'):
    return dict(family='na_rm_propagation', package='pkg', path=path, prefix=['# p'],
        region_old=[old], region_new=[new], cursor_idx=0, note='test',
        event_diff=event_diff(path, old, new))


def rename_ex(path='R/r.R'):
    return dict(family='rename_propagation', package='pkg', path=path, prefix=['# p'],
        region_old=['  total <- sum(x)'], region_new=['  total <- sum(value)'], cursor_idx=0,
        note='test', event_diff=event_diff(path, '  total <- sum(x)', '  total <- sum(value)'))


def mock_corpus_universe(names):
    return [corpus_entry(n) for n in names]


def frozen_assets(tmp_path, pilot_fns=('pilot',), pilot_noop=('after_close_brace', 'pilotnoop')):
    assets = tmp_path / 'assets'
    sources = assets / 'selection-sources'
    sources.mkdir(parents=True)
    (sources / 'holdout.jsonl').write_text('{}\n')
    for fam in ('rename_propagation', 'pipe_rewrite', 'format_propagation', 'doc_sync', 'na_rm_propagation'):
        (sources / (fam + '.jsonl')).write_text('{}\n')
    pilot_cases = [
        dict(kind='scenario', id=q.sha1_id(b'P:rename'), prompt='P:rename', scenario=rename_ex()),
        dict(kind='noop', case_kind=pilot_noop[0], id=pilot_noop[1], prompt='PN', cls='a_after_close_brace',
             expectation='no_proposal')]
    (assets / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pilot_cases))
    (assets / 'selection.json').write_text(json.dumps(dict(corpus_seed=20260820, noop_seed=7,
        corpus_functions=[dict(rel_path=f'synthetic/{fn}.R', fn=fn) for fn in pilot_fns])))
    for arm in q.ARMS:
        (assets / (arm + '.gguf')).write_bytes(b'test')
    return assets


def test_prepare_freezes_a_disjoint_cohort(tmp_path, monkeypatch):
    run, assets = tmp_path / 'run', frozen_assets(tmp_path)
    run.mkdir()
    heldout = [
        dict(_prompt='P:rename', scenario=rename_ex()),
        dict(_prompt='P:na', scenario=na_rm_ex()),
        dict(_prompt='P:na2', scenario=na_rm_ex(old='  sd(x)', new='  sd(x, na.rm = TRUE)', path='R/a2.R'))]
    report = {fam: dict(held_out=0, scored=0, capped_away=0) for fam in q.scenario.FAMILIES}
    report['rename_propagation'] = dict(held_out=1, scored=1, capped_away=0)
    report['na_rm_propagation'] = dict(held_out=2, scored=2, capped_away=0)
    monkeypatch.setattr(q.scenario, 'load_heldout', lambda cap: (
        {fam: [r for r in heldout if r['scenario']['family'] == fam] for fam in q.scenario.FAMILIES}, report))
    monkeypatch.setattr(q.noop, 'corpus_functions',
                        lambda want, seed=None: mock_corpus_universe(['pilot', 'one', 'two'])[:want])
    shape = dict(scenario_total=3, rename_total=1, fresh_scenario_ids=2, fresh_corpus_n=2, min_noop_cases=2,
                 pilot_scenario=1, pilot_noop=1, pilot_corpus=1)
    q.prepare(run, assets, shape=shape)
    rows = q.read_rows(run / 'cases.jsonl')
    scenario_rows = [r for r in rows if r['kind'] == 'scenario']
    noop_rows = [r for r in rows if r['kind'] == 'noop']
    assert len(scenario_rows) == 3
    assert {r['id'] for r in scenario_rows} >= {q.sha1_id(b'P:rename')}
    assert all(not r['rel_path'].startswith('authored/') for r in noop_rows)
    assert ('after_close_brace', 'pilotnoop') not in {(r['case_kind'], r['id']) for r in noop_rows}
    assert len(noop_rows) >= 2
    selection = json.loads((run / 'selection.json').read_text())
    assert selection['corpus_seed'] == q.CORPUS_SEED and selection['noop_seed'] == q.BUILD_SEED
    assert selection['disjointness']['case_overlap'] == []
    assert [f['fn'] for f in selection['corpus_functions']] == ['one', 'two']
    prepared = json.loads((run / 'prepared.json').read_text())
    assert prepared['expected_rows'] == 2 * len(rows)
    assert prepared['replication_ids'] == [q.sha1_id(b'P:rename')] and len(prepared['fresh_ids']) == 2


def test_prepare_rejects_pilot_seed_reuse(tmp_path):
    run, assets = tmp_path / 'run', frozen_assets(tmp_path)
    run.mkdir()
    (assets / 'selection.json').write_text(json.dumps(dict(corpus_seed=q.CORPUS_SEED, noop_seed=7,
        corpus_functions=[])))
    shape = dict(pilot_scenario=1, pilot_noop=1, pilot_corpus=0)
    with pytest.raises(ValueError, match='seeds'):
        q.load_pilot(assets, shape)


def test_pilot_requests_must_cover_the_replication_subset(tmp_path):
    assets = frozen_assets(tmp_path)
    (assets / 'pilot_requests.jsonl').write_text(json.dumps(dict(arm='Q8_0', kind='scenario',
        id='other', exact=True, valid_pass=True)) + '\n')
    with pytest.raises(ValueError, match='incomplete'):
        q.pilot_verdicts(assets, q.ARMS, {'rep-1'})


# --------------------------------------------------------------------------
# quality evaluate: registered gates, subsets, replication
# --------------------------------------------------------------------------

def quality_run(tmp_path, requests=None, pilot_requests=None, prereg=None):
    run = tmp_path / 'run'
    run.mkdir(exist_ok=True)
    scenario_ids = ['s1', 's2', 's3']
    q.write(run / 'prepared.json', dict(arms=q.ARMS,
        case_ids=[['scenario', sid] for sid in scenario_ids] + [['noop', 'n1'], ['noop', 'n2']],
        expected_rows=10, replication_ids=['s1', 's2'], fresh_ids=['s3'],
        preregistration=dict(source=str(prereg or q.PREREG),
                             sha256=q.registered_margins(prereg or q.PREREG)['sha256'])))
    rows = requests if requests is not None else []
    if not rows:
        for arm in q.ARMS:
            for sid in scenario_ids:
                rows.append(dict(arm=arm, kind='scenario', id=sid, family='rename_propagation',
                    exact=True, valid_pass=True, tokenized_prompt_tokens=10, served_prompt_tokens=10,
                    response=dict(choices=[dict(finish_reason='stop')])))
            for nid in ('n1', 'n2'):
                rows.append(dict(arm=arm, kind='noop', id=nid, cls='a_after_close_brace',
                    expectation='no_proposal', proposal=False, tokenized_prompt_tokens=10,
                    served_prompt_tokens=10, response=dict(choices=[dict(finish_reason='stop')])))
    (run / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assets = frozen_assets(tmp_path)
    if pilot_requests is None:
        pilot_requests = [dict(arm=arm, kind='scenario', id=sid, exact=True, valid_pass=True)
                          for arm in q.ARMS for sid in ('s1', 's2')]
    (assets / 'pilot_requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pilot_requests))
    return run, assets


def test_quality_gates_pass_on_a_clean_paired_run(tmp_path):
    run, assets = quality_run(tmp_path)
    q.evaluate(run, assets)
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['registered_gates']['all_quality_gates_pass']
    assert result['registered_band']['exact_lb_margin_pp'] == -2.0
    assert json.loads((run / 'verdict.json').read_text())['adoption'] == 'NOT-ASSESSED'
    assert result['subsets']['fresh52']['n'] == 1


def test_quality_point_below_the_registered_band_fails(tmp_path):
    run, assets = quality_run(tmp_path)
    rows = q.read_rows(run / 'requests.jsonl')
    for r in rows:
        if r['arm'] == 'Q4_K_M_imatrix' and r['kind'] == 'scenario':
            r['exact'] = False
            r['valid_pass'] = False
    (run / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    q.evaluate(run, assets)
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert not gates['exact_point_ge_registered']      # -33.3pp < -0.5pp
    assert not gates['valid_lb_ge_registered']
    assert not gates['all_quality_gates_pass']


def test_quality_noop_restraint_gate_and_rule_of_three(tmp_path):
    run, assets = quality_run(tmp_path)
    rows = q.read_rows(run / 'requests.jsonl')
    for r in rows:
        if r['arm'] == 'Q4_K_M_imatrix' and r['kind'] == 'noop':
            r['proposal'] = True
    (run / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    q.evaluate(run, assets)
    result = json.loads((run / 'evaluation.json').read_text())
    assert not result['registered_gates']['noop_fresh_losses_le_gains']
    assert result['paired']['Q4_K_M_imatrix']['noop_rule_of_three_pp'] == pytest.approx(150.0)


def test_quality_replication_flip_is_reported_not_hidden(tmp_path):
    flipped = []
    for arm in q.ARMS:
        flipped.append(dict(arm=arm, kind='scenario', id='s1', exact=arm == 'Q8_0', valid_pass=True))
        flipped.append(dict(arm=arm, kind='scenario', id='s2', exact=True, valid_pass=True))
    run, assets = quality_run(tmp_path, pilot_requests=flipped)
    q.evaluate(run, assets)
    result = json.loads((run / 'evaluation.json').read_text())
    replication = result['subsets']['replication']['Q4_K_M_imatrix']
    assert replication['exact_flips'] == ['s1'] and not replication['verdict_match']
    assert not result['registered_gates']['replication_1to1']


def test_quality_preregistration_cannot_change_after_freeze(tmp_path):
    other = tmp_path / 'other-prereg.json'
    other.write_text(json.dumps(dict(registered='LB >= -2pp AND point estimate >= -0.5pp',
        rationale='valid >=-2pp noop <= Q8 intent >= 0 cycle >= 1.15x drift <=3%')))
    run, assets = quality_run(tmp_path, prereg=q.PREREG)
    prepared = json.loads((run / 'prepared.json').read_text())
    prepared['preregistration']['source'] = str(other)
    (run / 'prepared.json').unlink()
    (run / 'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='[Pp]reregistration'):
        q.evaluate(run, assets)


def test_quality_missing_candidate_row_fails(tmp_path):
    run, assets = quality_run(tmp_path)
    rows = q.read_rows(run / 'requests.jsonl')
    (run / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1]))
    with pytest.raises(ValueError, match='coverage'):
        q.evaluate(run, assets)


# --------------------------------------------------------------------------
# intent evaluate
# --------------------------------------------------------------------------

def intent_case(cid, family, assertion, **kw):
    row = dict(id=cid, family=family, assertion=assertion,
               input=dict(filename='a.R', prefix_lines=['x <- 1'], suffix_lines=[], cursor_partial=''))
    row.update(kw)
    return row


def synthetic_suite():
    return [intent_case('live-sd-line', 'live', 'Add an sd line'),
            intent_case('live-post-brace', 'live', 'Close the brace'),
            intent_case('ren-1', 'rename_propagation', 'Rename the old name (score) to result',
                        gt_completion=['result <- 1']),
            intent_case('fb-1', 'finish_block', 'Finish the block')]


def intent_run(tmp_path, monkeypatch, scores=None):
    run = tmp_path / 'run'
    run.mkdir(exist_ok=True)
    assets = tmp_path / 'assets'
    assets.mkdir(exist_ok=True)
    suite = synthetic_suite()
    monkeypatch.setattr(i, 'suite', lambda a: suite)
    (assets / 'intent_suite_v1.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in suite))
    anchors = i.intent.anchors({c['id']: c for c in suite})
    if scores is None:
        scores = dict(Q8_0=[2, 0, 2, 1], Q4_K_M_imatrix=[2, 0, 2, 1])
    lines = [json.dumps(dict(anchor=name, request_sha256='h' + name, expected=expected,
                             score=expected, reason='anchor')) for name, _, _, expected in anchors]
    for arm in i.ARMS:
        for case, score in zip(suite, scores[arm]):
            lines.append(json.dumps(dict(arm=arm, id=case['id'], request_sha256='x', cached=False,
                                         score=score, reason='case')))
    (run / 'scores.jsonl').write_text('\n'.join(lines) + '\n')
    i.write(run / 'prepared.json', dict(arms=i.ARMS, case_ids=[c['id'] for c in suite], expected_rows=8,
        preregistration=dict(source=str(i.PREREG), sha256=i.registered_margins(i.PREREG)['sha256'])))
    return run, assets


def test_intent_gates_pass_on_a_non_negative_delta(tmp_path, monkeypatch):
    run, assets = intent_run(tmp_path, monkeypatch, scores=dict(Q8_0=[2, 0, 2, 1], Q4_K_M_imatrix=[2, 1, 2, 2]))
    i.evaluate_judge(run, assets)
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['registered_gates']['all_intent_gates_pass']
    assert result['arms']['Q4_K_M_imatrix']['satisfied_delta_pp'] == pytest.approx(25.0)
    assert json.loads((run / 'verdict.json').read_text())['adoption'] == 'NOT-ASSESSED'


def test_intent_point_gate_fails_on_a_net_loss(tmp_path, monkeypatch):
    run, assets = intent_run(tmp_path, monkeypatch, scores=dict(Q8_0=[2, 2, 2, 1], Q4_K_M_imatrix=[2, 0, 2, 1]))
    i.evaluate_judge(run, assets)
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert not gates['satisfied_point_delta_ge_registered']
    assert not gates['all_intent_gates_pass']


def test_intent_family_collapse_is_detected(tmp_path, monkeypatch):
    run, assets = intent_run(tmp_path, monkeypatch, scores=dict(Q8_0=[2, 2, 2, 1], Q4_K_M_imatrix=[0, 0, 2, 1]))
    i.evaluate_judge(run, assets)
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['by_family']['live']['collapsed']
    assert not result['registered_gates']['no_family_collapse']


def test_intent_anchor_mismatch_fails_loudly(tmp_path, monkeypatch):
    run, assets = intent_run(tmp_path, monkeypatch)
    rows = q.read_rows(run / 'scores.jsonl')
    rows[0]['score'] = 0
    (run / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='calibration'):
        i.evaluate_judge(run, assets)


def test_judge_payload_stays_arm_blind():
    case = synthetic_suite()[0]
    body = i.judge_body(case, ['x <- 2'])
    other = dict(case)
    other['assertion'] = 'Different'
    assert i.body_bytes(body) != i.body_bytes(i.judge_body(other, ['x <- 2']))
    same = dict(case, input=dict(case['input'], cursor_partial='x'))
    assert i.body_bytes(i.judge_body(same, ['x <- 2'])) != i.body_bytes(body)


# --------------------------------------------------------------------------
# timing evaluate
# --------------------------------------------------------------------------

TIMING_SHAPE = dict(formats=t.FORMATS, reps=1, trace_ids={'2k': ['a'], '8k': ['b'], 'short_edit': ['c']},
                    preregistration=dict(source=str(t.PREREG), sha256=t.registered_margins(t.PREREG)['sha256']),
                    tier='cpu')


def timing_row(fmt, cc, tid, prompt_ms, predicted_ms, predicted_n=35, stop_type='word'):
    cycle = prompt_ms + predicted_ms
    return dict(format=fmt, rep=0, trace_id=tid, ctx_class=cc, prompt_n=99,
        tokenized_prompt_tokens=99, prompt_ms=prompt_ms, predicted_ms=predicted_ms,
        wall_ms=cycle + 5, host_load=[1.0, 1.0, 1.0],
        raw_response=dict(stop=True, stop_type=stop_type, timings=dict(prompt_n=99,
            prompt_ms=prompt_ms, predicted_ms=predicted_ms, predicted_n=predicted_n)))


def timing_run(tmp_path, rows, name='run'):
    run = tmp_path / name
    run.mkdir(exist_ok=True)
    t.write(run / 'prepared.json', TIMING_SHAPE)
    (run / 'per_request.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return run


def base_timing_rows(q8=(1000, 500), q4=(500, 250), bookend=(1010, 505)):
    rows = []
    for fmt, (pm, dm) in (('Q8_0', q8), ('Q4_K_M_imatrix', q4), ('Q8_0-bookend', bookend)):
        for cc, tid in (('2k', 'a'), ('8k', 'b'), ('short_edit', 'c')):
            rows.append(timing_row(fmt, cc, tid, pm, dm))
    return rows


def test_timing_speedup_and_drift_gates_pass(tmp_path):
    run = timing_run(tmp_path, base_timing_rows())
    t.evaluate(run)
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['registered_gates']['all_timing_gates_pass']
    assert result['speedups']['Q4_K_M_imatrix']['by_ctx_class']['2k'] == pytest.approx(2.0)
    assert result['speedups']['Q4_K_M_imatrix']['by_ctx_class']['8k'] == pytest.approx(2.0)
    assert result['bookend_drift_pct']['2k'] == pytest.approx(1.0)
    assert json.loads((run / 'verdict.json').read_text())['adoption'] == 'NOT-ASSESSED'


def test_timing_speedup_gate_fails_below_the_registered_floor(tmp_path):
    run = timing_run(tmp_path, base_timing_rows(q4=(950, 475)))
    t.evaluate(run)
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert not gates['speedup_2k_ge_registered']
    assert not gates['all_timing_gates_pass']


def test_timing_bookend_drift_gate_fails(tmp_path):
    run = timing_run(tmp_path, base_timing_rows(bookend=(1100, 550)))
    t.evaluate(run)
    result = json.loads((run / 'evaluation.json').read_text())
    assert not result['registered_gates']['bookend_drift_le_registered']  # 10% drift


def test_timing_token_median_parity_gate(tmp_path):
    rows = base_timing_rows()
    for r in rows:
        if r['format'] == 'Q4_K_M_imatrix':
            r['raw_response']['timings']['predicted_n'] = 38
    run = timing_run(tmp_path, rows)
    t.evaluate(run)
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert not gates['token_median_parity']  # +3 tokens > +/-2 band


def test_timing_limit_rows_parity_gate(tmp_path):
    rows = [timing_row(fmt, cc, tid, pm, dm, stop_type=('limit' if fmt == 'Q4_K_M_imatrix' else 'word'))
            for fmt, (pm, dm) in (('Q8_0', (1000, 500)), ('Q4_K_M_imatrix', (500, 250)), ('Q8_0-bookend', (1010, 505)))
            for cc, tid in (('2k', 'a'), ('8k', 'b'), ('short_edit', 'c'))]
    run = timing_run(tmp_path, rows)
    t.evaluate(run)
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['formats']['Q4_K_M_imatrix']['by_ctx_class']['2k']['generation_limit_rows'] == 1
    assert result['registered_gates']['token_limit_rows_parity']  # delta 1 <= 3
    rows[3]['raw_response']['stop_type'] = 'limit'
    rows[4]['raw_response']['stop_type'] = 'limit'
    rows[5]['raw_response']['stop_type'] = 'limit'
    rows.append(timing_row('Q4_K_M_imatrix', '2k', 'a', 500, 250, stop_type='limit'))
    run2 = timing_run(tmp_path, rows, name='run2')
    with pytest.raises(ValueError, match='coverage'):
        t.evaluate(run2)


def test_timing_short_edit_class_is_mandatory(tmp_path):
    rows = [r for r in base_timing_rows() if r['ctx_class'] != 'short_edit']
    run = timing_run(tmp_path, rows)
    with pytest.raises(ValueError, match='coverage'):
        t.evaluate(run)


def test_timing_load1_floor_is_reported(tmp_path):
    rows = base_timing_rows()
    for r in rows:
        r['host_load'] = [9.0, 9.0, 9.0]
    run = timing_run(tmp_path, rows)
    t.evaluate(run)
    window = json.loads((run / 'evaluation.json').read_text())['window']
    assert window['load1_floor_exceeded'] and window['max_load1'] == pytest.approx(9.0)


# --------------------------------------------------------------------------
# end-to-end CPU smokes (subprocess, synthetic tiny data)
# --------------------------------------------------------------------------

def run_smoke(script, tmp_path):
    run, assets = tmp_path / 'run', tmp_path / 'assets'
    run.mkdir(), assets.mkdir()
    proc = subprocess.run([sys.executable, str(HERE.parent / 'scripts' / 'migration' / script),
                           'smoke', '--run', str(run), '--assets', str(assets)],
                          capture_output=True, text=True, timeout=300, cwd=HERE.parent)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, run


def test_smoke_quality(tmp_path):
    out, run = run_smoke('s2_quality.py', tmp_path)
    assert 'SMOKE-OK' in out
    assert (run / 'cases.jsonl').is_file() and (run / 'selection.json').is_file()
    assert (run / 'evaluation.json').is_file() and (run / 'verdict.json').is_file()
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert gates['all_quality_gates_pass']


def test_smoke_intent(tmp_path):
    out, run = run_smoke('s2_intent.py', tmp_path)
    assert 'SMOKE-OK' in out
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert gates['all_intent_gates_pass'] and gates['anchors_pass']


def test_smoke_timing(tmp_path):
    out, run = run_smoke('s2_timing.py', tmp_path)
    assert 'SMOKE-OK' in out
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['registered_gates']['all_timing_gates_pass']
    assert result['rows'] == 324
