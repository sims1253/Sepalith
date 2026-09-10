#!/usr/bin/env python3
"""S2 confirmation intent leg: frozen intent_suite_v1 x 2 arms, glm-5.3 judge.

Guardrail endpoint per the S2 design: satisfied point delta >= 0 with anchors
passing; CI is reported but uninformative at n = 44. Successor to b4_intent.py.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import run_intent_suite as intent
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write

ARMS = ['Q8_0', 'Q4_K_M_imatrix']
PORT = 18479
SEED = 20260905
Z_ONE_SIDED_95 = 1.6448536269514722
PREREG = Path(__file__).resolve().parents[2] / 'docs/validation/2026-09-10-s2-preregistration.json'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def suite(assets):
    rows = read_rows(assets / 'intent_suite_v1.jsonl')
    if len(rows) != 44 or len({r['id'] for r in rows}) != 44:
        raise ValueError('Require all 44 unique frozen intent cases')
    return rows


def registered_margins(path):
    """Read the registered intent guardrail from the preregistration json."""
    path = Path(path)
    doc = json.loads(path.read_text())
    text = doc['registered'] + ' ' + doc.get('rationale', '')
    m = re.search(r'intent\s*>=\s*(\d+(?:\.\d+)?)', text)
    if not m:
        raise ValueError('Registered intent gate not found in ' + str(path))
    return dict(intent_min_delta_pp=float(m.group(1)), registered=doc['registered'],
                source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(run, assets, prereg=PREREG):
    cases = suite(assets)
    for arm in ARMS:
        if not (assets / (arm + '.gguf')).is_file():
            raise ValueError('Missing frozen model: ' + arm)
    margins = registered_margins(prereg)
    write(run / 'prepared.json', dict(arms=ARMS, case_ids=[c['id'] for c in cases],
          expected_rows=88, ctx=8192, max_tokens=320, max_seconds=870,
          preregistration=dict(source=margins['source'], sha256=margins['sha256']),
          scope='S2 confirmation intent guardrail on the frozen 44-case suite v1; no new intent cases',
          gates=f'Registered: satisfied point delta >= {margins["intent_min_delta_pp"]}pp, anchors pass, no family collapse; CI uninformative at n=44'))


def complete(port, prompt, max_tokens):
    body = json.dumps(dict(prompt=prompt, max_tokens=max_tokens, temperature=0, seed=SEED,
                           stop=intent.STOPS, stream=False)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    return data, time.monotonic() - started


def measure(run, assets):
    p = json.loads((run / 'prepared.json').read_text())
    cases = suite(assets)
    counts = {}
    server = None

    def expire(*_):
        raise DeadlineExceeded('S2 intent generation exceeded its execution bound')

    old = signal.signal(signal.SIGALRM, expire)
    deadline = time.monotonic() + p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run / 'requests.jsonl').open('x') as out:
            for arm in p['arms']:
                server = GpuServer(assets / (arm + '.gguf'), PORT, ['-lv', '4', '--seed', str(SEED)],
                    ctx=p['ctx'], server=Path(os.environ['S1_RUNTIME']) / 'llama-server',
                    foreground=True, log_path=run / ('server-' + arm + '.log'))
                try:
                    server.start(ready_timeout=180)
                    check_offload(server.log_path.read_text())
                    for case in cases:
                        prompt = intent.render_prompt(case)
                        count = tokenize(server.port, prompt)
                        if count + p['max_tokens'] + 16 > p['ctx']:
                            raise ValueError('Intent prompt exceeds context')
                        if case['id'] in counts and counts[case['id']] != count:
                            raise ValueError('Intent cross-arm tokenizer mismatch')
                        counts[case['id']] = count
                        data, wall = complete(server.port, prompt, p['max_tokens'])
                        row = dict(arm=arm, id=case['id'], family=case['family'],
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            tokenized_prompt_tokens=count, response=data, wall_s=wall, at=time.time())
                        validate_generation(row)
                        row['prediction'] = intent.parse_prediction(data['choices'][0]['text'])
                        row['raw_chars'] = len(data['choices'][0]['text'])
                        row['judge_completion_truncated'] = len('\n'.join(row['prediction'])) > 1200
                        out.write(json.dumps(row, allow_nan=False) + '\n')
                        out.flush()
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


def validate_generation(row):
    data = row['response']
    if (row.get('error') or data.get('usage', {}).get('prompt_tokens') != row['tokenized_prompt_tokens']
            or data['choices'][0].get('finish_reason') not in ('stop', 'length')):
        raise ValueError('Invalid intent generation response')


def generation_coverage(rows, cases):
    if [(r['arm'], r['id']) for r in rows] != [(a, c['id']) for a in ARMS for c in cases]:
        raise ValueError('Incomplete or reordered intent generation coverage')
    by_id = {case['id']: case for case in cases}
    for row in rows:
        validate_generation(row)
        expected_prompt = hashlib.sha256(intent.render_prompt(by_id[row['id']]).encode()).hexdigest()
        if row['prompt_sha256'] != expected_prompt:
            raise ValueError('Intent prompt identity changed')
        if row['prediction'] != intent.parse_prediction(row['response']['choices'][0]['text']):
            raise ValueError('Intent parsed prediction changed')


def evaluate_generation(run, assets):
    rows = read_rows(run / 'requests.jsonl')
    generation_coverage(rows, suite(assets))
    write(run / 'evaluation.json', dict(rows=len(rows), length_rows=sum(
        r['response']['choices'][0]['finish_reason'] == 'length' for r in rows),
        judge_completion_truncated_rows=sum(r['judge_completion_truncated'] for r in rows)))
    write(run / 'verdict.json', dict(verdict='S2-CONFIRMATION-INTENT-GENERATION-COMPLETE',
        adoption='NOT-ASSESSED', boundary='Judge calibration and scores remain pending'))


def judge_body(case, prediction):
    """Established judge payload; arm identity never enters it."""
    inp = case['input']
    prompt = intent.JUDGE_PROMPT.format(path=inp['filename'],
        history='\n'.join(inp.get('edit_history_lines') or []) or '(none)',
        prefix='\n'.join(inp['prefix_lines'][-25:])[-1500:], partial=inp['cursor_partial'],
        suffix='\n'.join(inp['suffix_lines'][:10])[:800],
        completion='\n'.join(prediction)[:1200] if prediction else '(empty)',
        assertion=case['assertion'])
    return dict(model='glm-5.3', thinking={'type': 'enabled'}, reasoning_effort='low',
        messages=[dict(role='user', content=prompt)], response_format={'type': 'json_object'},
        max_tokens=800, temperature=0)


def body_bytes(body):
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def judge_score(response):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Judge response did not finish normally')
    result = json.loads(choice['message']['content'])
    if type(result.get('score')) is not int or result['score'] not in (0, 1, 2):
        raise ValueError('Judge score must be integer 0, 1 or 2')
    if not isinstance(result.get('reason'), str):
        raise ValueError('Judge reason missing')
    return result


def call_judge(body, audit):
    key = os.environ.get('ZAI_API_KEY')
    if not key:
        raise ValueError('Local ZAI_API_KEY is unavailable')
    payload = body_bytes(body)
    identity = hashlib.sha256(payload).hexdigest()
    for attempt in range(2):
        req = urllib.request.Request(intent.ENDPOINT, data=payload,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.load(response)
            audit.write(json.dumps(dict(request_sha256=identity, request=body,
                attempt=attempt, response=data, at=time.time())) + '\n')
            audit.flush()
            return identity, judge_score(data)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            # Never log request headers, credentials or arbitrary exception text.
            audit.write(json.dumps(dict(request_sha256=identity, attempt=attempt,
                error_type=type(exc).__name__, at=time.time())) + '\n')
            audit.flush()
            if attempt == 1:
                raise ValueError('Judge failed after two bounded attempts') from None
            time.sleep(2)


def judge(run, assets):
    cases = suite(assets)
    by_id = {c['id']: c for c in cases}
    rows = read_rows(assets / 'generation.jsonl')
    generation_coverage(rows, cases)

    def expire(*_):
        raise DeadlineExceeded('S2 intent judge exceeded one-hour bound')

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(3570)
    cache = {}
    try:
        with (run / 'judge-calls.jsonl').open('x') as audit, (run / 'scores.jsonl').open('x') as out:
            for name, case, prediction, expected in intent.anchors(by_id):
                identity, result = call_judge(judge_body(case, prediction), audit)
                out.write(json.dumps(dict(anchor=name, request_sha256=identity,
                    expected=expected, **result)) + '\n')
                out.flush()
                if result['score'] != expected:
                    raise ValueError('Intent judge calibration failed: ' + name)
            for row in rows:
                body = judge_body(by_id[row['id']], row['prediction'])
                identity = hashlib.sha256(body_bytes(body)).hexdigest()
                cached = identity in cache
                if not cached:
                    _, result = call_judge(body, audit)
                    cache[identity] = result
                out.write(json.dumps(dict(arm=row['arm'], id=row['id'], request_sha256=identity,
                    cached=cached, **cache[identity])) + '\n')
                out.flush()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def paired(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError('Invalid paired cohort')
    loss = sum(a and not b for a, b in zip(base, candidate))
    gain = sum(b and not a for a, b in zip(base, candidate))
    return dict(n=len(base), losses=loss, gains=gain, delta_pp=100 * (gain - loss) / len(base))


def satisfied_lower_bound(result):
    n = result['n']
    p10, p01 = result['losses'] / n, result['gains'] / n
    d = p01 - p10
    se = math.sqrt(max(0.0, p10 + p01 - d * d) / n)
    return dict(delta_pp=100 * d, se_pp=100 * se, lb_pp=100 * (d - Z_ONE_SIDED_95 * se),
                interpretation='uninformative at n=44; guardrail uses the point estimate')


def evaluate_judge(run, assets, prereg=PREREG):
    cases = suite(assets)
    rows = read_rows(run / 'scores.jsonl')
    anchors = rows[:3]
    expected = intent.anchors({c['id']: c for c in cases})
    if [(r.get('anchor'), r.get('score')) for r in anchors] != [(n, e) for n, _, _, e in expected]:
        raise ValueError('Missing or failed judge calibration')
    rows = rows[3:]
    if [(r.get('arm'), r.get('id')) for r in rows] != [(a, c['id']) for a in ARMS for c in cases]:
        raise ValueError('Incomplete judge coverage')
    if any(type(r.get('score')) is not int or r['score'] not in (0, 1, 2) for r in rows):
        raise ValueError('Invalid scored case')
    prepared = run / 'prepared.json'
    frozen = json.loads(prepared.read_text()).get('preregistration') or {} if prepared.is_file() else {}
    source = Path(frozen.get('source') or prereg)
    margins = registered_margins(source)
    if frozen.get('sha256') and frozen['sha256'] != margins['sha256']:
        raise ValueError('Preregistration changed after cohort freeze')
    summary = {}
    for arm in ARMS:
        scores = [r['score'] for r in rows if r['arm'] == arm]
        summary[arm] = dict(n=len(scores), mean=statistics.mean(scores),
            normalized_mean=statistics.mean(scores) / 2, satisfied=sum(s == 2 for s in scores))
    base = summary[ARMS[0]]
    base_scores = [r['score'] for r in rows if r['arm'] == ARMS[0]]
    for arm in ARMS[1:]:
        scores = [r['score'] for r in rows if r['arm'] == arm]
        summary[arm]['normalized_mean_delta_pp'] = 100 * (summary[arm]['normalized_mean'] - base['normalized_mean'])
        summary[arm]['satisfied_delta_pp'] = 100 * (summary[arm]['satisfied'] - base['satisfied']) / len(cases)
        summary[arm]['score_gains'] = sum(b > a for a, b in zip(base_scores, scores))
        summary[arm]['score_losses'] = sum(b < a for a, b in zip(base_scores, scores))
        summary[arm]['score_ties'] = sum(b == a for a, b in zip(base_scores, scores))
    by_id_rows = {(r['arm'], r['id']): r['score'] for r in rows}
    families = sorted({c['family'] for c in cases})
    by_family = {}
    collapses = []
    for family in families:
        ids = [c['id'] for c in cases if c['family'] == family]
        entry = dict(n=len(ids))
        for arm in ARMS:
            entry[arm + '_satisfied'] = sum(by_id_rows[(arm, i)] == 2 for i in ids)
        entry['satisfied_delta'] = entry[ARMS[1] + '_satisfied'] - entry[ARMS[0] + '_satisfied']
        entry['collapsed'] = entry[ARMS[1] + '_satisfied'] == 0 and entry[ARMS[0] + '_satisfied'] > 0
        if entry['collapsed']:
            collapses.append(family)
        by_family[family] = entry
    satisfied = paired([by_id_rows[(ARMS[0], c['id'])] == 2 for c in cases],
                       [by_id_rows[(ARMS[1], c['id'])] == 2 for c in cases])
    candidate = summary[ARMS[1]]
    gates = dict(
        satisfied_point_delta_ge_registered=candidate['satisfied_delta_pp'] >= margins['intent_min_delta_pp'],
        anchors_pass=True,
        no_family_collapse=not collapses)
    gates['all_intent_gates_pass'] = all(gates.values())
    write(run / 'evaluation.json', dict(rows=len(rows), arms=summary, by_family=by_family,
        satisfied_paired=satisfied, satisfied_lower_bound=satisfied_lower_bound(satisfied),
        registered_gates=gates, registered_band=margins,
        unique_judged_inputs=len({r['request_sha256'] for r in rows}), cache_hits=sum(r['cached'] for r in rows)))
    write(run / 'verdict.json', dict(verdict='S2-CONFIRMATION-INTENT-MEASURED', adoption='NOT-ASSESSED',
        boundary='Guardrail endpoint: point estimate, anchor calibration, family table. CI uninformative at n=44; '
                 'judge stochastic uncertainty unmeasured. Shared scores for byte-identical judge inputs. No quant promotion from this leg alone.'))


def smoke(run, assets):
    """CPU-only wiring smoke on a synthetic 4-case suite; no models, GPU or network."""
    def case(cid, family, assertion, **kw):
        row = dict(id=cid, family=family, assertion=assertion,
                   input=dict(filename='a.R', prefix_lines=['x <- 1'], suffix_lines=[], cursor_partial=''))
        row.update(kw)
        return row

    for arm in ARMS:
        (assets / (arm + '.gguf')).write_bytes(b'smoke')
    synthetic = [
        case('live-sd-line', 'live', 'Add an sd line'),
        case('live-post-brace', 'live', 'Close the brace'),
        case('ren-1', 'rename_propagation', 'Rename the old name (score) to result', gt_completion=['result <- 1']),
        case('fb-1', 'finish_block', 'Finish the block'),
    ]
    texts = {'live-sd-line': '  list(mean = m)\n>>>>>>> UPDATED', 'live-post-brace': '}\n>>>>>>> UPDATED',
             'ren-1': 'result <- 1\n>>>>>>> UPDATED', 'fb-1': '  x\n}\n>>>>>>> UPDATED'}
    class FakeServer:
        def __init__(self, model, port, flags, **kw):
            self.port = port
            self.log_path = run / 'server.log'
            self.log_path.write_text('smoke server log\n')
        def start(self, ready_timeout=0):
            return True
        def stop(self):
            return None

    prompts = {c['id']: intent.render_prompt(c) for c in synthetic}
    by_prompt = {prompt: cid for cid, prompt in prompts.items()}

    def fake_complete(port, prompt, max_tokens):
        cid = by_prompt[prompt]
        data = dict(usage=dict(prompt_tokens=30), choices=[dict(finish_reason='stop', text=texts[cid])])
        return data, 0.01

    saved_suite, saved_server, saved_offload, saved_tok, saved_complete = (
        suite, GpuServer, check_offload, tokenize, complete)
    try:
        globals()['suite'] = lambda a: synthetic
        globals()['GpuServer'] = FakeServer
        globals()['check_offload'] = lambda log: None
        globals()['tokenize'] = lambda port, prompt: 30
        globals()['complete'] = fake_complete
        prepare(run, assets)
        saved_runtime = os.environ.get('S1_RUNTIME')
        os.environ['S1_RUNTIME'] = str(assets)
        try:
            measure(run, assets)
        finally:
            if saved_runtime is None:
                os.environ.pop('S1_RUNTIME', None)
            else:
                os.environ['S1_RUNTIME'] = saved_runtime
        evaluate_generation(run, assets)
        generations = read_rows(run / 'requests.jsonl')
        (assets / 'generation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generations))
        # The battery runs each action in its own run dir; the smoke reuses one.
        for stage in ('evaluation.json', 'verdict.json'):
            (run / stage).unlink(missing_ok=True)

        def fake_call_judge(body, audit):
            identity = hashlib.sha256(body_bytes(body)).hexdigest()
            completion = body['messages'][0]['content']
            score = 2 if ('mean = m' in completion or 'result <- 1' in completion) else 0
            return identity, dict(score=score, reason='smoke')
        saved_call = call_judge
        globals()['call_judge'] = fake_call_judge
        try:
            judge(run, assets)
        finally:
            globals()['call_judge'] = saved_call
        evaluate_judge(run, assets)
    finally:
        globals()['suite'] = saved_suite
        globals()['GpuServer'] = saved_server
        globals()['check_offload'] = saved_offload
        globals()['tokenize'] = saved_tok
        globals()['complete'] = saved_complete
    gates = json.loads((run / 'evaluation.json').read_text())['registered_gates']
    assert gates['all_intent_gates_pass'], gates
    print('SMOKE-OK ' + json.dumps(gates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'measure', 'evaluate_generation', 'judge',
                                           'evaluate_judge', 'smoke'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prereg', type=Path, default=PREREG)
    args = parser.parse_args()
    if args.action == 'smoke':
        smoke(args.run, args.assets)
    elif args.action in ('judge', 'evaluate_generation'):
        globals()[args.action](args.run, args.assets)
    else:
        globals()[args.action](args.run, args.assets, args.prereg)
