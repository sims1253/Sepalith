#!/usr/bin/env python3
"""Separate frozen local intent generation from calibrated remote judging."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import run_intent_suite as intent
from s1_gpu import GpuServer, DeadlineExceeded, check_offload, tokenize, write

ARMS = ['Q8_0', 'Q6_K_imatrix', 'IQ4_XS_imatrix']


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def suite(assets):
    rows = read_rows(assets / 'intent_suite_v1.jsonl')
    if len(rows) != 44 or len({r['id'] for r in rows}) != 44:
        raise ValueError('Require all 44 unique frozen intent cases')
    return rows


def prepare(run, assets):
    cases = suite(assets)
    write(run / 'prepared.json', dict(arms=ARMS, case_ids=[c['id'] for c in cases],
          expected_rows=132, ctx=8192, max_tokens=320, max_seconds=870))


def measure(run, assets):
    p = json.loads((run / 'prepared.json').read_text())
    cases = suite(assets)
    counts = {}
    server = None
    def expire(*_):
        raise DeadlineExceeded('Intent generation exceeded 15-minute bound')
    old = signal.signal(signal.SIGALRM, expire)
    deadline = time.monotonic() + p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run / 'requests.jsonl').open('x') as out:
            for arm in p['arms']:
                server = GpuServer(assets / (arm + '.gguf'), 18476,
                    ['-lv', '4', '--seed', '20260905'], ctx=p['ctx'],
                    server=Path(os.environ['S1_RUNTIME']) / 'llama-server',
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
                        body = json.dumps(dict(prompt=prompt, max_tokens=p['max_tokens'],
                            temperature=0, seed=20260905, stop=intent.STOPS, stream=False)).encode()
                        req = urllib.request.Request(f'http://127.0.0.1:{server.port}/v1/completions',
                            data=body, headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req, timeout=120) as response:
                            data = json.load(response)
                        row = dict(arm=arm, id=case['id'], family=case['family'],
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            tokenized_prompt_tokens=count, response=data, at=time.time())
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
    write(run / 'verdict.json', dict(verdict='INTENT-GENERATION-COMPLETE',
        adoption='NOT-ASSESSED', boundary='Judge calibration and scores remain pending'))


def judge_body(case, prediction):
    """Preserve the established judge payload; arm identity never enters it."""
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
        raise DeadlineExceeded('Intent judge exceeded one-hour bound')
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


def evaluate_judge(run, assets):
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
    summary = {}
    for arm in ARMS:
        scores = [r['score'] for r in rows if r['arm'] == arm]
        summary[arm] = dict(n=len(scores), mean=statistics.mean(scores),
            normalized_mean=statistics.mean(scores)/2, satisfied=sum(s == 2 for s in scores))
    base = summary[ARMS[0]]
    base_scores = [r['score'] for r in rows if r['arm'] == ARMS[0]]
    for arm in ARMS[1:]:
        scores = [r['score'] for r in rows if r['arm'] == arm]
        summary[arm]['normalized_mean_delta_pp'] = 100 * (summary[arm]['normalized_mean'] - base['normalized_mean'])
        summary[arm]['satisfied_delta_pp'] = 100 * (summary[arm]['satisfied'] - base['satisfied']) / len(cases)
        summary[arm]['score_gains'] = sum(b > a for a, b in zip(base_scores, scores))
        summary[arm]['score_losses'] = sum(b < a for a, b in zip(base_scores, scores))
        summary[arm]['score_ties'] = sum(b == a for a, b in zip(base_scores, scores))
    write(run / 'evaluation.json', dict(rows=len(rows), arms=summary,
        unique_judged_inputs=len({r['request_sha256'] for r in rows}), cache_hits=sum(r['cached'] for r in rows)))
    write(run / 'verdict.json', dict(verdict='PAIRED-INTENT-QUALITY-MEASURED', adoption='NOT-ASSESSED',
        boundary='44 cases, shared scores for byte-identical judge inputs; judge stochastic uncertainty unmeasured. No formal noninferiority or quant promotion.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'measure', 'evaluate_generation', 'judge', 'evaluate_judge'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run, args.assets)
