#!/usr/bin/env python3
"""Calibrate frozen W33 inference before filling missing saved-row indices.

Existing results are preserved. A calibration mismatch is an explicit negative
scientific result, not permission to replace historical predictions.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import os
import statistics
import urllib.request


class DeadlineExceeded(RuntimeError):
    """Distinct from OSError, which the server readiness loop retries."""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/eval'))
import eval_ablation
import spec_bench


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(run, assets, fresh=False, gpu=False):
    examples = read(assets / 'examples.jsonl')
    if fresh:
        if len(examples) != 921 or any(not isinstance(r.get('prompt'), str) or not r['prompt']
                                     or not isinstance(r.get('target'), str) for r in examples):
            raise ValueError('Fresh plain-split evaluation requires exactly 921 valid prompt/target rows')
        if not (assets / 'model.gguf').is_file() or not os.access(assets / 'runtime/llama-server', os.X_OK):
            raise ValueError('Frozen model or executable is missing')
        write(run / 'prepared.json', dict(calibration=[], missing=list(range(len(examples))),
              fresh=True, gpu=gpu, input_rows=921, max_seconds=5370, max_tokens=640, context=16384,
              boundary='New current-v7 evaluation on frozen plain split; not a continuation of historical W33.'))
        return
    saved = [r for r in read(assets / 'saved.jsonl') if 'i' in r]
    indices = {r['i'] for r in saved}
    if len(indices) != len(saved) or not indices or max(indices) >= len(examples):
        raise ValueError('Invalid saved row indices')
    for row in saved:
        example = examples[row['i']]
        if row['package'] != (example.get('package') or example.get('package_or_repo')):
            raise ValueError('Saved package does not align with input row')
        for key in ('exact', 'first_line', 'line_f1', 'empty', 'latency_s'):
            if not isinstance(row.get(key), (int, float)):
                raise ValueError('Saved scoring schema is incomplete')
    if any(not isinstance(r.get('prompt'), str) or not isinstance(r.get('target'), str) for r in examples):
        raise ValueError('Invalid prompt/target schema')
    if not (assets / 'model.gguf').is_file() or not os.access(assets / 'runtime/llama-server', os.X_OK):
        raise ValueError('Frozen model or executable is missing')
    candidates = sorted(r['i'] for r in saved if not r.get('error') and not r.get('empty')
                        and isinstance(r.get('pred'), str) and len(r['pred']) < 600)
    if len(candidates) < 16:
        raise ValueError('Need 16 untruncated saved predictions for calibration')
    calibration = [candidates[i * (len(candidates) - 1) // 15] for i in range(16)]
    write(run / 'prepared.json', dict(calibration=calibration,
          missing=sorted(set(range(len(examples))) - indices), saved_rows=len(saved),
          input_rows=len(examples), max_seconds=5370, max_tokens=640, context=16384,
          saved_latency_median=statistics.median(r['latency_s'] for r in saved),
          saved_latency_p95=sorted(r['latency_s'] for r in saved)[int(.95 * (len(saved) - 1))],
          calibration_gate='All 16 normalized predictions and quality scores equal saved rows; zero request errors.',
          boundary='Missing-row completion only. Historical model/runtime identity is not recoverable from saved rows.'))


def measure(run, assets):
    prepared = json.loads((run / 'prepared.json').read_text())
    examples = read(assets / 'examples.jsonl')
    saved = {} if prepared.get('fresh') else {r['i']: r for r in read(assets / 'saved.jsonl') if 'i' in r}
    deadline = time.monotonic() + prepared['max_seconds']

    def expire(*_):
        raise DeadlineExceeded('W33 total process lifetime expired; 30 seconds reserved for cleanup')

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(prepared['max_seconds'])
    server = spec_bench.SpecServer(str(assets / 'model.gguf'), 18461,
                                  ['-ngl', '99', '-lv', '4'] if prepared.get('gpu') else [], threads=8,
                                  ctx=prepared['context'], foreground=True,
                                  server=assets / 'runtime/llama-server',
                                  log_path=run / 'server.log')
    rows = []
    try:
        server.start()
        if prepared.get('gpu'):
            import re
            log = (run / 'server.log').read_text(errors='replace')
            offload = re.findall(r'offloaded (\d+)/(\d+) layers', log)
            if not offload or int(offload[-1][0]) <= 0 or offload[-1][0] != offload[-1][1]:
                raise ValueError('Full GPU offload not verified; refuse silent CPU fallback')
        with (run / 'predictions.jsonl').open('x') as stream:
            for phase, indices in [('calibration', prepared['calibration']), ('repair', prepared['missing'])]:
                if phase == 'repair' and any(not r.get('matches_saved') for r in rows):
                    break
                for i in indices:
                    if time.monotonic() >= deadline:
                        raise DeadlineExceeded('W33 deadline expired')
                    example = examples[i]
                    if prepared.get('fresh'):
                        # Ask the frozen tokenizer before generation. Never let
                        # context shifting silently change an evaluation prompt.
                        request = urllib.request.Request('http://127.0.0.1:18461/tokenize',
                            data=json.dumps({'content': example['prompt'], 'add_special': True}).encode(),
                            headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(request, timeout=30) as response:
                            token_count = len(json.loads(response.read())['tokens'])
                        if token_count + prepared['max_tokens'] + 16 > prepared['context']:
                            raise ValueError(f'Row {i} exceeds frozen context; no truncation permitted')
                    metadata = {}
                    if prepared.get('fresh'):
                        request = urllib.request.Request('http://127.0.0.1:18461/v1/completions',
                            data=json.dumps({'prompt': example['prompt'], 'max_tokens': prepared['max_tokens'],
                                'temperature': 0, 'stop': ['>>>>>>> UPDATED'], 'stream': False}).encode(),
                            headers={'Content-Type': 'application/json'})
                        started = time.monotonic()
                        with urllib.request.urlopen(request, timeout=1200) as response:
                            result = json.loads(response.read())
                        text = result['choices'][0]['text']
                        seconds = time.monotonic() - started
                        usage = result['usage']
                        if type(usage.get('prompt_tokens')) is not int or abs(usage['prompt_tokens'] - token_count) > 16:
                            raise ValueError(f'Row {i}: tokenization/served prompt mismatch')
                        metadata = dict(finish_reason=result['choices'][0].get('finish_reason'),
                                        completion_tokens=usage.get('completion_tokens'),
                                        served_prompt_tokens=usage['prompt_tokens'])
                    else:
                        text, seconds = eval_ablation.complete(18461, example['prompt'], prepared['max_tokens'])
                    pred = eval_ablation.parse_pred(text)
                    scores = eval_ablation.score(pred, eval_ablation.gt_lines(example['target']))
                    row = dict(i=i, phase=phase, pred='\n'.join(pred), raw_text=text, latency_s=seconds,
                               **metadata, **scores)
                    if prepared.get('fresh'):
                        row.update(prompt_tokens=token_count, package=example.get('package'), kind=example.get('kind'))
                    if phase == 'calibration':
                        row['matches_saved'] = (row['pred'] == saved[i]['pred'] and
                                                all(row[k] == saved[i][k] for k in scores))
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ('i', 'phase', 'latency_s')}), flush=True)
    finally:
        signal.alarm(0)
        try:
            server.stop()
        finally:
            signal.signal(signal.SIGALRM, previous)


def evaluate(run):
    prepared = json.loads((run / 'prepared.json').read_text())
    rows = read(run / 'predictions.jsonl')
    calibration = [r for r in rows if r['phase'] == 'calibration']
    repair = [r for r in rows if r['phase'] == 'repair']
    if [r['i'] for r in calibration] != prepared['calibration']:
        raise ValueError('Incomplete calibration')
    passed = all(r['matches_saved'] for r in calibration)
    if [r['i'] for r in repair] != (prepared['missing'] if passed else []):
        raise ValueError('Unexpected or incomplete repair coverage')
    write(run / 'evaluation.json', dict(calibration_matches=sum(r['matches_saved'] for r in calibration),
                                      calibration_n=len(calibration), repair_n=len(repair),
                                      generation_limit_rows=sum(r.get('finish_reason') == 'length' for r in repair),
                                      repair_scores=eval_ablation.agg(repair) if repair else None))
    write(run / 'verdict.json', dict(verdict=('FRESH-PLAIN-SPLIT-MEASURED' if prepared.get('fresh') else
                                            'MISSING-ROWS-MEASURED') if passed else 'CALIBRATION-MISMATCH',
          adoption='NOT-ASSESSED',
          boundary='Retain historical rows separately; never infer original provenance or whole-W33 completion.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'measure', 'evaluate'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.run, a.assets, fresh=a.fresh, gpu=a.gpu)
    elif a.action == 'measure':
        measure(a.run, a.assets)
    else:
        evaluate(a.run)
