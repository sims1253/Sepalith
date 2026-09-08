#!/usr/bin/env python3
"""Run the registered S1 depth sweep with frozen inputs and tracked GPU servers."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import spec_bench as bench

ARMS = ['baseline'] + [f'ngram-simple@{n}' for n in (2, 3, 4, 8, 16, 48)] + [
    f'{arm}@{n}' for arm in ('draft-mtp', 'model-draft') for n in range(1, 6)] + ['baseline-bookend']


class GpuServer(bench.SpecServer):
    def cmd(self):
        argv = super().cmd()
        index = argv.index('-ngl')
        argv[index + 1] = '99'
        return argv


class DeadlineExceeded(BaseException):
    """Do not let the legacy stream error handler swallow the run deadline."""


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def selected(assets):
    by_class = bench.load_traces(assets / 'traces.jsonl')
    rows = [r for cc in ('2k', '8k') for r in bench.sample_traces(by_class[cc], 100)]
    if len(rows) != 200 or len({r['trace_id'] for r in rows}) != 200:
        raise ValueError('Require 100 unique traces per context class')
    return rows


def prepare(run, assets):
    rows = selected(assets)
    for name in ('model.gguf', 'mtp.gguf', 'draft.gguf'):
        if not (assets / name).is_file():
            raise ValueError('Missing frozen asset: ' + name)
    if not (Path(os.environ['S1_RUNTIME']) / 'llama-server').is_file():
        raise ValueError('Missing frozen runtime')
    write(run / 'prepared.json', dict(arms=ARMS, trace_ids=[r['trace_id'] for r in rows],
        reps=3, expected_rows=10800, ctx=10240, max_tokens=64, max_seconds=14370,
        scientific_scope='Registered S1 GPU sweep; CPU depth sweep remains separate',
        draft_boundary='b2 same-family stand-in, not a trained Matryoshka draft; MTP head is grafted base head'))


def check_offload(log, draft=False):
    matches = [(int(a), int(b)) for a, b in re.findall(r'offloaded (\d+)/(\d+) layers', log)]
    if len(matches) < (2 if draft else 1) or any(a != b or a <= 0 for a, b in matches):
        raise ValueError('Full target/draft GPU offload not verified')


def tokenize(port, prompt):
    body = json.dumps(dict(content=prompt, add_special=True)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/tokenize', data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return len(json.load(response)['tokens'])


def validate_response(response):
    if response.error or not response.stop_hit or response.ttft_ms is None:
        raise ValueError('Incomplete or failed streamed response: ' + str(response.error))
    if not math.isfinite(response.wall_ms) or not bench.gen_tps(response):
        raise ValueError('Missing or invalid response timings')


def measure(run, assets):
    prepared = json.loads((run / 'prepared.json').read_text())
    traces = selected(assets)
    baseline = {}
    server = None
    def expire(*_):
        raise DeadlineExceeded('S1 exceeded its four-hour execution bound')
    old = signal.signal(signal.SIGALRM, expire)
    deadline = time.monotonic() + prepared['max_seconds']
    signal.alarm(prepared['max_seconds'])
    try:
        with (run / 'per_request.jsonl').open('x') as output:
            for arm in prepared['arms']:
                model, flags, label = bench.arm_flags('baseline' if arm == 'baseline-bookend' else arm, model=assets / 'model.gguf',
                    model_mtp=assets / 'mtp.gguf', model_draft=assets / 'draft.gguf')
                label = arm
                flags += ['-lv', '4', '--seed', '20260905']
                if arm.startswith('model-draft'):
                    flags += ['-ngld', '99']
                server = GpuServer(model, 18471, flags, ctx=prepared['ctx'],
                    server=Path(os.environ['S1_RUNTIME']) / 'llama-server', foreground=True,
                    log_path=run / ('server-' + label + '.log'))
                try:
                    server.start(ready_timeout=300)
                    check_offload(server.log_path.read_text(), arm.startswith('model-draft'))
                    with (run / 'telemetry.jsonl').open('a') as telemetry:
                        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used', '--format=csv,noheader'], text=True, timeout=10)
                        telemetry.write(json.dumps(dict(arm=arm, at=time.time(), gpu=gpu, load=os.getloadavg())) + '\n')
                    warmup = bench.stream_completion(server.port, traces[-1]['prompt'], timeout=120)
                    validate_response(warmup)
                    with (run / 'warmups.jsonl').open('a') as warmup_file:
                        warmup_file.write(json.dumps(dict(arm=arm, text=warmup.text, timings=warmup.timings, wall_ms=warmup.wall_ms)) + '\n')
                    for trace in traces:
                        token_count = tokenize(server.port, trace['prompt'])
                        if token_count != trace['prompt_tokens']:
                            raise ValueError('Frozen trace token count differs from serving tokenizer')
                        if token_count + 64 + 16 > prepared['ctx']:
                            raise ValueError('Prompt would exceed context with output reserve')
                        for rep in range(prepared['reps']):
                            res = bench.stream_completion(server.port, trace['prompt'], timeout=120)
                            warm = bench.stream_completion(server.port, trace['prompt'], cache_prompt=True, timeout=120)
                            tps, rate = bench.acceptance_from_response(res.timings)
                            row = dict(arm=label, trace_id=trace['trace_id'], ctx_class=trace['ctx_class'], rep=rep,
                                prompt_tokens=trace['prompt_tokens'], tokenized_prompt_tokens=token_count,
                                n_prompt_srv=res.timings.get('prompt_n'), prompt_ms=res.timings.get('prompt_ms'),
                                n_predicted=res.n_predicted or res.timings.get('predicted_n'),
                                draft_n=res.timings.get('draft_n'), draft_n_accepted=res.timings.get('draft_n_accepted'),
                                gen_tps=bench.gen_tps(res), ttft_ms=res.ttft_ms, wall_ms=res.wall_ms,
                                stop_hit=res.stop_hit, error=res.error, accept_tps=tps, accept_rate=rate,
                                warm_prompt_ms=warm.timings.get('prompt_ms'), warm_ttft_ms=warm.ttft_ms,
                                cold_timings=res.timings, warm_timings=warm.timings, warm_gen_tps=bench.gen_tps(warm),
                                warm_wall_ms=warm.wall_ms, warm_error=warm.error, warm_stop_hit=warm.stop_hit,
                                gen_text=res.text, warm_gen_text=warm.text, host_load=os.getloadavg(),
                                recorded_at=time.time())
                            key = (trace['trace_id'], rep)
                            if arm == 'baseline':
                                baseline[key] = res.text
                            else:
                                row['matches_baseline'] = res.text == baseline[key]
                                row['warm_matches_baseline'] = warm.text == baseline[key]
                            output.write(json.dumps(row, allow_nan=False) + '\n'); output.flush()
                            validate_response(res); validate_response(warm)
                            if row['n_prompt_srv'] != token_count:
                                raise ValueError('Served cold prompt token count differs from tokenizer')
                        print(json.dumps(dict(arm=arm, trace_id=trace['trace_id'], rows=3)), flush=True)
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


def evaluate(run):
    p = json.loads((run / 'prepared.json').read_text())
    rows = [json.loads(line) for line in (run / 'per_request.jsonl').read_text().splitlines()]
    expected = [(arm, trace, rep) for arm in p['arms'] for trace in p['trace_ids'] for rep in range(p['reps'])]
    if [(r['arm'], r['trace_id'], r['rep']) for r in rows] != expected:
        raise ValueError('Incomplete, duplicate or reordered measurement coverage')
    if any(r['error'] or r['warm_error'] or not r['stop_hit'] or not r['warm_stop_hit'] or
           r['n_prompt_srv'] != r['tokenized_prompt_tokens'] for r in rows):
        raise ValueError('Response or prompt coverage gate failed')
    summary = bench.summarize(rows)
    for key, value in summary.items():
        rs = [r for r in rows if f"{r['arm']}|{r['ctx_class']}" == key]
        value['warm_matches_baseline'] = sum(r.get('warm_matches_baseline', False) for r in rs)
        value['cold_warm_matches'] = sum(r['gen_text'] == r['warm_gen_text'] for r in rs)
    baseline_rows = {(r['trace_id'], r['rep']): r for r in rows if r['arm'] == 'baseline'}
    for key, value in summary.items():
        paired = [r['gen_tps'] / baseline_rows[(r['trace_id'], r['rep'])]['gen_tps']
                  for r in rows if f"{r['arm']}|{r['ctx_class']}" == key]
        value['paired_speedup_median'] = statistics.median(paired)
        value['paired_speedup_quartiles'] = statistics.quantiles(paired, n=4)
    candidates = {k: v for k, v in summary.items() if not k.startswith('baseline')}
    winners = [k for k, v in candidates.items() if v.get('speedup_vs_baseline', 0) >= 1.4 and
               any((r.get('draft_n') or 0) > 0 for r in rows if f"{r['arm']}|{r['ctx_class']}" == k) and
               v.get('matches_baseline') == v['n'] and v['warm_matches_baseline'] == v['n']]
    write(run / 'evaluation.json', dict(rows=len(rows), arms_summary=summary))
    write(run / 'verdict.json', dict(verdict='GPU-SWEEP-MEASURED', candidate_winners=winners,
        drafting_observed={arm: any((r.get('draft_n') or 0) > 0 for r in rows if r['arm'] == arm)
                           for arm in p['arms'] if not arm.startswith('baseline')},
        adoption='REVIEW-REQUIRED' if winners else 'NO-QUALIFIED-GPU-WINNER',
        boundary='Full local GPU sweep only; check environment and paired uncertainty before promotion. CPU and real Matryoshka draft remain separate.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'measure', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    if args.action == 'evaluate':
        evaluate(args.run)
    else:
        globals()[args.action](args.run, args.assets)
