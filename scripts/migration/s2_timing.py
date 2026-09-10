#!/usr/bin/env python3
"""S2 confirmation CPU-cycle leg: Q8_0 vs Q4_K_M_imatrix with a Q8 bookend.

Local 5900X/WSL2 quiet window only (P12 idle-window discipline). Registered
gates from docs/validation/2026-09-10-s2-preregistration.json: paired median
speedup >= 1.15x on BOTH ctx classes, bookend drift <= 3%, token-length parity.
Successor to b4_timing.py (tier cpu only).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
import spec_bench as bench
from s1_gpu import DeadlineExceeded, tokenize, write

FORMATS = ['Q8_0', 'Q4_K_M_imatrix', 'Q8_0-bookend']
PORT = 18480
TRACES_PER_CLASS = 12
CTX_CLASSES = ('2k', '8k', 'short_edit')
REPS = 3
N_PREDICT = 48
CTX = 10240
THREADS = 8
SAMPLE_SEED = 20260905
LOAD1_DESCRIPTIVE_FLOOR = 4.0  # pilot ran at 8.3-8.6; the confirmation must not repeat that
PREREG = Path(__file__).resolve().parents[2] / 'docs/validation/2026-09-10-s2-preregistration.json'


def model_path(assets, fmt):
    return assets / (('Q8_0' if fmt == 'Q8_0-bookend' else fmt) + '.gguf')


def registered_margins(path):
    """Read the registered cycle gates from the preregistration json."""
    path = Path(path)
    doc = json.loads(path.read_text())
    text = doc['registered'] + ' ' + doc.get('rationale', '')
    speed = re.search(r'cycle\s*>=\s*(\d+(?:\.\d+)?)x', text)
    drift = re.search(r'drift\s*<=\s*(\d+(?:\.\d+)?)%', text)
    if not (speed and drift):
        raise ValueError('Registered S2 cycle gates not found in ' + str(path))
    return dict(speedup_min=float(speed.group(1)), drift_max_pct=float(drift.group(1)),
                registered=doc['registered'], source=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def sample_rows(rows, n, seed=SAMPLE_SEED):
    """Deterministic sample (same algorithm as bench.sample_traces)."""
    if n >= len(rows):
        return list(rows)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]


def selected(assets):
    by_class = bench.load_traces(assets / 'traces.jsonl')
    traces = {}
    for cc in ('2k', '8k'):
        if len(by_class[cc]) < TRACES_PER_CLASS:
            raise ValueError(f'Expected >= {TRACES_PER_CLASS} frozen {cc} traces')
        traces[cc] = bench.sample_traces(by_class[cc], TRACES_PER_CLASS)
    noop_rows = [json.loads(line) for line in (assets / 'noop_prompts.jsonl').read_text().splitlines()]
    if len(noop_rows) < TRACES_PER_CLASS or len({r['id'] for r in noop_rows}) != len(noop_rows):
        raise ValueError(f'Expected >= {TRACES_PER_CLASS} unique frozen noop prompts')
    traces['short_edit'] = [dict(trace_id=r['id'], prompt=r['prompt'], ctx_class='short_edit')
                            for r in sample_rows(noop_rows, TRACES_PER_CLASS)]
    for cc in CTX_CLASSES:
        if len({t['trace_id'] for t in traces[cc]}) != TRACES_PER_CLASS:
            raise ValueError(f'Expected {TRACES_PER_CLASS} unique {cc} traces')
    return traces


def prepare(run, assets, prereg=PREREG):
    traces = selected(assets)
    for fmt in FORMATS:
        if not model_path(assets, fmt).is_file():
            raise ValueError('Missing quantized model: ' + fmt)
    margins = registered_margins(prereg)
    write(run / 'prepared.json', dict(formats=FORMATS,
        trace_ids={cc: [t['trace_id'] for t in traces[cc]] for cc in CTX_CLASSES},
        reps=REPS, n_predict=N_PREDICT, expected_rows=len(FORMATS) * REPS * 3 * TRACES_PER_CLASS,
        ctx=CTX, threads=THREADS, max_seconds=8100, tier='cpu',
        preregistration=dict(source=margins['source'], sha256=margins['sha256']),
        scope='S2 confirmation CPU cycle timing: 12 traces per ctx class (2k, 8k) plus a 12-case short-edit class '
              'from frozen noop prompts; Q8 bookend brackets the run',
        gates=f'Registered: paired median speedup >= {margins["speedup_min"]}x on BOTH 2k and 8k; bookend drift '
              f'<= {margins["drift_max_pct"]}%; token-length parity (median within +/-2 of Q8, limit rows <= Q8+3)',
        p12='No TRAIN leg active; one contended-free evening; exactly one re-run allowed and only for documented window contamination'))


def request(port, prompt):
    body = json.dumps(dict(prompt=prompt, n_predict=N_PREDICT, temperature=0.0,
                           stream=False, cache_prompt=False, stop=[bench.STOP])).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=body,
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as response:
        value = json.load(response)
    return value, (time.monotonic() - start) * 1000


def measure(run, assets):
    p = json.loads((run / 'prepared.json').read_text())
    traces = selected(assets)
    server = None
    baseline_counts = None

    def expire(*_):
        raise DeadlineExceeded('S2 timing exceeded its execution bound')

    old = signal.signal(signal.SIGALRM, expire)
    deadline = time.monotonic() + p['max_seconds']
    signal.alarm(p['max_seconds'])
    try:
        with (run / 'per_request.jsonl').open('x') as output:
            for fmt in p['formats']:
                server = bench.SpecServer(model_path(assets, fmt), PORT, ['-lv', '4', '--seed', '20260905'],
                    ctx=p['ctx'], threads=p['threads'], server=Path(os.environ['S1_RUNTIME']) / 'llama-server',
                    foreground=True, log_path=run / ('server-' + fmt + '.log'))
                try:
                    server.start(ready_timeout=300)
                    with (run / 'telemetry.jsonl').open('a') as f:
                        f.write(json.dumps(dict(format=fmt, at=time.time(), load=os.getloadavg(), gpu=None)) + '\n')
                    prompts = [(cc, t) for cc in CTX_CLASSES for t in traces[cc]]
                    token_counts = {(cc, t['trace_id']): tokenize(server.port, t['prompt']) for cc, t in prompts}
                    if baseline_counts is None:
                        baseline_counts = token_counts
                    elif token_counts != baseline_counts:
                        raise ValueError('Cross-format tokenizer mismatch')
                    if max(token_counts.values()) + N_PREDICT + 16 > p['ctx']:
                        raise ValueError('Prompt exceeds context reserve')
                    with (run / 'token-audit.jsonl').open('a') as audit:
                        for (cc, tid), n in token_counts.items():
                            audit.write(json.dumps(dict(format=fmt, ctx_class=cc, trace_id=tid, tokens=n)) + '\n')
                    warmup, _ = request(server.port, traces['2k'][-1]['prompt'])
                    with (run / 'warmups.jsonl').open('a') as f:
                        f.write(json.dumps(dict(format=fmt, response=warmup)) + '\n')
                    for rep in range(p['reps']):
                        for cc in CTX_CLASSES:
                            for trace in traces[cc]:
                                data, wall = request(server.port, trace['prompt'])
                                timings = data.get('timings', {})
                                row = dict(format=fmt, rep=rep, trace_id=trace['trace_id'], ctx_class=cc,
                                    prompt_n=timings.get('prompt_n'),
                                    tokenized_prompt_tokens=token_counts[(cc, trace['trace_id'])],
                                    prompt_ms=timings.get('prompt_ms'), predicted_ms=timings.get('predicted_ms'),
                                    wall_ms=wall, raw_response=data, recorded_at=time.time(),
                                    host_load=os.getloadavg())
                                output.write(json.dumps(row, allow_nan=False) + '\n')
                                output.flush()
                                if row['prompt_n'] != row['tokenized_prompt_tokens'] or not data.get('stop'):
                                    raise ValueError('Prompt parity or complete-response gate failed')
                                if any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) or row[k] < 0
                                       for k in ('prompt_ms', 'predicted_ms')):
                                    raise ValueError('Missing or invalid timing')
                    print(json.dumps(dict(format=fmt, rows=p['reps'] * 3 * TRACES_PER_CLASS)), flush=True)
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


def evaluate(run, prereg=PREREG):
    p = json.loads((run / 'prepared.json').read_text())
    rows = [json.loads(l) for l in (run / 'per_request.jsonl').read_text().splitlines()]
    expected = [(fmt, rep, tid) for fmt in p['formats'] for rep in range(p['reps'])
                for cc in CTX_CLASSES for tid in p['trace_ids'][cc]]
    if [(r['format'], r['rep'], r['trace_id']) for r in rows] != expected:
        raise ValueError('Incomplete or duplicate S2 timing coverage')
    if any(r['prompt_n'] != r['tokenized_prompt_tokens'] or not r['raw_response'].get('stop') for r in rows):
        raise ValueError('Invalid S2 timing request')
    frozen = p.get('preregistration') or {}
    source = Path(frozen.get('source') or prereg)
    margins = registered_margins(source)
    if frozen.get('sha256') and frozen['sha256'] != margins['sha256']:
        raise ValueError('Preregistration changed after cohort freeze')
    summary = {}
    for fmt in p['formats']:
        rs = [r for r in rows if r['format'] == fmt]
        entry = dict(n=len(rs), max_load1=max(r['host_load'][0] for r in rs),
            cycle_ms_median=statistics.median(r['prompt_ms'] + r['predicted_ms'] for r in rs),
            request_wall_ms_median=statistics.median(r['wall_ms'] for r in rs))
        entry['by_ctx_class'] = {}
        for cc in CTX_CLASSES:
            cr = [r for r in rs if r['ctx_class'] == cc]
            entry['by_ctx_class'][cc] = dict(n=len(cr),
                cycle_ms_median=statistics.median(r['prompt_ms'] + r['predicted_ms'] for r in cr),
                predicted_tokens_median=statistics.median(
                    r['raw_response']['timings']['predicted_n'] for r in cr),
                generation_limit_rows=sum(r['raw_response'].get('stop_type') == 'limit' for r in cr))
        summary[fmt] = entry
    base = {(r['rep'], r['ctx_class'], r['trace_id']): r['prompt_ms'] + r['predicted_ms']
            for r in rows if r['format'] == 'Q8_0'}
    speedups = {}
    parity = {}
    for fmt in p['formats']:
        rs = [r for r in rows if r['format'] == fmt]
        speedups[fmt] = dict(
            overall=statistics.median(base[(r['rep'], r['ctx_class'], r['trace_id'])] /
                                      (r['prompt_ms'] + r['predicted_ms']) for r in rs),
            by_ctx_class={cc: statistics.median(
                base[(r['rep'], cc, r['trace_id'])] / (r['prompt_ms'] + r['predicted_ms'])
                for r in rs if r['ctx_class'] == cc) for cc in CTX_CLASSES})
        if fmt != 'Q8_0':
            parity[fmt] = {cc: dict(
                predicted_tokens_median_delta=summary[fmt]['by_ctx_class'][cc]['predicted_tokens_median']
                    - summary['Q8_0']['by_ctx_class'][cc]['predicted_tokens_median'],
                limit_rows_delta=summary[fmt]['by_ctx_class'][cc]['generation_limit_rows']
                    - summary['Q8_0']['by_ctx_class'][cc]['generation_limit_rows'])
                for cc in CTX_CLASSES}
    drift = {cc: 100 * (summary['Q8_0-bookend']['by_ctx_class'][cc]['cycle_ms_median']
                        / summary['Q8_0']['by_ctx_class'][cc]['cycle_ms_median'] - 1) for cc in CTX_CLASSES}
    nominee = 'Q4_K_M_imatrix'
    gates = dict(
        speedup_2k_ge_registered=speedups[nominee]['by_ctx_class']['2k'] >= margins['speedup_min'],
        speedup_8k_ge_registered=speedups[nominee]['by_ctx_class']['8k'] >= margins['speedup_min'],
        bookend_drift_le_registered=all(abs(drift[cc]) <= margins['drift_max_pct'] for cc in CTX_CLASSES),
        token_median_parity=all(abs(v['predicted_tokens_median_delta']) <= 2 for v in parity[nominee].values()),
        token_limit_rows_parity=all(v['limit_rows_delta'] <= 3 for v in parity[nominee].values()))
    gates['all_timing_gates_pass'] = all(gates.values())
    contamination = dict(max_load1=max(s['max_load1'] for s in summary.values()),
        load1_floor=LOAD1_DESCRIPTIVE_FLOOR,
        load1_floor_exceeded=any(s['max_load1'] > LOAD1_DESCRIPTIVE_FLOOR for s in summary.values()),
        note='Descriptive floor: pilot ran at load1 8.3-8.6; one re-run allowed only for documented contamination')
    write(run / 'evaluation.json', dict(rows=len(rows), tier=p['tier'], formats=summary,
        speedups=speedups, token_parity=parity, bookend_drift_pct=drift,
        registered_gates=gates, registered_band=margins, window=contamination))
    write(run / 'verdict.json', dict(verdict='S2-CONFIRMATION-CPU-TIMING-MEASURED', adoption='NOT-ASSESSED',
        boundary='Local 5900X/WSL2 quiet window only; Kaggle timing cannot certify product latency. '
                 'Varying output lengths and cold cache; review token parity before trusting the speedup. '
                 'Exactly one re-run allowed, only for documented window contamination, never for a failed speedup.'))


def smoke(run, assets):
    """CPU-only wiring smoke on synthetic tiny traces; no models, no GPU."""
    def trace(i, cc):
        return dict(trace_id=f'{cc}-{i:02d}', ctx_class=cc, prompt=f'{cc} prompt {i}\n', prompt_tokens=100,
                    target='x\n>>>>>>> UPDATED')
    (assets / 'traces.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in
        [trace(i, cc) for cc in ('2k', '8k') for i in range(TRACES_PER_CLASS)]))
    (assets / 'noop_prompts.jsonl').write_text(''.join(json.dumps(
        dict(id=f'se-{i:02d}', prompt=f'short edit {i}\n')) + '\n' for i in range(TRACES_PER_CLASS)))

    class FakeServer:
        def __init__(self, model, port, flags, **kw):
            self.port, self.log_path = port, run / 'server.log'
        def start(self, ready_timeout=0):
            return True
        def stop(self):
            return None

    timing_ms = {'Q8_0': (1000.0, 500.0), 'Q4_K_M_imatrix': (500.0, 250.0), 'Q8_0-bookend': (1010.0, 505.0)}

    def fake_request(port, prompt):
        fmt = fake_request.current
        prompt_ms, predicted_ms = timing_ms[fmt]
        data = dict(stop=True, stop_type='word', timings=dict(prompt_n=99, prompt_ms=prompt_ms,
            predicted_ms=predicted_ms, predicted_n=35))
        return data, prompt_ms + predicted_ms + 5

    saved_server, saved_tokenize, saved_request = bench.SpecServer, tokenize, request

    class FormatServer(FakeServer):
        def __init__(self, model, port, flags, **kw):
            super().__init__(model, port, flags, **kw)
            # The bookend shares the Q8 file; the format rides on the log path.
            fake_request.current = Path(kw['log_path']).name[len('server-'):-len('.log')]
    try:
        for fmt in FORMATS:
            (assets / (fmt + '.gguf')).write_bytes(b'smoke')
        bench.SpecServer = FormatServer
        globals()['tokenize'] = lambda port, prompt: 99
        globals()['request'] = fake_request
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
        evaluate(run)
    finally:
        bench.SpecServer = saved_server
        globals()['tokenize'] = saved_tokenize
        globals()['request'] = saved_request
    result = json.loads((run / 'evaluation.json').read_text())
    assert result['registered_gates']['all_timing_gates_pass'], result['registered_gates']
    assert abs(result['speedups']['Q4_K_M_imatrix']['by_ctx_class']['2k'] - 2.0) < 1e-9
    print('SMOKE-OK ' + json.dumps(result['registered_gates']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'measure', 'evaluate', 'smoke'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--prereg', type=Path, default=PREREG)
    args = parser.parse_args()
    if args.action == 'evaluate':
        evaluate(args.run, args.prereg)
    elif args.action == 'smoke':
        smoke(args.run, args.assets)
    else:
        globals()[args.action](args.run, args.assets)
