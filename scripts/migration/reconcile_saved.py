#!/usr/bin/env python3
"""Recompute saved benchmark evidence without model execution or API calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/eval'))
import latency_load
import spec_bench


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reconcile(evaluation, traces, ablation):
    files = {}

    def capture(relative):
        path = evaluation / relative
        data = path.read_bytes()
        files[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return read(path)

    records = capture('results_specbench/run-20260906T044638-n100-r3/per_request.jsonl')
    rows = [r for r in records if 'arm' in r]
    keys = [(r['arm'], r['ctx_class'], r['trace_id'], r['rep']) for r in rows]
    if len(keys) != 1800 or len(set(keys)) != 1800:
        raise ValueError('S1 request coverage differs from the trimmed recipe')
    s1 = {}
    for arm in ('baseline', 'ngram-simple@2', 'model-draft@2'):
        for cc in ('2k', '8k'):
            group = [r for r in rows if r['arm'] == arm and r['ctx_class'] == cc]
            if len(group) != 300 or Counter(r['rep'] for r in group) != {0: 100, 1: 100, 2: 100}:
                raise ValueError('S1 arm/class coverage mismatch')
            s1[f'{arm}|{cc}'] = {
                'n': len(group), 'errors': sum(bool(r.get('error')) for r in group),
                'gen_tps_median': statistics.median(r['gen_tps'] for r in group),
                'ttft_ms_median': statistics.median(r['ttft_ms'] for r in group),
                'warm_ttft_ms_median': statistics.median(r['warm_ttft_ms'] for r in group),
                'recorded_matches': sum(r.get('matches_baseline') is True for r in group),
                'mismatching_traces': len({r['trace_id'] for r in group if r.get('matches_baseline') is False}),
            }
    quant = capture('results_quantserve/per_request.jsonl')
    keys = [(r['format'], r['trace_id'], r['rep']) for r in quant]
    if len(keys) != 150 or len(set(keys)) != 150:
        raise ValueError('S2 request coverage mismatch')
    s2 = {}
    for fmt in sorted({r['format'] for r in quant}):
        group = [r for r in quant if r['format'] == fmt]
        s2[fmt] = {'n': len(group), 'errors': sum(bool(r.get('error')) for r in group),
                   'cycle_ms_median': statistics.median(r['keystroke_cycle_ms'] for r in group)}
    serial = capture('results_v1c_ttft.jsonl')
    sweep = capture('results_v1c_sweep.jsonl')
    selected = spec_bench.load_traces(traces)
    for leg, records in [('ttft', serial), ('sweep', sweep)]:
        expected = {}
        for cc, group in selected.items():
            ids = [r['trace_id'] for r in spec_bench.sample_traces(group, 50 if leg == 'ttft' else 64)]
            if leg == 'ttft':
                expected.update({(cc, i): t for i, t in enumerate(ids)})
            else:
                expected.update({(cc, level, stream, req): ids[(stream * 10 + req) % len(ids)]
                                 for level in (1, 2, 4) for stream in range(level) for req in range(10)})
        actual = {}
        for row in records:
            key = (row['ctx_class'], row['i']) if leg == 'ttft' else (
                row['ctx_class'], row['level'], row['stream'], row['req'])
            if key in actual:
                raise ValueError('Duplicate V1c request')
            actual[key] = row['trace_id']
        if actual != expected:
            raise ValueError('V1c deterministic trace selection differs')
    saved = [r for r in capture('results_ablation_v7_on_plain.jsonl') if 'i' in r]
    examples = read(ablation)
    ids = [r['i'] for r in saved]
    if len(ids) != len(set(ids)) or any(i < 0 or i >= len(examples) for i in ids):
        raise ValueError('Invalid saved W33 indices')
    aligned = sum(r['package'] == (examples[r['i']].get('package') or examples[r['i']].get('package_or_repo'))
                  for r in saved)
    return {'schema_version': 1, 'source_files': files,
            's1': s1, 's2': s2,
            'v1c': {'ttft': latency_load.summarize_ttft(serial),
                    'sweep': latency_load.summarize_sweep(sweep),
                    'coverage': {'ttft': len(serial), 'sweep': len(sweep)},
                    'trace_selection_verified': True},
            'w33': {'input_rows': len(examples), 'saved_rows': len(saved),
                    'empty': sum(bool(r.get('empty')) for r in saved),
                    'errors': sum(bool(r.get('error')) for r in saved),
                    'package_aligned_rows': aligned,
                    'missing_indices': sorted(set(range(len(examples))) - set(ids)) if aligned == len(saved) else None,
                    'provenance': 'Saved file has no model/input/runtime hashes; coverage is not provenance verification.'},
            'verdicts': {'s1': 'No adoption for measured CPU depth-2 arms; broader depth/MTP/GPU scope untested.',
                         's2': 'Timing collected; quality gate unverified, no ship-matrix promotion.',
                         'v1c': 'Saved v7 measurement only; calibration and adoption unassessed.',
                         'w33': 'Saved rows do not establish the historical input identity. Repair requires aligned inputs.'}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'traces', 'ablation', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    result = reconcile(a.evaluation, a.traces, a.ablation)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
