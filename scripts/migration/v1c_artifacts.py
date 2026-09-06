#!/usr/bin/env python3
"""Validate and archive a V1c serving measurement without promoting a model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def prepare(run, traces):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
    import spec_bench
    by_class = spec_bench.load_traces(traces)
    for cc in ('2k', '8k'):
        ids = [r['trace_id'] for r in by_class[cc]]
        if len(ids) < 64 or len(ids) != len(set(ids)):
            raise ValueError(f'Insufficient or duplicate {cc} traces')
    write(run / 'prepared.json', {
        'trace_sha256': digest(traces), 'platform': platform.platform(),
        'python': platform.python_version(),
        'serial_ids': {cc: [r['trace_id'] for r in spec_bench.sample_traces(rs, 50)]
                       for cc, rs in by_class.items()},
        'sweep_ids': {cc: [r['trace_id'] for r in spec_bench.sample_traces(rs, 64)]
                      for cc, rs in by_class.items()},
        'scope': 'v7 serving column; full V1c calibration is separate',
    })


def evaluate(run):
    prepared = json.loads((run / 'prepared.json').read_text())
    result = run / 'measurement'
    rows = {}
    for leg in ('ttft', 'sweep'):
        rows[leg] = [json.loads(line) for line in
                     (result / f'results_v1c_{leg}.jsonl').read_text().splitlines()]
    expected = {
        'ttft': {(cc, i): trace for cc, ids in prepared['serial_ids'].items()
                 for i, trace in enumerate(ids)},
        'sweep': {(cc, level, stream, req): ids[(stream * 10 + req) % len(ids)]
                  for cc, ids in prepared['sweep_ids'].items()
                  for level in (1, 2, 4) for stream in range(level) for req in range(10)},
    }
    failures = []
    for leg, records in rows.items():
        keys = [(r['ctx_class'], r['i']) if leg == 'ttft' else
                (r['ctx_class'], r['level'], r['stream'], r['req']) for r in records]
        if len(set(keys)) != len(keys) or set(keys) != set(expected[leg]):
            raise ValueError(f'{leg}: missing, duplicate or unexpected requests')
        for key, row in zip(keys, records):
            if row['trace_id'] != expected[leg][key] or row['leg'] != leg:
                raise ValueError(f'{leg}: trace alignment mismatch')
            for field in ('wall_ms', 'tokens'):
                if not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError(f'{leg}: invalid {field}')
            ttft = row.get('ttft_ms')
            if ttft is not None and (not isinstance(ttft, (int, float)) or not math.isfinite(ttft) or ttft < 0):
                raise ValueError('Invalid TTFT')
            if row.get('error') or row.get('starved') or ttft is None or not (
                    row.get('completed') if leg == 'ttft' else row.get('completed') or row.get('aborted')):
                failures.append({'leg': leg, 'key': key, 'error': row.get('error'),
                                 'starved': row.get('starved')})
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/eval'))
    import latency_load
    computed = {'ttft': latency_load.summarize_ttft(rows['ttft']),
                'sweep': latency_load.summarize_sweep(rows['sweep'])}
    reported = json.loads((result / 'summary.json').read_text())
    if any(reported[k] != v for k, v in computed.items()):
        raise ValueError('Summary disagrees with raw rows')
    write(run / 'evaluation.json', {'counts': {k: len(v) for k, v in rows.items()},
                                   'failures': failures, **computed})
    verdict = 'MEASUREMENT-COMPLETE' if not failures else 'INCONCLUSIVE'
    write(run / 'verdict.json', {'verdict': verdict, 'adoption': 'NOT-ASSESSED',
          'scope': 'v7 serving column only',
          'criteria': 'All 100 serial and 140 sweep requests aligned; no error or starvation; finite TTFT.',
          'limits': ['Quiet-window evidence requires operator review.',
                     'Full V1c needs qwen/minicpm ordering and v8_2/base parity calibration.',
                     'Chunk counts are client-observed abort-waste proxies, not server token accounting.']})
    (run / 'VERDICT.md').write_text(
        f'{verdict}\n\nAdoption: NOT-ASSESSED. This is the v7 serving column.\n'
        'Full V1c calibration and quiet-window review remain separate.\n')


def archive(run, archive_root):
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / run.name
    if target.exists():
        raise ValueError('Archive destination exists; preserve it and use a new attempt')
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=archive_root) as tmp:
        staging = Path(tmp) / 'attempt'
        staging.mkdir()
        inventory = []
        # The final step log is still open. Retain all other closed records and
        # the complete source. Final runner receipts remain in local run state.
        for path in sorted(run.rglob('*')):
            if path.is_symlink():
                raise ValueError('Archive refuses symlinks')
            if not path.is_file() or path.name in ('03-archive.log', 'archive.json'):
                continue
            rel = path.relative_to(run)
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            before = digest(path)
            shutil.copyfile(path, dest)
            if digest(dest) != before or digest(path) != before:
                raise ValueError(f'Archive copy changed: {rel}')
            with dest.open('rb') as f:
                os.fsync(f.fileno())
            inventory.append({'path': str(rel), 'sha256': before, 'bytes': dest.stat().st_size})
        write(staging / 'archive-manifest.json', inventory)
        staging.rename(target)
    write(run / 'archive.json', {'path': str(target), 'files': inventory,
          'manifest_sha256': digest(target / 'archive-manifest.json'),
          'boundary': 'Before runner final success; final execution/step receipt remains in local state.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate', 'archive'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run, args.traces)
    elif args.action == 'evaluate':
        evaluate(args.run)
    else:
        archive(args.run, args.archive)
