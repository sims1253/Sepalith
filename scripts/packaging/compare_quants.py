#!/usr/bin/env python3
"""Pair scenario-harness result files; reject missing or mismatched examples."""
import argparse
import json
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    index = {r['id']: r for r in rows}
    if not rows or len(index) != len(rows):
        raise ValueError(f'Empty or duplicate evaluation rows: {path}')
    if any(r.get('error') or r.get('fail_kind') == 'error' for r in rows):
        raise ValueError(f'Transport errors invalidate comparison: {path}')
    validate_scores(index)
    return index


def validate_scores(rows):
    if not rows:
        raise ValueError('Comparison requires nonempty evaluation rows')
    for row in rows.values():
        for metric in ('exact', 'valid_pass'):
            value = row.get(metric)
            if type(value) not in (bool, int) or value not in (0, 1):
                raise ValueError(f'{metric} must be a binary outcome')
        if row.get('error') or row.get('fail_kind') == 'error':
            raise ValueError('Transport errors invalidate comparison')


def compare(baseline, candidate):
    validate_scores(baseline)
    validate_scores(candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError('Evaluation row IDs differ between arms')
    n = len(baseline)
    result = {'n': n}
    for metric in ('exact', 'valid_pass'):
        a = sum(r[metric] for r in baseline.values())
        b = sum(r[metric] for r in candidate.values())
        result[metric] = dict(baseline=a/n, candidate=b/n, delta_pp=100*(b-a)/n,
                             lost=sum(bool(baseline[k][metric]) and not candidate[k][metric] for k in baseline),
                             gained=sum(bool(candidate[k][metric]) and not baseline[k][metric] for k in baseline))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('candidates', nargs='+', type=Path)
    args = ap.parse_args()
    baseline = load(args.baseline)
    print(json.dumps({str(p): compare(baseline, load(p)) for p in args.candidates}, indent=2))
