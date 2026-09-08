#!/usr/bin/env python3
"""Audit archived b4 results and join quality with trace-level cycle timing.

Offline analysis only: no inference, model promotion or source archive edits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ('Q8_0', 'Q4_K_M', 'Q4_K_M_imatrix')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(base, candidate):
    assert base.keys() == candidate.keys()
    losses = sum(base[k] and not candidate[k] for k in base)
    gains = sum(candidate[k] and not base[k] for k in base)
    discord = losses + gains
    p = min(1, 2 * sum(math.comb(discord, i) for i in range(min(losses, gains) + 1)) / 2**discord)
    return dict(n=len(base), losses=losses, gains=gains,
                delta_pp=100 * (gains-losses)/len(base),
                mcnemar_exact_two_sided=p,
                verdict='NO-SIGNIFICANT-DIFFERENCE' if p >= .05 else
                ('WINNER-A' if losses > gains else 'WINNER-B'))


def review(repo):
    records, roots, verification = {}, {}, {}
    for tag in ('b4-quant-quality', 'b4-intent-judge', 'b4-timing-cpu', 'b4-timing-gpu'):
        path = repo / 'docs/validation' / ('2026-09-08-' + tag + '.json')
        record = json.loads(path.read_text())
        root = Path(record['archive'])
        assert digest(root/'closed-manifest.json') == record['closed_manifest_sha256']
        inventory = json.loads((root/'closed-manifest.json').read_text())
        for item in inventory:
            artifact = root/item['path']
            assert artifact.stat().st_size == item['bytes']
            assert digest(artifact) == item['sha256'], str(artifact)
        assert record['status'] == 'succeeded'
        records[tag], roots[tag] = record, root
        verification[tag] = dict(attempt=record['attempt'], files=len(inventory),
                                 record_sha256=digest(path), archive=str(root))
    def rows(tag, filename):
        return [json.loads(line) for line in (roots[tag]/filename).read_text().splitlines()]
    quality = rows('b4-quant-quality', 'requests.jsonl')
    scores = [r for r in rows('b4-intent-judge', 'scores.jsonl') if 'arm' in r]
    comparisons = {}
    for arm in ARMS[1:]:
        comparisons[arm] = {}
        for name, data, field, n in [('scenario_exact', quality, 'exact', 255),
                                    ('scenario_valid', quality, 'valid_pass', 255),
                                    ('intent_satisfied', scores, 'score', 44)]:
            selected = [r for r in data if name.startswith('intent') or r['kind']=='scenario']
            columns = []
            for a in (ARMS[0], arm):
                subset = [r for r in selected if r['arm']==a]
                column = {r['id']: (r[field]==2 if field=='score' else r[field]) for r in subset}
                assert len(column)==len(subset)==n
                columns.append(column)
            comparisons[arm][name] = paired(*columns)
        comparisons[arm]['observed_exact_intent_1pp_gate'] = all(
            comparisons[arm][k]['delta_pp'] >= -1 for k in ('scenario_exact','intent_satisfied'))
    timing = {}
    for tier in ('cpu', 'gpu'):
        data = rows('b4-timing-'+tier, 'per_request.jsonl')
        columns = {}
        for arm in (*ARMS, 'Q8_0-bookend'):
            subset = [r for r in data if r['format']==arm]
            indexed = {(r['trace_id'],r['rep']):r for r in subset}
            assert len(indexed)==len(subset)==30
            traces = sorted({r['trace_id'] for r in subset})
            assert len(traces)==10
            assert set(indexed)=={(t,r) for t in traces for r in range(3)}
            columns[arm] = {t:statistics.median(indexed[t,r]['prompt_ms'] +
                            indexed[t,r]['predicted_ms'] for r in range(3)) for t in traces}
        assert all(c.keys()==columns[ARMS[0]].keys() for c in columns.values())
        timing[tier] = {}
        for arm in (*ARMS[1:], 'Q8_0-bookend'):
            ratios = {t:columns[ARMS[0]][t]/columns[arm][t] for t in columns[arm]}
            timing[tier][arm] = dict(trace_median_speedup=statistics.median(ratios.values()),
                faster_traces=sum(v>1 for v in ratios.values()), trace_speedups=ratios)
    return dict(verification=verification, paired=comparisons, timing=timing,
        operational='Four completed archives verified; original receipts unchanged.',
        scientific='Stock Q4 misses observed exact gate. Imatrix Q4 passes observed exact/intent gate only; validity regression and uncertainty prevent production promotion.',
        adoption='NO-PROMOTION',
        limits=['Ten timing traces, three repeated measurements each; not 30 independent traces.',
                'Cold 2K cycles with variable output lengths; not equal-token throughput.',
                'McNemar difference tests do not prove 1pp noninferiority.',
                'Single calibrated judge; stochastic judge uncertainty unmeasured.',
                'Historical receipts lack Q4 parent hashes; consult separate deterministic reproduction audit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.repo)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
