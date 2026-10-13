"""Compare fixed CPT evaluations without confusing rows with documents."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    rows = value['row_metrics']
    assert len(rows) == value['denominators']['validation_rows']
    assert len({r['row_id'] for r in rows}) == len(rows)
    for row in rows:
        assert row['loss_tokens'] > 0
        assert all(math.isfinite(row[k]) for k in ('loss_sum', 'mean_causal_nll'))
        assert math.isclose(row['loss_sum'], row['mean_causal_nll'] * row['loss_tokens'], rel_tol=1e-10)
    tokens = sum(r['loss_tokens'] for r in rows)
    assert tokens == value['denominators']['validation_loss_tokens']
    assert math.isclose(sum(r['loss_sum'] for r in rows) / tokens, value['metrics']['mean_causal_nll'], rel_tol=1e-10)
    return value, hashlib.sha256(raw).hexdigest()


def compare(baseline, candidate):
    a, ah = load(baseline)
    b, bh = load(candidate)
    assert a['denominators'] == b['denominators']
    assert a['case_ids'] == b['case_ids']
    assert a['validation_role'] == b['validation_role'] == 'package_heldout_train_causal_lm'
    packages = defaultdict(lambda: {'loss_tokens': 0, 'baseline_loss_sum': 0.0, 'candidate_loss_sum': 0.0, 'rows': 0})
    for old, new in zip(a['row_metrics'], b['row_metrics']):
        for key in ('row_id', 'document_id', 'package_id', 'loss_tokens'):
            assert old[key] == new[key], f'paired row identity differs: {key}'
        p = packages[old['package_id']]
        p['loss_tokens'] += old['loss_tokens']
        p['baseline_loss_sum'] += old['loss_sum']
        p['candidate_loss_sum'] += new['loss_sum']
        p['rows'] += 1
    panel = []
    for name, values in packages.items():
        delta = (values['candidate_loss_sum'] - values['baseline_loss_sum']) / values['loss_tokens']
        panel.append({'package_id': name, **values, 'delta_nll': delta})
    before, after = (x['metrics']['mean_causal_nll'] for x in (a, b))
    return {
        'schema': 'sepalith.cpt.paired-package-readout.v1',
        'baseline': {'path': str(baseline.resolve()), 'sha256': ah, 'nll': before},
        'candidate': {'path': str(candidate.resolve()), 'sha256': bh, 'nll': after},
        'denominators': a['denominators'],
        'delta_nll': after - before,
        'relative_nll_change': after / before - 1,
        'perplexity_ratio': math.exp(after - before),
        'package_count': len(panel),
        'packages_improved': sum(p['delta_nll'] < 0 for p in panel),
        'packages_worsened': sum(p['delta_nll'] > 0 for p in panel),
        'packages': sorted(panel, key=lambda p: p['delta_nll'], reverse=True),
        'interpretation': 'Paired TRAIN holdout diagnostic. No automatic promotion, significance, or edit-quality claim.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'packages'}))
