"""Summarize completed paired development probes without opening final data."""
import collections
import json
import pathlib
import statistics

w = pathlib.Path(__file__).resolve().parent
arms = json.loads((w / 'arms.json').read_text())
baseline = None
summary = {'scope': '75 DEV cases and 45 TRAIN cases cold/warm per arm',
           'limitations': ['One sequential pass; latency needs confirmation.',
                           'Exact text is not a semantic correctness assessment.',
                           'TRAIN protocol parser alone does not score semantic no-ops.'],
           'arms': {}}
for arm in arms:
    name = arm['arm']
    source = w if name in ('q8', 'q4_uncalibrated', 'q4_k_m_calibrated') else w.parent / 'r2-quant-development-lowbits-v1'
    terminal = json.loads((source / (name + '-terminal.json')).read_text())
    assert terminal['pid_absent'], name
    dev = [json.loads(line) for line in (source / name / 'dev.jsonl').read_text().splitlines()]
    train = json.loads((source / name / 'train.json').read_text())['requests']
    assert len(dev) == 75 and len({r['row_id'] for r in dev}) == 75
    assert len(train) == 90 and len({(r['row_id'], r['phase']) for r in train}) == 90
    if baseline is None:
        baseline = {r['row_id']: r for r in dev}
    assert set(baseline) == {r['row_id'] for r in dev}
    families = {}
    for family in sorted({r['family'] for r in dev}):
        rows = [r for r in dev if r['family'] == family]
        families[family] = dict(denominator=len(rows), **{
            key: sum(bool(r[key]) for r in rows)
            for key in ['context_protocol_valid', 'exact_edit', 'correct_noop', 'false_suggestion']})
    summary['arms'][name] = {
        'source_packet': str(source),
        'dev_denominator': len(dev), 'edit_denominator': 43, 'noop_denominator': 32,
        **{key: sum(bool(r[key]) for r in dev)
           for key in ['context_protocol_valid', 'exact_edit', 'correct_noop', 'false_suggestion']},
        'families': families,
        'raw_and_token_parity_with_q8': sum(
            r['raw_text'] == baseline[r['row_id']]['raw_text'] and
            r['returned_token_ids'] == baseline[r['row_id']]['returned_token_ids'] for r in dev),
        'train_requests': len(train),
        'train_protocol_status': dict(collections.Counter(r['protocol_status'] for r in train)),
        'train_combined_wall_median_ms': statistics.median(r['combined_case_wall_ms'] for r in train),
        'train_cold_wall_median_ms': statistics.median(r['combined_case_wall_ms'] for r in train if r['phase'] == 'cold'),
        'train_warm_wall_median_ms': statistics.median(r['combined_case_wall_ms'] for r in train if r['phase'] == 'warm'),
    }
reference = summary['arms']['q8']
assert [reference[k] for k in ['context_protocol_valid', 'exact_edit', 'correct_noop', 'false_suggestion']] == [69, 26, 26, 4]
(w / 'root-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
