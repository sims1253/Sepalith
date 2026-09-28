"""Independent frozen-record join; does not admit training data."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')

def rows(record):
    h = hashlib.sha256()
    n = 0
    with Path(record['path']).open('rb') as stream:
        for line in stream:
            h.update(line)
            n += 1
            yield json.loads(line)
    assert h.hexdigest() == record['sha256'], record['path']
    assert n == record['rows'], (record['path'], n)

counts = collections.Counter()
global_ids = set()
for shard in range(6, 12):
    manifest = json.loads((ROOT / f'shard-{shard:04d}/manifest.json').read_text())
    binding = manifest['streaming_binding']
    provenance = {}
    for row in rows(binding['provenance_ledger']):
        assert row['row_id'] not in provenance
        provenance[row['row_id']] = row
    packets = collections.defaultdict(list)
    for packet in rows(binding['candidate_packet']):
        packets[packet['row_ref']['row_id']].append(packet['row_ref'])
    selected = set(binding['queued_ids'])
    assert len(selected) == binding['queued_rows']
    expected = {rid for rid, r in provenance.items() if r['status'] == 'provenance_pass_semantic_analyzer_queued'}
    assert selected == expected
    observed = set()
    for semantic in rows(manifest['output_binding']):
        rid = semantic['row_id']
        assert rid in selected and rid not in observed and rid not in global_ids
        observed.add(rid)
        global_ids.add(rid)
        original = provenance[rid]
        assert original['source_path_sha256'] == semantic['source_sha256']
        for flag in ['global_train', 'global_group_source_membership', 'protected_disjoint', 'source_line_group_join', 'source_parse_ok', 'source_stat_stable', 'description_stat_stable', 'strict_protocol_ok']:
            assert original[flag] is True, (rid, flag)
        assert original['license_decision']['ok'] is True
        matching = [ref for ref in packets[rid] if ref['family'] == original['family'] and ref['package_id'] == original['package_id'] and ref['group_id'] == original['group_id'] and ref['line'] == original['raw_source_line'] and ref['split'] == 'train_group']
        assert matching, rid
        counts[semantic['status']] += 1
    assert observed == selected
    print(json.dumps({'shard': shard, 'joined_rows': len(observed)}), flush=True)
print(json.dumps({'rows': len(global_ids), 'status_counts': dict(counts), 'source_record_join': 'pass', 'training_admission': False}), flush=True)
