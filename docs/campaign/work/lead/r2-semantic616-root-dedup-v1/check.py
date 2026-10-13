import collections
import hashlib
import json
from pathlib import Path

BASE = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
RECEIPT = json.loads((BASE / 'receipts/DAT-10-semantic763-dedup-integration.json').read_text())

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def stream(spec):
    h = hashlib.sha256()
    count = 0
    with Path(spec['path']).open('rb') as file:
        for line in file:
            h.update(line)
            count += 1
            yield json.loads(line)
    assert h.hexdigest() == spec['sha256'], spec['path']
    assert count == spec['rows'], (count, spec['rows'])

def source_keys(value):
    hashes, paths = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                if key in {'source_sha256', 'source_snapshot_sha256', 'before_snapshot_sha256', 'after_snapshot_sha256', 'post_edit_snapshot_sha256'}:
                    hashes.add(item)
                if key in {'source_path', 'source_snapshot_path', 'path', 'event_path'}:
                    path = item.replace('\\', '/')
                    paths.add('R/' + path.split('/R/', 1)[1] if '/R/' in path else path)
            elif isinstance(item, dict):
                nested_hashes, nested_paths = source_keys(item)
                hashes.update(nested_hashes)
                paths.update(nested_paths)
    return hashes, paths

def compact(row):
    return {'id': row['id'], 'pair': digest(row['prompt_text'] + '\0' + row['target_text']), 'prompt': digest(row['prompt_text']), 'target': digest(row['target_body_text']), 'package': row['package_id']}

existing = {}
for row in stream(RECEIPT['inputs']['authoritative_15006']):
    assert row['id'] not in existing
    existing[row['id']] = compact(row)
side_ids = set()
for row in stream(RECEIPT['inputs']['authoritative_context']):
    rid = row['row_id']
    assert rid not in side_ids and rid in existing
    side_ids.add(rid)
    existing[rid]['sources'] = source_keys(row['source_identity'])
assert side_ids == set(existing)
provenance = {row['row_id']: row for row in stream(RECEIPT['outputs']['candidate_provenance'])}
indexes = {name: collections.defaultdict(set) for name in ['id', 'pair', 'prompt', 'source_target', 'path_target']}

def keys(row):
    yield 'id', row['id']
    yield 'pair', row['pair']
    yield 'prompt', row['prompt']
    for sha in row['sources'][0]:
        yield 'source_target', (sha, row['target'])
    for path in row['sources'][1]:
        yield 'path_target', (row['package'], path, row['target'])

for row in existing.values():
    for kind, key in keys(row):
        indexes[kind][key].add(row['id'])
findings = []
candidate_ids = set()
for original in stream(RECEIPT['outputs']['candidate_tokenrows']):
    row = compact(original)
    assert row['id'] not in candidate_ids
    candidate_ids.add(row['id'])
    row['sources'] = source_keys(provenance[row['id']]['source_identity'])
    for kind, key in keys(row):
        if indexes[kind].get(key):
            findings.append({'id': row['id'], 'kind': kind, 'matches': sorted(indexes[kind][key])})
    for kind, key in keys(row):
        indexes[kind][key].add(row['id'])
assert candidate_ids == set(provenance)
result = {'existing_rows': len(existing), 'candidate_rows': len(candidate_ids), 'findings': findings, 'input_hashes_verified': True, 'source_key_scope': 'recursive identity including event_path; exact target bytes; no normalization of target', 'training_admission': False}
print(json.dumps(result, sort_keys=True))
assert not findings, 'Duplicate or conflict requires review'
