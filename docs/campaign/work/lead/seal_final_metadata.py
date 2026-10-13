#!/usr/bin/env python3
"""Lock final candidate references without inspecting prompt or target content."""
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
DATA = Path('/mnt/e/sepalith/campaign-20260915/data-work')
DEST = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/sealed-final')
SPLIT_SHA = 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
SOURCE_SHA = '27a1c2c67dcd73155e9c21f7f3dc5e9f76511dd16f10ce151fef5c2451bd49fc'


def checked_json(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, path
    return json.loads(raw)


def fresh_write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def main():
    split = checked_json(DATA / 'DAT-02-global-split-v2.json', SPLIT_SHA)
    sources = checked_json(DATA / 'DAT-01-source-hashes.json', SOURCE_SHA)
    source_map = {item['path']: item for item in sources['files']}
    script = PLAN / 'docs/campaign/work/data/DAT-02-build-split-v2.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == '842eb619079c12215305ee856ddeeb14468ef97654958541c8f9733737b8670a'
    spec = importlib.util.spec_from_file_location('frozen_split_metadata', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity_map = {}
    groups = {group['group_id']: group for group in split['groups']}
    for group in groups.values():
        for token in group['identity_forms'] + group['parent_tokens']:
            assert token not in identity_map or identity_map[token] == group['group_id']
            identity_map[token] = group['group_id']
    selected_groups = {
        key: group for key, group in groups.items()
        if group['split'] == 'final_candidate_group' and group['final_capable_rows'] > 0
    }
    paths = sorted({path for group in selected_groups.values() for path in group['files']
                    if '/cases_v1/' in path or path.endswith('/edit_pairs_v1/examples.jsonl')})
    helper = module.SplitBuilder({})
    references, observed_sources = [], []
    for name in paths:
        path = Path(name)
        expected = source_map[name]
        sha = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for line_number, raw in enumerate(stream, 1):
                sha.update(raw)
                count += len(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                tokens, parents = helper.tokens_for(row, name, line_number)
                matched = {identity_map[token] for token in tokens if token in identity_map}
                if len(matched) != 1:
                    continue
                group_id = next(iter(matched))
                if group_id not in selected_groups:
                    continue
                if '/cases_v1/' in name:
                    boundary = module.present(row, ['package', 'path', 'prefix', 'region_old', 'region_new', 'suffix', 'cursor_idx']) and any(module.nonempty(row, [key]) for key in ['content_hash', 'corpus_key', 'base_sample_id'])
                    family = path.stem
                else:
                    boundary = module.nonempty(row, ['repo', 'sha_full', 'path', 'prefix', 'region_old', 'region_new', 'suffix', 'cursor_idx'])
                    family = 'edit_pairs'
                if not boundary:
                    continue
                references.append({'source_path': name, 'source_sha256': expected['sha256'],
                                   'line': line_number, 'raw_line_sha256': hashlib.sha256(raw).hexdigest(),
                                   'group_id': group_id, 'family': family, 'parent_tokens': parents})
        assert count == expected['bytes'] and sha.hexdigest() == expected['sha256'], name
        observed_sources.append({'path': name, 'bytes': count, 'sha256': sha.hexdigest()})
    assert len(references) == split['counts']['actual_clean_final_candidate_rows'] == 706
    assert len({row['group_id'] for row in references}) == 49
    assert len({(row['source_path'], row['line']) for row in references}) == 706
    references.sort(key=lambda row: (row['group_id'], row['source_path'], row['line']))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        'schema_version': 'sepalith.final-candidate-references.v1', 'created_at': now,
        'status': 'sealed candidate references; semantic admission remains pending freeze',
        'unlock_not_before': '2026-09-14T10:00:00+00:00',
        'requires_weight_and_harness_freeze_receipt': True,
        'split_id': split['split_id'], 'split_sha256': SPLIT_SHA,
        'source_hash_manifest_sha256': SOURCE_SHA, 'source_inventory': observed_sources,
        'groups': list(selected_groups.values()), 'rows': references,
        'counts': {'candidate_rows': 706, 'groups': 49, 'named_packages': 47, 'named_repositories': 3,
                   'families': dict(collections.Counter(row['family'] for row in references))},
        'exclusions': ['train_group', 'dev_group', 'sft_v3/eval and alias/parent overlap',
                       'sft_v7 packages', 'historical scenario and evaluation rows', 'TU3',
                       'teacher-selection and post-training sources', 'missing structured source boundary'],
        'content_access': 'JSON records decoded only to inspect identity metadata and boundary presence; no prompt/target value displayed, rendered, tokenized, generated, scored or used for tuning.',
        'freeze_admission_rules': ['Verify these source/line hashes before extraction.',
                                   'Require source-supported prediction-time geometry and useful evidence.',
                                   'Exclude contradictions, leakage, truncated targets and unsupported operations before model scoring.',
                                   'Apply the frozen renderer/tokenizer equally to every selected candidate and fallback.',
                                   'No replacement based on model outputs; report all exclusions and family/package shortfalls.'],
        'limitations': '706 is a candidate denominator, not 706 validated final examples. No-op and genuine typing/history coverage and separate 2K/4K/8K stress selection remain unresolved; do not mark DAT-08 complete from this lock.'
    }
    DEST.mkdir(mode=0o700, parents=True, exist_ok=False)
    artifact = fresh_write(DEST / 'candidate-references-v1.json', manifest)
    descriptor = fresh_write(DEST / 'seal.json', {'manifest': artifact, 'created_at': now,
                           'unlock_not_before': manifest['unlock_not_before'], 'status': manifest['status']})
    directory = os.open(DEST, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    receipt = {'task': 'DAT-08', 'owner': 'lead', 'observed_at': now,
               'status': 'partial; final candidate metadata locked', 'manifest': artifact,
               'seal': descriptor, 'counts': manifest['counts'],
               'source_files_hash_verified': len(observed_sources),
               'content_access': manifest['content_access'], 'limits': manifest['limitations'],
               'next': 'Complete preregistered no-op/typing and stress selection without tuning on final outcomes; content evaluation remains prohibited before freeze.'}
    receipt_path = PLAN / 'docs/campaign/receipts/DAT-08-final-metadata-lock.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
