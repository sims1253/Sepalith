#!/usr/bin/env python3
"""Inventory all supplemental TRAIN source categories without reading payloads.

The frozen global package order and both split ledgers are checked before any
normalized package tree is opened.  This pass records path/stat/license
metadata only.  It deliberately does not tokenize, hash, or write source
payloads; the next pass can read only metadata-eligible regular R files after
the inventory is reviewed.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import concurrent.futures
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
ORDER = PLAN / 'docs/campaign/work/r2-cpt-long-stage-review-v1/next-global-package-order.json'
SPLIT = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
PARTITION = PLAN / 'docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json'
NORMALIZED = Path('/mnt/h/sepalith/normalized')
DEFAULT_OUTPUT = HERE / 'metadata-inventory-v1'
ORDER_SHA = 'c7255cbbb664b7a12322b7bb9acf40c1b23c73dc9e1869531443205dbc9335b8'
SPLIT_SHA = 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
PARTITION_SHA = '6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06'
EXPECTED_GROUPS = 8867
SUPPORTED_SUFFIXES = {'.r', '.rmd', '.qmd', '.rnw', '.rd'}
RELEVANT_CATEGORIES = {
    '.r': 'regular_R',
    '.rmd': 'literate_Rmd',
    '.qmd': 'literate_qmd',
    '.rnw': 'literate_Rnw',
    '.rd': 'Rd_documentation',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_description(raw: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    key: str | None = None
    for line in raw.decode('utf-8').splitlines():
        if line[:1].isspace() and key:
            fields[key] += ' ' + line.strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            fields[key] = value.strip()
        else:
            key = None
    return fields


def recognized_license(value: str) -> bool:
    # Keep this gate textually aligned with raw_cpt_broader.allowed_license.
    import re
    return bool(re.search(r'(^|[ |+,(])(?:A?GPL|LGPL|MIT|BSD|Apache|Artistic|MPL|CC0|Unlimited)(?:[- (]|$)', value))


def registry_from_split(split: dict) -> dict[str, tuple[str, str]]:
    registry: dict[str, tuple[str, str]] = {}
    for group in split['groups']:
        value = (group['group_id'], group['split'])
        for identity in group.get('identity_forms', []):
            if not identity.startswith('pkg:'):
                continue
            key = identity[4:].lower()
            if key in registry and registry[key] != value:
                raise ValueError(f'ambiguous package registry identity: {key}')
            registry[key] = value
    return registry


def preflight() -> tuple[list[dict], dict[str, tuple[str, str]], dict[str, str], dict]:
    if sha(ORDER) != ORDER_SHA:
        raise ValueError('global package order hash mismatch')
    if sha(SPLIT) != SPLIT_SHA:
        raise ValueError('global split hash mismatch')
    if sha(PARTITION) != PARTITION_SHA:
        raise ValueError('CPT partition hash mismatch')
    order = json.loads(ORDER.read_text(encoding='utf-8'))
    entries = order['packages']
    if len(entries) != EXPECTED_GROUPS or order.get('eligible_next_count') != EXPECTED_GROUPS:
        raise ValueError('frozen order extent differs')
    split = json.loads(SPLIT.read_text(encoding='utf-8'))
    registry = registry_from_split(split)
    partition_data = json.loads(PARTITION.read_text(encoding='utf-8'))
    parts = partition_data['groups']
    if sum(value == 'cpt_validation' for value in parts.values()) != 556:
        raise ValueError('CPT validation count differs')
    for entry in entries:
        if entry.get('split') != 'train_group':
            raise ValueError(f'non-TRAIN entry reached supplemental scope: {entry}')
        if registry.get(entry['name'].lower()) != (entry['group_id'], 'train_group'):
            raise ValueError(f'global TRAIN registry mismatch: {entry}')
        if parts.get(entry['group_id']) != 'cpt_train':
            raise ValueError(f'CPT validation group reached supplemental scope: {entry}')
    manifest = {
        'schema': 'DAT-10-cpt-supplemental-source-preflight-v1',
        'status': 'pass',
        'groups': EXPECTED_GROUPS,
        'scope': 'all frozen eligible global TRAIN groups (initial 775 plus remaining 8092)',
        'supplemental_regular_category': 'case-insensitive regular .R/.r outside normalized package/version/package/R',
        'separate_literate_categories': ['.Rmd', '.qmd', '.Rnw', 'Rd'],
        'payload_read': False,
        'payload_written': False,
        'heldout_payload_read': False,
        'pins': {str(ORDER): ORDER_SHA, str(SPLIT): SPLIT_SHA, str(PARTITION): PARTITION_SHA},
        'no_caps': {'groups': None, 'families': None, 'files': None, 'bytes': None, 'wall_time': None},
        'license_gate': 'metadata-only frozen recognized-license classification; unresolved licenses remain named review status',
        'dedup': 'content hash deferred to reviewed regular-R payload pass; no dedup claim in this metadata pass',
    }
    return entries, registry, parts, manifest


def category_for(path: Path, source: Path) -> tuple[str, str, bool]:
    relative = path.relative_to(source)
    suffix = path.suffix.lower()
    category = RELEVANT_CATEGORIES.get(suffix)
    inside_r = bool(relative.parts and relative.parts[0] == 'R')
    if category is None:
        return 'other_regular', 'inside_package_R' if inside_r else 'outside_package_R', False
    scope = 'inside_package_R' if inside_r else 'outside_package_R'
    if category == 'regular_R' and inside_r:
        return 'regular_R_under_package_R_existing_main', scope, False
    if category == 'regular_R':
        return 'regular_R_outside_package_R_supplemental_candidate', scope, True
    return category, scope, False


def metadata_file(entry: dict, source: Path, path: Path, description: dict, description_sha: str,
                  package_status: str) -> dict | None:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as error:
        return {
            'schema': 'DAT-10-cpt-supplemental-file-metadata-v1',
            'package': entry['name'], 'group_id': entry['group_id'], 'split': 'train_group',
            'cpt_partition': 'cpt_train', 'path': str(path),
            'relative_path': str(path.relative_to(source)), 'category': 'metadata_stat_failure',
            'scope': 'unknown', 'status': 'repair_required_metadata_stat_failure',
            'error': type(error).__name__, 'payload_read': False, 'content_sha256': None,
        }
    if not stat.S_ISREG(state.st_mode):
        return None
    category, scope, direct_candidate = category_for(path, source)
    if category == 'other_regular':
        return None
    relative = path.relative_to(source)
    path_id = hashlib.sha256(('DAT10-CPT-supplemental-file-v1\0' + entry['group_id'] + '\0' + str(relative)).encode()).hexdigest()
    status = 'metadata_eligible_payload_review'
    if package_status != 'metadata_eligible_recognized_license':
        status = 'blocked_by_package_' + package_status
    elif category == 'regular_R_under_package_R_existing_main':
        status = 'existing_main_category'
    elif category != 'regular_R_outside_package_R_supplemental_candidate':
        status = 'separate_extraction_evidence_review'
    return {
        'schema': 'DAT-10-cpt-supplemental-file-metadata-v1',
        'source_id': path_id,
        'package': entry['name'], 'group_id': entry['group_id'], 'split': 'train_group',
        'cpt_partition': 'cpt_train', 'path': str(path), 'relative_path': str(relative),
        'category': category, 'scope': scope, 'direct_regular_R_candidate': direct_candidate,
        'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns, 'inode': state.st_ino,
        'device': state.st_dev, 'description_sha256': description_sha,
        'license': description.get('License', ''), 'license_status': package_status,
        'size_status': 'long_file_profile_required' if state.st_size > 4 * 1024 * 1024 else 'within_profile_file_size',
        'content_sha256': None, 'dedup_status': 'pending_payload_hash',
        'payload_read': False, 'payload_written': False,
    }


def inventory_group(entry: dict, registry: dict[str, tuple[str, str]], parts: dict[str, str]) -> dict:
    group = entry['group_id']
    package = entry['name']
    record = {
        'schema': 'DAT-10-cpt-supplemental-group-inventory-v1',
        'package': package, 'group_id': group, 'split': entry['split'], 'cpt_partition': parts[group],
        'seeded_order_index': None, 'package_root': str(NORMALIZED / package),
        'metadata_status': 'unknown', 'description_sha256': None, 'license': None,
        'files': [], 'category_counts': {}, 'category_bytes': {}, 'walk_errors': [],
        'payload_read': False, 'payload_written': False, 'heldout_payload_read': False,
    }
    package_root = NORMALIZED / package
    try:
        versions = sorted((item for item in os.scandir(package_root) if item.is_dir(follow_symlinks=False)), key=lambda item: item.name)
    except OSError as error:
        record['metadata_status'] = 'package_root_read_failure'
        record['walk_errors'].append({'path': str(package_root), 'reason': type(error).__name__})
        return record
    if len(versions) != 1:
        record['metadata_status'] = 'package_layout_version_count_not_one'
        record['walk_errors'].append({'path': str(package_root), 'reason': 'version_count_not_one', 'count': len(versions)})
        return record
    version = versions[0].name
    source = package_root / version / package
    description_path = source / 'DESCRIPTION'
    description = {}
    description_sha = None
    try:
        state_before = description_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(state_before.st_mode) or description_path.is_symlink():
            raise ValueError('DESCRIPTION not regular')
        raw = description_path.read_bytes()
        state_after = description_path.stat(follow_symlinks=False)
        if (state_before.st_dev, state_before.st_ino, state_before.st_size, state_before.st_mtime_ns) != (state_after.st_dev, state_after.st_ino, state_after.st_size, state_after.st_mtime_ns):
            raise ValueError('DESCRIPTION changed during read')
        description = parse_description(raw)
        description_sha = hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, ValueError) as error:
        record['metadata_status'] = 'description_read_or_decode_failure'
        record['walk_errors'].append({'path': str(description_path), 'reason': type(error).__name__})
    else:
        record['description_sha256'] = description_sha
        record['license'] = description.get('License', '')
        if description.get('Package') != package:
            record['metadata_status'] = 'description_package_mismatch'
        elif not record['license'] or description.get('License_restricts_use', '').lower() == 'yes' or description.get('License_is_FOSS', '').lower() == 'no':
            record['metadata_status'] = 'license_requires_review'
        elif not recognized_license(record['license']):
            record['metadata_status'] = 'license_not_in_frozen_recognized_families'
        else:
            record['metadata_status'] = 'metadata_eligible_recognized_license'
    if not source.is_dir() or source.is_symlink():
        record['metadata_status'] = 'package_source_root_missing_or_symlink'
        record['walk_errors'].append({'path': str(source), 'reason': 'source_root_missing_or_symlink'})
        return record
    record['source_root'] = str(source)
    stack = [source]
    counts = collections.Counter()
    bytes_by_category = collections.Counter()
    while stack:
        current = stack.pop()
        try:
            children = sorted(os.scandir(current), key=lambda item: item.name, reverse=True)
        except OSError as error:
            record['walk_errors'].append({'path': str(current), 'reason': type(error).__name__})
            counts['directory_read_failure'] += 1
            continue
        for item in children:
            path = Path(item.path)
            try:
                # Directory detection is required for recursion.  For every
                # other entry, inspect the suffix before doing a second stat;
                # the metadata pass only needs source-format files.
                if item.is_dir(follow_symlinks=False):
                    counts['directory'] += 1
                    stack.append(path)
                    continue
                suffix = path.suffix.lower()
                if item.is_symlink():
                    if suffix in SUPPORTED_SUFFIXES:
                        relative = path.relative_to(source)
                        category = 'symlink_' + RELEVANT_CATEGORIES[suffix]
                        counts[category] += 1
                        record['files'].append({
                            'schema': 'DAT-10-cpt-supplemental-file-metadata-v1',
                            'source_id': hashlib.sha256(('DAT10-CPT-supplemental-file-v1\0' + group + '\0' + str(relative)).encode()).hexdigest(),
                            'package': package, 'group_id': group, 'split': 'train_group', 'cpt_partition': 'cpt_train',
                            'path': str(path), 'relative_path': str(relative), 'category': category,
                            'scope': 'inside_package_R' if relative.parts and relative.parts[0] == 'R' else 'outside_package_R',
                            'status': 'repair_required_symlink', 'payload_read': False, 'payload_written': False,
                            'content_sha256': None,
                        })
                    else:
                        counts['symlink_other'] += 1
                elif suffix not in SUPPORTED_SUFFIXES:
                    counts['other_regular'] += 1
                elif item.is_file(follow_symlinks=False):
                    value = metadata_file(entry, source, path, description, description_sha, record['metadata_status'])
                    if value is None:
                        counts['other_regular'] += 1
                        continue
                    counts[value['category']] += 1
                    bytes_by_category[value['category']] += value.get('bytes', 0)
                    record['files'].append(value)
                else:
                    counts['nonregular'] += 1
            except OSError as error:
                record['walk_errors'].append({'path': str(path), 'reason': type(error).__name__})
                counts['entry_stat_failure'] += 1
    record['version'] = version
    record['category_counts'] = dict(sorted(counts.items()))
    record['category_bytes'] = dict(sorted(bytes_by_category.items()))
    record['files'].sort(key=lambda item: item.get('relative_path', ''))
    return record


def summarize(group_records: list[dict], elapsed: float) -> dict:
    totals = collections.Counter()
    bytes_by_category = collections.Counter()
    status_counts = collections.Counter()
    groups_by_status = collections.Counter()
    package_errors = []
    relevant_files = 0
    candidate_files = 0
    for record in group_records:
        groups_by_status[record['metadata_status']] += 1
        if record.get('walk_errors'):
            package_errors.extend(record['walk_errors'])
        for category, count in record.get('category_counts', {}).items():
            totals[category] += count
        for category, value in record.get('category_bytes', {}).items():
            bytes_by_category[category] += value
        for row in record.get('files', []):
            relevant_files += 1
            status_counts[row.get('status', 'missing_status')] += 1
            candidate_files += int(row.get('direct_regular_R_candidate', False))
    return {
        'schema': 'DAT-10-cpt-supplemental-category-summary-v1',
        'status': 'metadata_complete',
        'groups': len(group_records), 'expected_groups': EXPECTED_GROUPS,
        'initial_order_groups': min(775, len(group_records)),
        'remaining_order_groups': max(0, len(group_records) - 775),
        'relevant_file_rows': relevant_files, 'direct_regular_R_candidate_files': candidate_files,
        'category_counts': dict(sorted(totals.items())),
        'category_bytes': dict(sorted(bytes_by_category.items())),
        'group_metadata_status_counts': dict(sorted(groups_by_status.items())),
        'file_status_counts': dict(sorted(status_counts.items())),
        'package_walk_error_count': len(package_errors),
        'payload_read': False, 'payload_written': False, 'heldout_payload_read': False,
        'elapsed_seconds': round(elapsed, 3),
    }


def new_summary_state() -> dict:
    return {
        'groups': 0, 'totals': collections.Counter(), 'bytes_by_category': collections.Counter(),
        'status_counts': collections.Counter(), 'groups_by_status': collections.Counter(),
        'relevant_files': 0, 'candidate_files': 0, 'package_errors': 0,
    }


def update_summary_state(state: dict, record: dict) -> None:
    state['groups'] += 1
    state['groups_by_status'][record['metadata_status']] += 1
    state['package_errors'] += len(record.get('walk_errors', []))
    for category, count in record.get('category_counts', {}).items():
        state['totals'][category] += count
    for category, value in record.get('category_bytes', {}).items():
        state['bytes_by_category'][category] += value
    for row in record.get('files', []):
        state['relevant_files'] += 1
        state['status_counts'][row.get('status', 'missing_status')] += 1
        state['candidate_files'] += int(row.get('direct_regular_R_candidate', False))


def summary_from_state(state: dict, elapsed: float, status: str = 'metadata_walk_in_progress') -> dict:
    return {
        'schema': 'DAT-10-cpt-supplemental-category-summary-v1', 'status': status,
        'groups': state['groups'], 'expected_groups': EXPECTED_GROUPS,
        'initial_order_groups': min(775, state['groups']),
        'remaining_order_groups': max(0, state['groups'] - 775),
        'relevant_file_rows': state['relevant_files'], 'direct_regular_R_candidate_files': state['candidate_files'],
        'category_counts': dict(sorted(state['totals'].items())),
        'category_bytes': dict(sorted(state['bytes_by_category'].items())),
        'group_metadata_status_counts': dict(sorted(state['groups_by_status'].items())),
        'file_status_counts': dict(sorted(state['status_counts'].items())),
        'package_walk_error_count': state['package_errors'],
        'payload_read': False, 'payload_written': False, 'heldout_payload_read': False,
        'elapsed_seconds': round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 2:
        raise ValueError('--workers must be between 1 and 2')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    try:
        os.nice(10)
    except OSError:
        pass
    entries, registry, parts, manifest = preflight()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'groups').mkdir(exist_ok=True)
    write_json(output / 'preflight.json', manifest)
    write_json(output / 'run-manifest.json', {**manifest, 'inventory_script_sha256': sha(Path(__file__))})
    started = time.monotonic()
    state = new_summary_state()
    missing: list[tuple[int, dict, Path]] = []
    for index, entry in enumerate(entries):
        destination = output / 'groups' / f'{index:06d}-{entry["group_id"]}.json'
        if destination.exists():
            record = json.loads(destination.read_text(encoding='utf-8'))
            if record.get('group_id') != entry['group_id']:
                raise ValueError(f'group shard identity mismatch: {destination}')
            if record.get('seeded_order_index') != index:
                raise ValueError(f'group shard order identity mismatch: {destination}')
            update_summary_state(state, record)
        else:
            missing.append((index, entry, destination))

    def commit(index: int, entry: dict, destination: Path, record: dict) -> None:
        if record.get('group_id') != entry['group_id']:
            raise ValueError(f'worker group identity mismatch: {entry}')
        record['seeded_order_index'] = index
        temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination)
        update_summary_state(state, record)

    completed = state['groups']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inventory_group, entry, registry, parts): (index, entry, destination)
                   for index, entry, destination in missing}
        for future in concurrent.futures.as_completed(futures):
            index, entry, destination = futures[future]
            commit(index, entry, destination, future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(entries):
                partial = summary_from_state(state, time.monotonic() - started)
                write_json(output / 'progress.json', {
                    'schema': 'DAT-10-cpt-supplemental-progress-v1', 'status': 'metadata_walk_in_progress',
                    'groups_done': completed, 'groups_total': len(entries), **partial,
                    'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                })
                print(canonical({'stage': 'metadata_walk', 'groups_done': completed, 'groups_total': len(entries),
                                 'relevant_file_rows': partial['relevant_file_rows'],
                                 'direct_regular_R_candidate_files': partial['direct_regular_R_candidate_files'],
                                 'elapsed_seconds': partial['elapsed_seconds']}), flush=True)
    if completed != len(entries):
        raise ValueError(f'completed group count differs: {completed}')
    # Keep the final ledgers in frozen order, while retaining only one group
    # record in memory during the merge.  The per-group shards are resumable.
    package_path = output / 'package-inventory.jsonl'
    source_path = output / 'source-inventory.jsonl'
    with package_path.open('w', encoding='utf-8') as package_stream, source_path.open('w', encoding='utf-8') as source_stream:
        for index, entry in enumerate(entries):
            destination = output / 'groups' / f'{index:06d}-{entry["group_id"]}.json'
            record = json.loads(destination.read_text(encoding='utf-8'))
            package_stream.write(canonical({key: value for key, value in record.items() if key != 'files'}) + '\n')
            for row in record.get('files', []):
                source_stream.write(canonical(row) + '\n')
    for path in (package_path, source_path):
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    summary = summary_from_state(state, time.monotonic() - started, status='metadata_complete')
    write_json(output / 'category-summary.json', summary)
    write_json(output / 'progress.json', {
        'schema': 'DAT-10-cpt-supplemental-progress-v1', 'status': 'metadata_complete',
        'groups_done': len(records), 'groups_total': len(entries), **summary,
        'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    print(canonical({'stage': 'complete', **summary,
                     'package_inventory_sha256': sha(package_path), 'source_inventory_sha256': sha(source_path),
                     'category_summary_sha256': sha(output / 'category-summary.json')}), flush=True)


if __name__ == '__main__':
    main()
