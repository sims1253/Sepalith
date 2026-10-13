#!/usr/bin/env python3
"""Freeze the bounded corrected-plus-short DAT-10 candidate subset.

The larger candidate registry in this directory is retained as an optional
extended artifact.  This script derives the primary packet from its
source-class ordered prefix, then independently checks row contracts, exact
source IDs, DAT-02/CPT binding, prompt consistency, and the metadata-only
finite sampler/loader interfaces.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
OUT = PLAN / 'work/r2-task-mixture-v1'
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
FULL_ROWS = OUT / 'candidate-token-rows.jsonl'
FULL_PROV = OUT / 'candidate-provenance.jsonl'
FULL_EXCLUSIONS = OUT / 'candidate-exclusions.jsonl'
CORRECTED_INPUT = PLAN / 'work/lead/finish-corrected-train-v1/train-token-rows.jsonl'
SHORT_INPUT = PLAN / 'work/r2-short-task-preparation-v1/candidate-packets.jsonl'
DAT05_PROVENANCE = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/provenance.jsonl')
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
DAT02 = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
PARTITION = PLAN / 'work/r2-corpus-preparation-v1/cpt-train-group-partition.json'
ROW_OUT = OUT / 'verified-corrected-short-token-rows.jsonl'
PROV_OUT = OUT / 'verified-corrected-short-provenance.jsonl'
EXCL_OUT = OUT / 'verified-corrected-short-exclusions.jsonl'
REPORT_OUT = OUT / 'verified-corrected-short-report.json'
SCHEDULE_OUT = OUT / 'verified-corrected-short-draw-manifest.json'
SOURCE_OUT = OUT / 'verified-corrected-short-source-manifest.json'
LOADER_LOG = OUT / 'verified-subset-loader-probe.log'
CHECK_LOG = OUT / 'verified-subset-checks.log'
SAMPLER = EXEC / 'experiments/training/campaign_sampling.py'
SFT_DATA = EXEC / 'experiments/training/campaign_sft_data.py'

EXPECTED = {
    'corrected_rows': 11526,
    'short_rows': 411,
    'accepted_corrected': 8115,
    'accepted_short': 411,
    'excluded_corrected': 3411,
    'reserved_groups': 556,
}

def digest_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open('rb', buffering=4 * 1024 * 1024) as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n

def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def canonical_hash(value: object) -> str:
    return digest_bytes(canonical(value))

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def jsonl_ids(path: Path, *, nested_row: bool = False) -> set[str]:
    result: set[str] = set()
    with path.open('rb', buffering=4 * 1024 * 1024) as f:
        for line_no, raw in enumerate(f, 1):
            item = json.loads(raw)
            if nested_row:
                item = item.get('row', {})
            ident = item.get('id')
            if not isinstance(ident, str) or not ident:
                raise AssertionError(f'missing source id {path}:{line_no}')
            if ident in result:
                raise AssertionError(f'duplicate source id {path}:{line_no}:{ident}')
            result.add(ident)
    return result

def main() -> None:
    for path in (FULL_ROWS, FULL_PROV, FULL_EXCLUSIONS, CORRECTED_INPUT, SHORT_INPUT, DAT05_PROVENANCE, TOKENIZER, DAT02, PARTITION, SAMPLER, SFT_DATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    corrected_ids = jsonl_ids(CORRECTED_INPUT)
    short_ids = jsonl_ids(SHORT_INPUT, nested_row=True)
    assert len(corrected_ids) == EXPECTED['corrected_rows'], len(corrected_ids)
    assert len(short_ids) == EXPECTED['short_rows'], len(short_ids)

    dat02_doc = json.loads(DAT02.read_text(encoding='utf-8'))
    groups = {item['group_id']: item for item in dat02_doc['groups']}
    partition_doc = json.loads(PARTITION.read_text(encoding='utf-8'))
    pmap = partition_doc['groups']
    reserved = {gid for gid, value in pmap.items() if value == 'cpt_validation'}
    assert len(reserved) == EXPECTED['reserved_groups'], len(reserved)

    sys.path.insert(0, str(EXEC / 'experiments/training'))
    sys.path.insert(0, str(EXEC / 'packages/sepalith/src'))
    protocol = load_module('dat10_subset_protocol', EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py')
    sampling = load_module('dat10_subset_sampling', SAMPLER)

    # The full builder writes source classes as one deterministic prefix:
    # corrected, short, structured, completion.  Stop at the first deferred
    # class and require exact expected counts rather than filtering arbitrary
    # later rows.
    row_hash = hashlib.sha256()
    prov_hash = hashlib.sha256()
    excl_hash = hashlib.sha256()
    row_count = 0
    prov_count = 0
    source_counts = Counter()
    family_counts = Counter()
    operation_counts = Counter()
    prompt_hashes: dict[str, str] = {}
    target_labels = 0
    total_tokens = 0
    prompt_tokens = 0
    max_total = 0
    max_target = 0
    min_target = None
    packages: set[str] = set()
    subset_metadata = []
    seen_ids: set[str] = set()
    in_primary = True
    with ROW_OUT.open('wb') as rows_out, PROV_OUT.open('wb') as prov_out:
        with FULL_ROWS.open('rb', buffering=4 * 1024 * 1024) as rows_in, FULL_PROV.open('rb', buffering=4 * 1024 * 1024) as prov_in:
            for line_no, (row_raw, prov_raw) in enumerate(zip(rows_in, prov_in), 1):
                prov = json.loads(prov_raw)
                source_class = prov.get('source_class')
                if source_class not in {'corrected', 'short'}:
                    in_primary = False
                if not in_primary:
                    break
                row = json.loads(row_raw)
                ident = row.get('id')
                assert source_class in {'corrected', 'short'}
                assert ident == prov.get('id') == prov.get('input_row_id') or source_class == 'short', (line_no, ident, prov.get('id'), prov.get('input_row_id'))
                if source_class == 'corrected':
                    assert ident in corrected_ids, ident
                    assert prov.get('input_row_id') == ident, (line_no, ident, prov.get('input_row_id'))
                else:
                    assert ident in short_ids, ident
                assert ident not in seen_ids, (line_no, ident)
                seen_ids.add(ident)
                assert row.get('split') == 'train'
                assert row.get('renderer_id') == 'zeta2-prm03-v1'
                protocol.validate_training_row(row)
                ids = row['input_ids']
                assert len(ids) <= 4096, (ident, len(ids))
                target_count = int(row['target_token_count']) + 1
                assert target_count <= 192, (ident, target_count)
                assert len(ids) == int(row['target_start']) + target_count
                assert prov.get('split_binding') == 'bound_cpt_train', (ident, prov.get('split_binding'))
                gid = prov.get('group_id')
                assert isinstance(gid, str) and gid in groups and gid in pmap, (ident, gid)
                assert pmap[gid] == 'cpt_train', (ident, gid, pmap.get(gid))
                assert gid not in reserved, (ident, gid)
                assert groups[gid].get('split') in {'train', 'train_group'}, (ident, gid, groups[gid].get('split'))
                assert row.get('package_id') == prov.get('package_id'), (ident, row.get('package_id'), prov.get('package_id'))
                assert prov.get('target_operation') == row.get('target_operation'), ident
                assert prov.get('target_label_tokens_including_protocol_EOS') == target_count, ident
                assert prov.get('total_tokens') == len(ids), ident
                ph = hashlib.sha256(row['prompt_text'].encode('utf-8')).hexdigest()
                th = hashlib.sha256(row['target_text'].encode('utf-8')).hexdigest()
                assert prov.get('prompt_sha256') == ph, ident
                assert prov.get('target_sha256') == th, ident
                assert prov.get('row_canonical_sha256') == canonical_hash(row), ident
                prior = prompt_hashes.get(ph)
                assert prior is None, (ident, ph, prior)
                prompt_hashes[ph] = th
                rows_out.write(row_raw); row_hash.update(row_raw)
                prov_out.write(prov_raw); prov_hash.update(prov_raw)
                row_count += 1; prov_count += 1
                source_counts[source_class] += 1
                family_counts[row['family']] += 1
                operation_counts[row['target_operation']] += 1
                target_labels += target_count
                total_tokens += len(ids)
                prompt_tokens += int(row['target_start'])
                max_total = max(max_total, len(ids))
                max_target = max(max_target, target_count)
                min_target = target_count if min_target is None else min(min_target, target_count)
                packages.add(row['package_id'])
                subset_metadata.append({
                    'row_id': ident,
                    'family': row['family'],
                    'source_id': gid,
                    'package_id': row['package_id'],
                    'split': 'train',
                    'semantic_noop': row['target_operation'] == 'no_op',
                    'operation': row['target_operation'],
                    'prompt_tokens': int(row['target_start']),
                    'target_tokens': target_count,
                    'total_tokens': len(ids),
                    'length_bucket': 'short' if len(ids) <= 2048 else 'long',
                    'naturally_long': len(ids) > 2048,
                    'source_kind': 'ordinary',
                    'provenance': f'verified-{source_class}:{gid}',
                })
        rows_out.flush()
    assert source_counts == Counter({'corrected': EXPECTED['accepted_corrected'], 'short': EXPECTED['accepted_short']}), source_counts
    assert row_count == prov_count == sum(source_counts.values()) == 8526, (row_count, prov_count, source_counts)

    # Exclusions are filtered only by the same two source classes.  This keeps
    # all explicit binding/cap exclusions for the primary subset auditable.
    excl_counts = Counter()
    excl_total = 0
    with EXCL_OUT.open('wb') as out, FULL_EXCLUSIONS.open('rb', buffering=4 * 1024 * 1024) as src:
        for raw in src:
            item = json.loads(raw)
            if item.get('source_class') not in {'corrected', 'short'}:
                continue
            out.write(raw); excl_hash.update(raw)
            excl_total += 1
            excl_counts[(item['source_class'], item['reason'])] += 1
        out.flush()
    assert excl_total == EXPECTED['excluded_corrected'], excl_total
    assert dict(excl_counts) == {
        ('corrected', 'cpt_validation_reserved'): 432,
        ('corrected', 'target_over_192_including_EOS'): 541,
        ('corrected', 'group_not_in_cpt_partition'): 2438,
    }, excl_counts

    # Build a schedule from metadata only.  No prompt, target, or token-ID
    # fields enter campaign_sampling.
    split_id = dat02_doc.get('split_id')
    schedule = sampling.build_draw_manifest(
        subset_metadata,
        max_steps=500,
        effective_batch=16,
        split_id=split_id,
        seed=3407,
        token_rows_sha256=row_hash.hexdigest(),
        requested_draws=8000,
        noop_fraction=0.10,
        family_ceiling=0.25,
        small_pack_cap=3,
        ordinary_replay_cap=8,
        naturally_long_fraction=0.20,
        short_max_tokens=2048,
        long_max_tokens=4096,
    )
    sampling.validate_draw_manifest(schedule)
    SCHEDULE_OUT.write_text(json.dumps(schedule, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')

    # Re-read the output bytes and check the production loader's structural
    # interface.  This does not load a model or invoke CUDA.
    rows_digest, rows_bytes = digest_file(ROW_OUT)
    schedule_digest, schedule_bytes = digest_file(SCHEDULE_OUT)
    assert rows_digest == row_hash.hexdigest()
    loader = load_module('dat10_subset_sft_data', SFT_DATA)
    inspected_rows, draw_indices, inspected_schedule, exposure = loader.inspect_training_data(
        {'path': str(ROW_OUT), 'sha256': rows_digest},
        {'path': str(SCHEDULE_OUT), 'sha256': schedule_digest},
        renderer_id='zeta2-prm03-v1', max_sequence_tokens=4096,
        max_steps=500, effective_batch=16,
    )
    assert len(inspected_rows) == row_count
    assert len(draw_indices) == 8000
    assert inspected_schedule['token_rows_sha256'] == rows_digest
    assert len(exposure) == 500

    # Source manifest separates the immutable input pins from derived files.
    source_manifest = {
        'schema': 'dat10.r2.task_mixture_source_manifest.v1',
        'policy': 'exact row/group IDs; corrected and short only; no authored package-name joins; all CPT validation groups excluded',
        'inputs': {
            str(CORRECTED_INPUT): {'sha256': digest_file(CORRECTED_INPUT)[0], 'bytes': digest_file(CORRECTED_INPUT)[1], 'rows': len(corrected_ids)},
            str(SHORT_INPUT): {'sha256': digest_file(SHORT_INPUT)[0], 'bytes': digest_file(SHORT_INPUT)[1], 'rows': len(short_ids)},
            str(DAT02): {'sha256': digest_file(DAT02)[0], 'bytes': digest_file(DAT02)[1], 'groups': len(groups), 'split_id': split_id},
            str(PARTITION): {'sha256': digest_file(PARTITION)[0], 'bytes': digest_file(PARTITION)[1], 'cpt_train_groups': sum(v == 'cpt_train' for v in pmap.values()), 'cpt_validation_groups': len(reserved)},
            str(DAT05_PROVENANCE): {'sha256': digest_file(DAT05_PROVENANCE)[0], 'bytes': digest_file(DAT05_PROVENANCE)[1], 'role': 'exact corrected-row provenance join upstream'},
            str(TOKENIZER): {'sha256': digest_file(TOKENIZER)[0], 'bytes': digest_file(TOKENIZER)[1], 'role': 'pinned tokenizer identity'},
            str(FULL_ROWS): {'sha256': digest_file(FULL_ROWS)[0], 'bytes': digest_file(FULL_ROWS)[1], 'role': 'upstream extended registry from which primary prefix is derived'},
            str(FULL_PROV): {'sha256': digest_file(FULL_PROV)[0], 'bytes': digest_file(FULL_PROV)[1], 'role': 'upstream extended provenance ledger'},
            str(EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py'): {'sha256': digest_file(EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py')[0], 'bytes': digest_file(EXEC / 'packages/sepalith/src/sepalith/campaign_protocol.py')[1]},
            str(SAMPLER): {'sha256': digest_file(SAMPLER)[0], 'bytes': digest_file(SAMPLER)[1]},
            str(SFT_DATA): {'sha256': digest_file(SFT_DATA)[0], 'bytes': digest_file(SFT_DATA)[1]},
            str(OUT / 'freeze_verified_subset.py'): {'sha256': digest_file(OUT / 'freeze_verified_subset.py')[0], 'bytes': digest_file(OUT / 'freeze_verified_subset.py')[1], 'role': 'reproducible primary extraction/checks'},
            str(OUT / 'propose_noop25_schedule.py'): {'sha256': digest_file(OUT / 'propose_noop25_schedule.py')[0], 'bytes': digest_file(OUT / 'propose_noop25_schedule.py')[1], 'role': 'reproducible sensitivity schedule builder'},
        },
        'derived': {
            str(ROW_OUT): {'sha256': rows_digest, 'bytes': rows_bytes, 'rows': row_count},
            str(PROV_OUT): {'sha256': prov_hash.hexdigest(), 'bytes': PROV_OUT.stat().st_size, 'rows': prov_count},
            str(EXCL_OUT): {'sha256': excl_hash.hexdigest(), 'bytes': EXCL_OUT.stat().st_size, 'rows': excl_total},
            str(SCHEDULE_OUT): {'sha256': schedule_digest, 'bytes': schedule_bytes, 'draws': len(schedule['row_ids'])},
        },
    }
    sensitivity_schedule = OUT / 'verified-corrected-short-draw-manifest-noop25.json'
    if sensitivity_schedule.is_file():
        sensitivity_hash, sensitivity_bytes = digest_file(sensitivity_schedule)
        source_manifest['derived'][str(sensitivity_schedule)] = {'sha256': sensitivity_hash, 'bytes': sensitivity_bytes, 'draws': 16000, 'role': 'separate no-op sensitivity proposal'}
    SOURCE_OUT.write_text(json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')

    report = {
        'schema': 'dat10.r2.task_mixture_verified_subset.v1',
        'status': 'candidate_prepared_not_training_admitted',
        'scope': 'smallest bounded corrected-plus-short subset; structured/completion legacy materializations deferred',
        'source_policy': 'corrected rows joined by exact corrected id to DAT-05 provenance in the upstream frozen builder; short rows retain their audited source_provenance group_id; exact DAT-02 group_id and explicit CPT partition checks repeated here',
        'counts': {
            'corrected_input_rows': len(corrected_ids),
            'short_input_rows': len(short_ids),
            'accepted_rows': row_count,
            'accepted_by_source_class': dict(sorted(source_counts.items())),
            'accepted_by_family': dict(sorted(family_counts.items())),
            'accepted_by_operation': dict(sorted(operation_counts.items())),
            'excluded_primary_rows': excl_total,
            'excluded_by_reason': {f'{a}:{b}': n for (a, b), n in sorted(excl_counts.items())},
            'reserved_cpt_validation_groups': len(reserved),
            'accepted_groups': len({m['source_id'] for m in subset_metadata}),
            'accepted_packages': len(packages),
            'prompt_duplicate_count': 0,
            'prompt_conflicting_target_count': 0,
            'target_label_tokens_including_protocol_eos': target_labels,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'min_target_tokens_including_eos': min_target,
            'max_target_tokens_including_eos': max_target,
            'max_total_tokens': max_total,
            'no_op_rows': operation_counts.get('no_op', 0),
            'no_op_row_fraction': operation_counts.get('no_op', 0) / row_count,
        },
        'schedule': {
            'path': str(SCHEDULE_OUT),
            'sha256': schedule_digest,
            'status': schedule['status'],
            'max_steps': schedule['max_steps'],
            'effective_batch': schedule['effective_batch'],
            'draw_count': len(schedule['row_ids']),
            'seed': 3407,
            'noop_fraction_policy': 0.10,
            'family_ceiling_policy': 0.25,
            'naturally_long_fraction_ceiling': 0.20,
            'achieved_mixture': schedule.get('achieved_mixture'),
        },
        'checks': {
            'row_protocol_validation': 'PASS',
            'exact_source_id_membership': 'PASS',
            'dat02_cpt_partition_binding': 'PASS',
            'reserved_validation_group_exclusion': 'PASS',
            'prompt_contradiction_audit': 'PASS (0 duplicate/conflicting prompt hashes)',
            'target_cap_including_eos': 'PASS (all <=192)',
            'full_context_cap': 'PASS (all <=4096)',
            'sampler_manifest_validation': 'PASS',
            'production_loader_inspect_training_data': 'PASS',
        },
        'loader_probe': {'path': str(LOADER_LOG), 'rows': len(inspected_rows), 'draws': len(draw_indices), 'batches': len(exposure)},
        'limitations': [
            'Preparation artifact only; no data admission, training launch, model load, or quality claim.',
            'Structured and completion legacy candidates remain in the optional extended artifact and are not part of this primary packet.',
            'This bounded packet records exact row/group/source pins and derived bytes; it does not re-scan normalized source documents.',
            'No-op rows are sampled at a finite 10% row-slot reservation; their shorter target-token mass is reported and is not silently reweighted.',
        ],
    }
    # Replace the intentionally non-serialized placeholder with the exact
    # computed no-op label token total, avoiding another full JSONL pass.
    noop_target_labels = 0
    with ROW_OUT.open() as f:
        for line in f:
            r = json.loads(line)
            if r['target_operation'] == 'no_op':
                noop_target_labels += int(r['target_token_count']) + 1
    report['counts']['no_op_target_label_tokens'] = noop_target_labels
    report['counts']['no_op_target_token_fraction'] = noop_target_labels / target_labels
    report['counts']['no_op_target_token_fraction'] = noop_target_labels / target_labels
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')

    LOADER_LOG.write_text(json.dumps({'status': 'PASS', 'rows': len(inspected_rows), 'draws': len(draw_indices), 'batches': len(exposure), 'max_batch_size': max(x['draws'] for x in exposure), 'target_loss_tokens': sum(x['target_loss_tokens'] for x in exposure), 'prompt_loss_tokens': sum(x['prompt_loss_tokens'] for x in exposure)}, sort_keys=True) + '\n')
    CHECK_LOG.write_text(json.dumps({'status': 'PASS', 'rows': row_count, 'provenance_rows': prov_count, 'excluded_rows': excl_total, 'source_counts': dict(source_counts), 'family_counts': dict(sorted(family_counts.items())), 'operation_counts': dict(operation_counts), 'target_label_tokens': target_labels, 'total_tokens': total_tokens, 'reserved_groups': len(reserved), 'accepted_groups': len({m['source_id'] for m in subset_metadata}), 'prompt_duplicates': 0, 'prompt_conflicts': 0, 'schedule_status': schedule['status'], 'schedule_draws': len(schedule['row_ids']), 'loader_rows': len(inspected_rows), 'loader_batches': len(exposure)}, sort_keys=True) + '\n')
    print(json.dumps({'status':'PASS','rows':row_count,'source_counts':dict(source_counts),'families':dict(sorted(family_counts.items())),'operations':dict(operation_counts),'target_label_tokens':target_labels,'total_tokens':total_tokens,'no_op_target_label_tokens':noop_target_labels,'schedule':schedule['status'],'draws':len(schedule['row_ids']),'rows_sha256':rows_digest,'provenance_sha256':prov_hash.hexdigest(),'exclusions_sha256':excl_hash.hexdigest(),'schedule_sha256':schedule_digest}, sort_keys=True))

if __name__ == '__main__':
    main()
