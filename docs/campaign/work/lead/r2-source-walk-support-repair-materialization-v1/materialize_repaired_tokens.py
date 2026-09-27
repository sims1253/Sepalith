#!/usr/bin/env python3
"""Tokenize repaired TRAIN prompts without writing source text.

This pass starts from the frozen support-review packets, reconstructs each
complete normalized before-state in memory, and runs the reviewed structured
adapter.  It emits token IDs, hashes, lengths, and source-selection metadata;
prompt/target/source strings are removed before serialization.  The five
source-context profiles are all measured so a full-file replay cannot be
mistaken for the old 6,000-UTF-16-unit cropped token rows.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PREP = PLAN / 'docs/campaign/work/lead/r2-source-walk-support-repair-v1/prepare_repair.py'
ROOT_REVIEW_RECEIPT = PLAN / 'docs/campaign/receipts/DAT-10-roxy10017-root-context-review.json'
ROOT_CANDIDATE_IDS = PLAN / 'docs/campaign/work/lead/r2-roxy10017-root-context-review-v1/candidate-ids.json'
SUPPORT_LEDGER = PLAN / 'docs/campaign/work/lead/r2-source-walk-support-review-v1/support-ledger.jsonl'
SHARD_ROOT = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
PACKET_RELATIVE = 'structured-materialization-v1/candidate-packets.jsonl'
OUT_DEFAULT = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017')
TRAINING_DIR = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training')
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
TOKENIZER_SHA = '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
SOURCE_SHA = '0d70ccc40a716a8a27cb508e39a16c9a5119a9bc9d0b5b8ee8362aa4b67f6193'
BUDGETS: tuple[int | str, ...] = (6000, 8192, 16384, 32768, 'full_file')

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
import campaign_structured_batch as structured_batch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def load_preparation_module():
    spec = importlib.util.spec_from_file_location('dat10_repair_preparation', PREP)
    if spec is None or spec.loader is None:
        raise RuntimeError('repair_preparation_module_missing')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load_preparation_module()
review = prep.review
adapter = prep.adapter
token_audit = prep.token_audit


class DirectPinnedTokenizer:
    """Expose the pinned JSON backend through the protocol encoder seam."""

    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend

    def encode(self, text: str, *, add_special_tokens: bool = False,
               split_special_tokens: bool = True) -> list[int]:
        if add_special_tokens is not False or split_special_tokens is not True:
            raise ValueError('materializer_tokenizer_policy_violation')
        return self.backend.encode(text, add_special_tokens=False).ids


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb', buffering=4 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def strip_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    # Token IDs and counts are sufficient for the root trainer.  Source and
    # target strings never cross the output boundary.
    forbidden = {'prompt_text', 'target_text', 'target_body_text'}
    return {key: value for key, value in row.items() if key not in forbidden}


def length_profile(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'prompt_with_bos': row['target_start'],
        'prompt_without_bos': row['prompt_token_count'],
        'target_body_tokens': row['target_body_token_count'],
        'target_terminal_tokens': row['target_terminal_token_count'],
        'response_with_terminal_eos': row['target_token_count'] + 1,
        'target_token_count': row['target_token_count'],
        'sequence': len(row['input_ids']),
        'train_target_le_1024': row['target_body_token_count'] <= 1024,
        'sft_4096': len(row['input_ids']) <= 4096,
        'sft_8192': len(row['input_ids']) <= 8192,
        'sft_16384': len(row['input_ids']) <= 16384,
        'sft_32768': len(row['input_ids']) <= 32768,
    }


def reconstruct(ledger: dict[str, Any], packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str, int]:
    raw, package = review.packet_raw(packet)
    source_path = Path(ledger['source_path'])
    source_bytes = source_path.read_bytes()
    after_hash = sha_bytes(source_bytes)
    if after_hash != ledger['source_sha256']:
        raise ValueError('normalized_after_source_hash_mismatch')
    source_text = source_bytes.decode('utf-8')
    source_eol = 'crlf' if '\r\n' in source_text else 'lf'
    source_lines = source_text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    positions = review.locate_inverse_target(source_lines, raw['region_new'], raw['suffix'])
    if len(positions) != 1:
        raise ValueError('normalized_after_target_suffix_not_unique')
    position = positions[0]
    start = position['target_start_line']
    removed = len(raw['region_new']) + position['blank_after_target']
    before_lines = source_lines[:start] + [''] + source_lines[start + removed:]
    before_text = ('\r\n' if source_eol == 'crlf' else '\n').join(before_lines)
    before_bytes = before_text.encode('utf-8', 'surrogatepass')
    before_hash = sha_bytes(before_bytes)
    expected_before = ledger.get('context', {}).get('full_snapshot_sha256')
    if before_hash != expected_before:
        raise ValueError('derived_before_hash_mismatch_support_review')
    ref = copy.deepcopy(packet['row_ref'])
    ref.update({
        'package_id': package,
        'source_snapshot_text': before_text,
        'source_snapshot_sha256': before_hash,
        'source_snapshot_provenance_path': 'derived:normalized-after/' + str(source_path),
        'cursor_encoding': 'scenario_char_offset',
        'workspace_revision_before': before_hash,
        'target_line': start,
        'parent_identity': {
            'package': package, 'path': raw['path'],
            'normalized_after_source_path': str(source_path),
            'normalized_after_source_sha256': after_hash,
            'pre_edit_derivation': 'remove_exact_mined_roxygen_target_retain_physical_blank_anchor',
        },
    })
    converted = adapter.convert_structured(raw, ref)
    if converted.get('status') != 'converted':
        raise ValueError('full_snapshot_conversion_failed:' + str(converted.get('reason')))
    structured_batch.apply_result(converted)
    return raw, converted, package, before_hash, start


def materialize_one(ledger: dict[str, Any], packet: dict[str, Any], tokenizer: Tokenizer) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, converted, package, before_hash, target_line = reconstruct(ledger, packet)
    context_profiles: list[dict[str, Any]] = []
    selected_rows: dict[str, dict[str, Any]] = {}
    for budget in BUDGETS:
        max_units = 1_000_000_000 if budget == 'full_file' else int(budget)
        context, selection = token_audit.selected_context(converted, max_source_utf16=max_units)
        row = prep.review.token_audit.build_training_row(
            context,
            operation=converted['operation'],
            region_new=converted['target_body'],
            tokenizer=tokenizer,
            row_id=ledger['row_id'],
            family=ledger['family'],
            package_id=package,
            split='train',
        )
        lengths = length_profile(row)
        context_projection = review.structural_projection(context.to_dict())
        profile = {
            'budget': budget,
            'used_utf16_units': selection.get('used_utf16_units'),
            'required_utf16_units': selection.get('required_utf16_units'),
            'omission_count': len(selection.get('omissions', [])),
            'overflow': selection.get('overflow'),
            'required_overflow': selection.get('required_overflow'),
            'captured_context_anchors_match': True,
            'support_revalidation_required': bool(selection.get('support_revalidation_required')),
            'context_projection_sha256': sha_bytes(json.dumps(context_projection, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()),
            'document_eol': context.document_eol,
            'prompt_sha256': sha_bytes(row['prompt_text'].encode('utf-8')),
            'target_sha256': sha_bytes(row['target_text'].encode('utf-8')),
            'lengths': lengths,
        }
        context_profiles.append(profile)
        if budget == 'full_file':
            selected_rows['full_file'] = strip_text_fields(row)
    target_body = raw['region_new']
    base = {
        'schema': 'DAT-10-source-walk-support-repair-token-profile-v1',
        'row_id': ledger['row_id'], 'family': ledger['family'], 'group_id': ledger['group_id'],
        'package_id': package, 'source_file': ledger['source_file'], 'source_line': ledger['source_line'],
        'source_path': ledger['source_path'], 'normalized_after_source_sha256': ledger['source_sha256'],
        'derived_before_sha256': before_hash, 'derived_before_target_line': target_line,
        'raw_line_sha256': ledger['raw_line_sha256'], 'target_semantics': converted['operation'],
        'target_body_line_count': len(target_body), 'target_body_sha256': sha_bytes('\n'.join(target_body).encode('utf-8')),
        'original_review_reasons': ledger['reasons'], 'target_rewritten': False,
        'admission': 'review_only_unadmitted', 'source_payload_written': False,
        'context_profiles': context_profiles,
        'full_file_token_row_available': True,
    }
    token_record = {
        **base,
        'schema': 'DAT-10-source-walk-support-repair-token-row-v1',
        'source_context': context_profiles[-1],
        'token_row': selected_rows['full_file'],
        'tokenizer_json_sha256': selected_rows['full_file']['tokenizer_json_sha256'],
        'token_ids_written': True,
        'source_text_written': False,
        'target_text_written': False,
    }
    # Keep the two output ledgers independently useful while ensuring all
    # profiles and full-file token IDs refer to the same reconstructed row.
    return base, token_record


def load_repair_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    root_receipt = json.loads(ROOT_REVIEW_RECEIPT.read_text(encoding='utf-8'))
    candidate_ids = set(json.loads(ROOT_CANDIDATE_IDS.read_text(encoding='utf-8')))
    if root_receipt.get('candidate_count') != 10017 or len(candidate_ids) != 10017:
        raise ValueError('root_context_candidate_scope_mismatch')
    unresolved = set(root_receipt.get('unresolved_ids', []))
    if candidate_ids.intersection(unresolved):
        raise ValueError('root_unresolved_id_reached_materialization')
    repairs: dict[str, dict[str, Any]] = {}
    with SUPPORT_LEDGER.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('row_id') in candidate_ids:
                if row.get('status') not in ('supported_source_replay_pending_root_admission', 'repair_required'):
                    raise ValueError('candidate_status_not_source_reviewed:' + str(row.get('row_id')))
                repairs[row['row_id']] = row
    if set(repairs) != candidate_ids:
        raise ValueError('root_context_candidate_support_join_incomplete')
    packets: dict[str, dict[str, Any]] = {}
    packet_pins = []
    for shard_no in range(5):
        packet_path = SHARD_ROOT / f'shard-{shard_no:04d}' / PACKET_RELATIVE
        packet_pins.append({'path': str(packet_path), 'sha256': sha_file(packet_path)})
        with packet_path.open(encoding='utf-8') as stream:
            for line in stream:
                packet = json.loads(line)
                row_id = str(packet['row_ref']['row_id'])
                if row_id in candidate_ids:
                    if row_id in packets:
                        raise ValueError('duplicate_candidate_packet:' + row_id)
                    packets[row_id] = packet
    if set(packets) != candidate_ids:
        raise ValueError('root_context_candidate_packet_join_incomplete')
    pins = {
        'root_context_receipt': {'path': str(ROOT_REVIEW_RECEIPT), 'sha256': sha_file(ROOT_REVIEW_RECEIPT)},
        'root_candidate_ids': {'path': str(ROOT_CANDIDATE_IDS), 'sha256': sha_file(ROOT_CANDIDATE_IDS), 'rows': len(candidate_ids)},
        'support_ledger': {'path': str(SUPPORT_LEDGER), 'sha256': sha_file(SUPPORT_LEDGER), 'rows': len(repairs)},
        'candidate_packets': packet_pins,
    }
    return repairs, packets, pins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT_DEFAULT)
    parser.add_argument('--max-rows', type=int)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError('--max-rows must be positive')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    try:
        os.nice(10)
    except OSError:
        pass
    os.environ.update(TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='2', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    if sha_file(TOKENIZER) != TOKENIZER_SHA:
        raise ValueError('tokenizer_identity_mismatch')
    support_receipt = PLAN / 'docs/campaign/receipts/DAT-10-source-walk-support-review.json'
    repair_receipt = PLAN / 'docs/campaign/receipts/DAT-10-source-walk-support-repair-preparation.json'
    if not support_receipt.is_file() or not repair_receipt.is_file():
        raise ValueError('frozen_repair_receipt_missing')
    repairs, packets, input_pins = load_repair_inputs()
    row_ids = sorted(repairs)
    if args.max_rows is not None:
        row_ids = row_ids[:args.max_rows]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'status.json', {
        'schema': 'DAT-10-source-walk-support-repair-token-materialization-status-v1',
        'status': 'running', 'scope_rows': len(row_ids), 'full_repair_scope_rows': len(repairs),
        'budgets': list(BUDGETS), 'tokenizer_json_sha256': TOKENIZER_SHA,
        'source_scenario_sha256': SOURCE_SHA, 'source_text_written': False,
        'target_text_written': False, 'cuda': False, 'training': False,
        'root_context_candidate_scope': '10017 root-verified supported/repaired IDs; unresolved 1856e8135aa52a51d1421b2d held',
        'updated_at': datetime.now(timezone.utc).isoformat(),
    })
    tokenizer_backend = Tokenizer.from_file(str(TOKENIZER))
    tokenizer_backend.encode_special_tokens = True
    tokenizer = DirectPinnedTokenizer(tokenizer_backend)
    profiles: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    for count, row_id in enumerate(row_ids, 1):
        try:
            profile, token_row = materialize_one(repairs[row_id], packets[row_id], tokenizer)
            profiles.append(profile)
            token_rows.append(token_row)
        except (AssertionError, KeyError, OSError, UnicodeError, ValueError, adapter.ProtocolError) as error:
            errors.append({'row_id': row_id, 'error': type(error).__name__ + ':' + str(error)})
        if count == 1 or count % 100 == 0 or count == len(row_ids):
            print(json.dumps({'stage': 'token_materialization', 'rows_done': count, 'rows_total': len(row_ids),
                              'profiles_pass': len(profiles), 'errors': len(errors),
                              'elapsed_seconds': round(time.monotonic() - started, 1)}, sort_keys=True), flush=True)
    profiles.sort(key=lambda row: row['row_id'])
    token_rows.sort(key=lambda row: row['row_id'])
    write_jsonl(output / 'repaired-token-profiles.jsonl', profiles)
    write_jsonl(output / 'repaired-full-file-token-rows.jsonl', token_rows)
    write_jsonl(output / 'materialization-errors.jsonl', errors)
    aggregate = {
        'schema': 'DAT-10-source-walk-support-repair-token-materialization-summary-v1',
        'status': 'complete' if not errors and len(profiles) == len(row_ids) else 'complete_with_errors',
        'scope_rows_requested': len(row_ids), 'profiles_rows': len(profiles), 'token_rows': len(token_rows),
        'errors': len(errors), 'full_repair_scope_rows': len(repairs),
        'budgets': list(BUDGETS), 'tokenizer_json_sha256': TOKENIZER_SHA,
        'source_scenario_sha256': SOURCE_SHA, 'source_text_written': False,
        'target_text_written': False, 'admitted_rows': 0, 'trained_rows': 0,
        'old_cropped_sequence_gt_4096': 0,
        'actual_full_file_sequence_gt_4096': sum(not p['context_profiles'][-1]['lengths']['sft_4096'] for p in profiles),
        'actual_full_file_target_body_gt_1024': sum(not p['context_profiles'][-1]['lengths']['train_target_le_1024'] for p in profiles),
        'profile_sequence_gt_4096': {
            str(budget): sum(not p['context_profiles'][i]['lengths']['sft_4096'] for p in profiles)
            for i, budget in enumerate(BUDGETS)
        },
        'elapsed_seconds': round(time.monotonic() - started, 3),
    }
    write_json(output / 'materialization-summary.json', aggregate)
    write_json(output / 'status.json', {**aggregate, 'status': aggregate['status'], 'updated_at': datetime.now(timezone.utc).isoformat()})
    receipt = {
        'schema': 'DAT-10-source-walk-support-repair-token-materialization-v1',
        'status': aggregate['status'], 'decision': 'review_only_unadmitted_actual_token_lengths',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'constraints': {'cpu_threads_max': 2, 'cuda': False, 'training': False, 'dev_payloads': False, 'final_payloads': False,
                        'source_text_written': False, 'target_text_written': False},
        'inputs': {
            'repair_preparation_receipt': str(repair_receipt), 'repair_preparation_receipt_sha256': sha_file(repair_receipt),
            'support_review_receipt': str(support_receipt), 'support_review_receipt_sha256': sha_file(support_receipt),
            'tokenizer': str(TOKENIZER), 'tokenizer_sha256': TOKENIZER_SHA, 'scenario_source_sha256': SOURCE_SHA,
            'root_context_scope': input_pins,
        },
        'method': {
            'inverse': 'remove exact mined region_new followed by zero-or-more physical blank lines and insert one blank anchor',
            'selection_profiles': list(BUDGETS),
            'token_row_policy': 'full_file profile token IDs plus profile lengths; no prompt/target/source strings serialized',
            'target_policy': 'complete region_new is passed unchanged to adapter; no target rewrite or truncation',
            'geometry': 'full adapter conversion and structured byte/UTF-16 application are rerun per row',
        },
        'summary': aggregate,
        'artifacts': [],
    }
    for path in (output / 'repaired-token-profiles.jsonl', output / 'repaired-full-file-token-rows.jsonl',
                 output / 'materialization-errors.jsonl', output / 'materialization-summary.json', Path(__file__)):
        artifact = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}
        if path.name == 'repaired-token-profiles.jsonl': artifact['rows'] = len(profiles)
        elif path.name == 'repaired-full-file-token-rows.jsonl': artifact['rows'] = len(token_rows)
        elif path.name == 'materialization-errors.jsonl': artifact['rows'] = len(errors)
        receipt['artifacts'].append(artifact)
    receipt_path = PLAN / 'docs/campaign/receipts/DAT-10-source-walk-support-repair-token-materialization.json'
    write_json(receipt_path, receipt)
    print(json.dumps({'stage': 'complete', 'receipt': str(receipt_path), **aggregate}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
