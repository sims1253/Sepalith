"""Candidate-only source selection, token lengths and rendered collisions.

Outputs deliberately wrap token rows: they are not an admitted SFT registry.
Source validation, licensing, split provenance and scientific admission remain
separate gates. The CLI reads only an explicitly hashed train/dev conversion
file and never discovers or opens held-out examples.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sepalith.campaign_protocol import (
    PromptContext, TOKENIZER_JSON_SHA256, TOKENIZER_CONFIG_SHA256,
    build_training_row, utf16_length,
)
from sepalith.campaign_selection import (
    CAMPAIGN_SELECTION_POLICY_ID, DEFAULT_SELECTION_UTF16_UNITS,
    SelectionScope, logical_lines, select_source_window,
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def length_profiles(row):
    prompt = row['target_start']  # Includes the one manual BOS.
    response = len(row['input_ids']) - prompt  # Includes terminal and EOS.
    return {
        'prompt_with_bos': prompt,
        'prompt_without_bos': row['prompt_token_count'],
        'target_body': row['target_body_token_count'],
        'response_with_terminal_eos': response,
        'sequence': len(row['input_ids']),
        'new_contract_under_legacy_480_170_thresholds': (
            row['prompt_token_count'] <= 480 and row['target_token_count'] <= 170),
        'sft_2048': len(row['input_ids']) <= 2048,
        'sft_4096': len(row['input_ids']) <= 4096,
        'rl_prompt2048_response192': prompt <= 2048 and response <= 192,
    }


def selected_context(result, *, max_source_utf16=DEFAULT_SELECTION_UTF16_UNITS):
    context = PromptContext.from_mapping(result['context'])
    source = result['selection_source']
    if source['availability'] not in ('full_snapshot', 'source_builder_window'):
        raise ValueError('unsupported_selection_source_availability')
    if 'text' in source and 'document_text' in source and source['text'] != source['document_text']:
        raise ValueError('selection_source_text_aliases_disagree')
    if 'document_text' in source:
        text = source['document_text']
    elif 'text' in source:
        text = source['text']
    else:
        # read_bytes avoids Python universal-newline normalization.
        text = Path(source['path']).read_bytes().decode('utf-8')
    if not isinstance(text, str):
        raise ValueError('selection_source_text_required')
    source_sha = text_digest(text)
    if source_sha != source['content_sha256'] or source_sha != context.replacement_range.content_sha256:
        raise ValueError('selection_source_document_hash_mismatch')
    without_crlf = text.replace('\r\n', '')
    if '\r' in without_crlf or ('\r\n' in text and '\n' in without_crlf):
        raise ValueError('mixed_or_lone_cr_source')
    expected_eol = 'crlf' if '\r\n' in text else 'lf'
    if context.document_eol != expected_eol:
        raise ValueError('selection_source_eol_mismatch')
    lines = logical_lines(text)
    start, end = source['region_start_line'], source['region_end_line']
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(lines):
        raise ValueError('selection_region_outside_document')
    physical_region = tuple(lines[start:end + 1])
    # An empty physical editor line has a zero-width range and therefore
    # uses the frozen protocol's canonical [] region. The line still exists
    # in the source document; removing it would merge an insertion with the
    # following suffix or put an EOF range outside the buffer.
    canonical_empty_line = not context.region_old and physical_region == ('',)
    if physical_region != context.region_old and not canonical_empty_line:
        raise ValueError('selection_region_text_mismatch')
    geometry = context.replacement_range
    if (geometry.start.line != start or geometry.end.line != end
            or geometry.start.character != 0 or geometry.end.character != utf16_length(lines[end])):
        raise ValueError('selection_requires_complete_logical_line_range')
    scope_data = source.get('scope')
    scope = SelectionScope(**scope_data) if scope_data is not None else None
    selected = select_source_window(
        lines=lines, region_start_line=start, region_end_line=end, scope=scope,
        max_utf16_units=max_source_utf16, document_sha256=source_sha,
    )
    context = replace(context, prefix=selected.prefix, suffix_lines=selected.suffix)
    # Revalidate the modified context through the frozen public contract.
    context = PromptContext.from_mapping(context.to_dict())
    return context, {
        **selected.to_dict(), 'availability': source['availability'],
        'support_revalidation_required': bool(selected.omissions),
        'policy_id_combined': CAMPAIGN_SELECTION_POLICY_ID,
    }


def prepare_candidate(packet, tokenizer, *, max_source_utf16=DEFAULT_SELECTION_UTF16_UNITS):
    ref, result = packet['row_ref'], packet['result']
    if ref['split'] not in ('train_group', 'dev_group'):
        raise ValueError('only_explicit_train_dev_references_are_allowed')
    for key in ('row_id', 'group_id', 'file', 'source_sha256', 'raw_line_sha256'):
        if not isinstance(ref.get(key), str) or not ref[key]:
            raise ValueError('missing_source_reference:' + key)
    if result['status'] != 'converted':
        raise ValueError('adapter_did_not_convert:' + str(result.get('reason')))
    family = packet.get('family', ref.get('family'))
    parent = result.get('provenance', {}).get('parent_identity', {})
    package = ref.get('package_id') or parent.get('package')
    if not package and parent.get('repo'):
        package = 'repo:' + parent['repo']
    if not package:
        # A parent group is explicit, but is not a substitute for a package.
        raise ValueError('package_identity_required')
    context, selection = selected_context(result, max_source_utf16=max_source_utf16)
    row = build_training_row(
        context, operation=result['operation'], region_new=result['target_body'],
        tokenizer=tokenizer, row_id=ref['row_id'], family=family, package_id=package,
        split='train' if ref['split'] == 'train_group' else 'dev',
    )
    return {
        'status': 'tokenizer_candidate_only', 'row': row,
        'context': context.to_dict(), 'source_ref': ref,
        'source_provenance': result.get('provenance', {}),
        'selection': selection, 'lengths': length_profiles(row),
        'prompt_sha256': text_digest(row['prompt_text']),
        'target_sha256': text_digest(row['target_text']),
        'admitted_for_training': False,
    }


def rendered_collisions(candidates):
    by_prompt = defaultdict(list)
    for candidate in candidates:
        by_prompt[candidate['prompt_sha256']].append(candidate)
    rows = []
    for prompt_sha, group in sorted(by_prompt.items()):
        if len(group) < 2:
            continue
        targets = {r['target_sha256'] for r in group}
        splits = {r['row']['split'] for r in group}
        rows.append({
            'prompt_sha256': prompt_sha, 'different_targets': len(targets) > 1,
            'cross_split': len(splits) > 1,
            'members': [{
                'id': r['row']['id'], 'split': r['row']['split'],
                'group_id': r['source_ref']['group_id'], 'target_sha256': r['target_sha256'],
            } for r in group],
            'disposition': 'requires_source_and_target_adjudication' if len(targets) > 1 else 'exact_prompt_target_duplicate',
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-jsonl', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-source-utf16', type=int, default=DEFAULT_SELECTION_UTF16_UNITS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output_must_be_fresh')
    if digest(args.input_jsonl) != args.input_sha256:
        raise ValueError('conversion_input_hash_mismatch')
    for name, expected in [('tokenizer.json', TOKENIZER_JSON_SHA256), ('tokenizer_config.json', TOKENIZER_CONFIG_SHA256)]:
        if digest(args.tokenizer / name) != expected:
            raise ValueError('tokenizer_hash_mismatch:' + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    candidates, excluded, seen = [], [], set()
    with args.input_jsonl.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            packet = json.loads(line)
            row_id = packet.get('row_ref', {}).get('row_id')
            if not row_id or row_id in seen:
                raise ValueError('missing_or_duplicate_conversion_identity')
            seen.add(row_id)
            try:
                candidates.append(prepare_candidate(packet, tokenizer, max_source_utf16=args.max_source_utf16))
            except (ValueError, KeyError, TypeError) as error:
                excluded.append({'line': number, 'id': row_id, 'family': packet.get('family'), 'reason': str(error)})
    collisions = rendered_collisions(candidates)
    families = {}
    for family in sorted({r['row']['family'] for r in candidates}):
        rows = [r for r in candidates if r['row']['family'] == family]
        families[family] = {'rows': len(rows), 'profiles': dict(Counter(
            key for row in rows for key, value in row['lengths'].items() if type(value) is bool and value)),
            'sequence_min': min(r['lengths']['sequence'] for r in rows),
            'sequence_max': max(r['lengths']['sequence'] for r in rows)}
    args.output.mkdir(parents=True)
    target = args.output / 'candidate-token-rows.jsonl'
    with target.open('w', encoding='utf-8') as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {
        'task': 'DAT-04', 'status': 'candidate_token_audit_not_registry_admission',
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'input': str(args.input_jsonl), 'input_sha256': args.input_sha256,
        'candidate_file': str(target), 'candidate_file_sha256': digest(target),
        'converted_inputs': len(seen), 'tokenizer_candidates': len(candidates),
        'excluded': excluded, 'families': families, 'rendered_collisions': collisions,
        'limitations': [
            'Complete labels retained; no target truncation or silent required-source truncation.',
            '480/170 uses the new tokenizer and contract; it is not a reproduction of historical admission.',
            'This audit does not verify licensing, source validators, or final-set overlap and grants no training admission.',
            'Different targets require source-aware adjudication; valid generative alternatives are not automatically contradictory.',
        ],
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'candidate_rows': len(candidates), 'excluded': len(excluded), 'collision_groups': len(collisions), 'report': str(args.output / 'report.json')}))


if __name__ == '__main__':
    main()
