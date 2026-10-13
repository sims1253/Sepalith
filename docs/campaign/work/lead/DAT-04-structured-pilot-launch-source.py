"""Bounded train-only structured candidates with fresh source checks.

The output is a conversion candidate file, not an admitted training registry.
Only selected train records are decoded from source JSONL files. Other bytes
are streamed opaquely for the declared file hash. Final content is not parsed.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from campaign_admission_structured import (
    _row_ref_from_audit, _snapshot_ref, _normalized_path, convert_structured,
)
from campaign_token_audit import selected_context

AUDIT_SHA = '9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd'
SPLIT_SHA = 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
CANONICAL = Path('/home/m0hawk/Documents/Sepalith')
FAMILIES = ('rename_propagation', 'pipe_rewrite', 'na_rm_propagation',
            'format_propagation', 'no_op', 'roxygen_drafting')
BOUNDARY_KEYS = ('prefix', 'region_old', 'region_new', 'suffix', 'cursor_idx', 'family', 'package', 'path')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(name):
    path = CANONICAL / 'experiments/synthetic-data' / (name + '.py')
    spec = importlib.util.spec_from_file_location('campaign_source_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {'path': str(path), 'sha256': file_hash(path)}


def boundary(row):
    return {key: row.get(key, []) if key == 'suffix' else row.get(key) for key in BOUNDARY_KEYS}


def apply_result(result):
    source = result['selection_source']
    text = source.get('text', source.get('document_text'))
    if text is None:
        text = Path(source['path']).read_bytes().decode('utf-8')
    context = result['context']
    eol = '\r\n' if context['document_eol'] == 'crlf' else '\n'
    lines = text.split(eol)
    geometry = context['replacement_range']
    assert sha(text.encode()) == geometry['content_sha256'] == source['content_sha256']

    def offset(position):
        number, units = position['line'], position['character']
        assert 0 <= number < len(lines)
        encoded = lines[number].encode('utf-16-le')
        assert 0 <= units * 2 <= len(encoded)
        return sum(len(line) + len(eol) for line in lines[:number]) + len(encoded[:units * 2].decode('utf-16-le'))

    first, last = offset(geometry['start']), offset(geometry['end'])
    assert text[first:last].replace(eol, '\n') == '\n'.join(context['region_old'])
    applied = text if result['operation'] == 'no_op' else text[:first] + eol.join(result['target_body']) + text[last:]
    assert sha(applied.encode()) == result['provenance']['post_edit_snapshot_sha256']
    return applied


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-family', type=int, default=128)
    parser.add_argument('--package-family-cap', type=int, default=8)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args()
    assert 1 <= args.per_family <= 5000 and 1 <= args.package_family_cap <= 100
    assert 1 <= args.max_seconds <= 1800 and not args.output.exists()
    started = time.monotonic()
    assert file_hash(args.split) == SPLIT_SHA
    groups = {g['group_id']: g for g in json.loads(args.split.read_text())['groups']}
    selected, seen_counts = [], Counter()
    family_count, group_count = Counter(), Counter()
    digest = hashlib.sha256()
    # DAT03 is ordered by the hash-based row ID. The first bounded prefix per
    # family/group is deterministic and does not use labels or model reward.
    with args.audit.open('rb') as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            family = record['family']
            if record['split'] != 'train_group' or family not in FAMILIES:
                continue
            seen_counts[family] += 1
            group = groups[record['group_id']]
            assert group['split'] == 'train_group'
            if family_count[family] >= args.per_family or group_count[(family, record['group_id'])] >= args.package_family_cap:
                continue
            selected.append(record)
            family_count[family] += 1
            group_count[(family, record['group_id'])] += 1
    assert digest.hexdigest() == AUDIT_SHA
    args.output.mkdir(parents=True)
    (args.output / 'selected-audit-rows.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in selected))
    by_file = defaultdict(list)
    for record in selected:
        by_file[record['file']].append(record)
    raw_rows, source_files = {}, []
    for filename, records in sorted(by_file.items()):
        wanted = {r['line']: r for r in records}
        path = Path(filename)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                if number in wanted:
                    record = wanted[number]
                    assert sha(line) == record['raw_line_sha256']
                    assert sha(f"{filename}\0{number}\0{sha(line)}".encode())[:24] == record['row_id']
                    raw_rows[record['row_id']] = json.loads(line)
        after = path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        assert all(r['source_sha256'] == digest.hexdigest() for r in records)
        source_files.append({'path': filename, 'bytes': before.st_size, 'sha256': digest.hexdigest(), 'selected_records': len(records)})
        print(json.dumps({'stage': 'source_verified', 'path': filename, 'selected': len(records)}), flush=True)
    read_seconds = time.monotonic() - started
    scenarios, scenarios_ref = load_source('scenarios')
    roxygen, roxygen_ref = load_source('roxygen_drafting')
    suffix, suffix_ref = load_source('suffix_scenarios')
    constructors = [scenarios_ref, roxygen_ref, suffix_ref]
    snapshot_cache, mined_cache, licenses = {}, {}, {}
    accepted, exclusions, outcome_families = 0, [], Counter()
    packets_path = args.output / 'candidate-packets.jsonl'
    with packets_path.open('w', encoding='utf-8') as output:
        for index, record in enumerate(selected):
            if time.monotonic() - started > args.max_seconds:
                exclusions.extend({'id': r['row_id'], 'family': r['family'], 'reason': 'timebox_not_attempted'} for r in selected[index:])
                break
            family = record['family']
            try:
                raw = raw_rows[record['row_id']]
                assert raw['family'] == family
                if family == 'no_op' and raw.get('kind') not in ('after_close_brace', 'blank_between'):
                    raise ValueError('no_op_kind_requires_separate_support_review')
                norm = _normalized_path(raw)
                if norm is None:
                    raise ValueError('unique_normalized_parent_source_required')
                source_bytes = norm.read_bytes()
                if len(source_bytes) > 1024 * 1024:
                    raise ValueError('source_file_over_bounded_read_limit')
                if scenarios.parser.parse(source_bytes).root_node.has_error:
                    raise ValueError('normalized_parent_R_parse_error')
                key = (family, str(norm), sha(source_bytes))
                if family in ('no_op', 'roxygen_drafting'):
                    if key not in mined_cache:
                        mined, stats = [], Counter()
                        if family == 'roxygen_drafting':
                            roxygen.extract_file(raw['package'], norm.name, source_bytes, mined, stats)
                        else:
                            suffix.extract_no_op(raw['package'], norm.name, source_bytes, mined, Counter(), stats)
                        mined_cache[key] = [boundary(r) for r in mined]
                    if boundary(raw) not in mined_cache[key]:
                        raise ValueError('fresh_constructor_does_not_reproduce_source_boundary_and_target')
                    if family == 'roxygen_drafting' and any(tag in '\n'.join(raw['region_new']) for tag in ('@author', '@references', '@source', '@examples', '@seealso')):
                        raise ValueError('documentation_external_facts_require_separate_support_review')
                else:
                    scenarios.validate_example(raw)
                description = norm.parents[1] / 'DESCRIPTION'
                if str(description) not in licenses:
                    content = description.read_bytes()
                    fields = [line for line in content.decode('utf-8').splitlines() if line.startswith(('License:', 'License_restricts_use:', 'License_is_FOSS:'))]
                    if not any(line.startswith('License:') for line in fields):
                        raise ValueError('license_field_missing')
                    if any(line in ('License_restricts_use: yes', 'License_is_FOSS: no') for line in fields):
                        raise ValueError('source_license_restriction_requires_review')
                    licenses[str(description)] = {'path': str(description), 'sha256': sha(content), 'fields': fields}
                ref = _row_ref_from_audit(record)
                snapshot_key = (family, raw['package'], raw.get('version'), raw['path'], record['source'])
                if snapshot_key not in snapshot_cache:
                    enriched = _snapshot_ref(raw, dict(ref))
                    if enriched is None:
                        raise ValueError('source_snapshot_unresolved')
                    snapshot_cache[snapshot_key] = {k: v for k, v in enriched.items() if k not in ref}
                ref.update(deepcopy(snapshot_cache[snapshot_key]))
                result = convert_structured(raw, ref)
                if result['status'] != 'converted':
                    raise ValueError('adapter:' + result['reason'])
                applied = apply_result(result)
                context, selection = selected_context(result)
                # For local propagation the immutable target region and the
                # verified preceding event contain the complete edit rule.
                # Drafting requires the full captured function suffix.
                if family == 'roxygen_drafting' and list(context.suffix_lines[:len(raw['suffix'])]) != raw['suffix']:
                    raise ValueError('required_function_suffix_omitted_by_source_selection')
                if family not in ('no_op', 'roxygen_drafting') and len(context.history) != 1:
                    raise ValueError('propagation_requires_one_verified_history_event')
                if family in ('rename_propagation', 'pipe_rewrite', 'na_rm_propagation', 'format_propagation') and scenarios.parser.parse(applied.encode()).root_node.has_error:
                    raise ValueError('post_edit_full_buffer_R_parse_error')
                row_ref = _row_ref_from_audit(record)
                row_ref['package_id'] = raw['package']
                packet = {'row_ref': row_ref, 'family': family, 'result': result,
                          'validation': {'fresh_source_constructor': True, 'full_buffer_application': True,
                                         'normalized_parent_R_parse': True, 'source_selection_support_check': True,
                                         'source_path': str(norm), 'source_sha256': sha(source_bytes),
                                         'license_evidence': licenses[str(description)],
                                         'parent_group_split': 'train_group', 'admission': False}}
                output.write(json.dumps(packet, ensure_ascii=False, separators=(',', ':')) + '\n')
                accepted += 1
                outcome_families[family] += 1
            except (AssertionError, ValueError, KeyError, OSError, UnicodeError) as error:
                exclusions.append({'id': record['row_id'], 'family': family, 'reason': str(error) or type(error).__name__})
            if (index + 1) % 64 == 0:
                output.flush()
                print(json.dumps({'stage': 'conversion', 'attempted': index + 1, 'converted': accepted, 'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)
    report = {'task': 'DAT-04', 'status': 'source_checked_candidates_not_training_registry',
              'observed_at': datetime.now(timezone.utc).isoformat(), 'selected': len(selected), 'converted': accepted,
              'selected_families': dict(family_count), 'converted_families': dict(outcome_families),
              'population_metadata_counts': dict(seen_counts), 'excluded': exclusions,
              'exclusion_counts': dict(Counter(e['reason'] for e in exclusions)),
              'elapsed_seconds': time.monotonic() - started, 'source_read_seconds': read_seconds,
              'source_files': source_files, 'constructor_sources': constructors,
              'candidate_packets': str(packets_path), 'candidate_packets_sha256': file_hash(packets_path),
              'audit_sha256': AUDIT_SHA, 'split_sha256': SPLIT_SHA,
              'adapter_sha256': file_hash(Path(__file__).with_name('campaign_admission_structured.py')),
              'limitations': ['No training admission; tokenizer, rendered-collision and lead scientific review remain.',
                              'Hash-based bounded family selection is not a population quality estimate.',
                              'Source-derived simulations are not observed editor trajectories.',
                              'License evidence is preserved; no new license interpretation is issued.'],
              'final_content_parsed': False, 'cuda_started': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(args.output / 'report.json'), 'converted': accepted, 'excluded': len(exclusions)}))


if __name__ == '__main__':
    main()
