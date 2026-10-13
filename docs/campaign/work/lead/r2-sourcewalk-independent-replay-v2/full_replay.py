#!/usr/bin/env python3
"""Independent 76,279-row source-walk provenance replay.

The index phase reads each raw scenario stream exactly once and publishes
per-shard raw records atomically.  The replay phase commits one shard at a
time.  A terminal receipt is reusable only when every immutable binding and
the output hashes still match.
"""
from __future__ import annotations

import argparse, collections, hashlib, importlib.util, json, os, shutil, signal, sys, uuid
from pathlib import Path
from typing import Any, Iterable

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
HOLD_LEDGER = Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-SFT-admission-review-v1/mechanical-hold-ledger.jsonl')
HOLD_SHA = '6a626df3c55b6bc1907eb3b64f8e8fec50095f07db47db0d796f17e658dda751'
GLOBAL = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
GLOBAL_SHA = 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
CPT = PLAN / 'docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json'
CPT_SHA = '6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06'
LICENSE = PLAN / 'docs/campaign/work/r2-corpus-preparation-v1/raw_cpt.py'
LICENSE_SHA = 'e02d588ac77d3a8bf8710217c8d1c678e730681de998932219afe75a2709dd80'
STRICT = PLAN / 'docs/campaign/work/lead/r2-roxy8597-root-fixes-v1/audit_authoritative_stream.py'
STRICT_SHA = '7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37'
EXPECTED = {'all': 76279, 'roxygen_drafting': 72052, 'no_op': 4227,
            'upstream_mechanical_hold_noop': 7}
TERMINAL_IDS = [97666, 24779, 651, 8089, 23877]
STOP = False

def _stop(_signum, _frame):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb', buffering=4 << 20) as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path); fsync_dir(path.parent)

def atomic_lines(path: Path, values: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp'); count = 0
    with temp.open('x') as stream:
        for value in values:
            stream.write(canonical(value) + '\n'); count += 1
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path); fsync_dir(path.parent)
    return count, sha(path)

def load_module(name: str, path: Path, expected_sha: str):
    if sha(path) != expected_sha: raise RuntimeError(f'{name}_hash_mismatch')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f'{name}_import_failed')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module

def dcf_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}; key = None
    for line in text.splitlines():
        if line[:1].isspace() and key: out[key] += ' ' + line.strip()
        elif ':' in line:
            key, value = line.split(':', 1); key = key.strip(); out[key] = value.strip()
    return out

def positive_license(description: bytes, expected_package: str, parser) -> dict[str, Any]:
    try: fields = dcf_fields(description.decode('utf-8'))
    except UnicodeDecodeError: return {'ok': False, 'reason': 'description_non_utf8'}
    required = {'Package': fields.get('Package', '').strip(), 'License': fields.get('License', '').strip()}
    if not all(required.values()): return {'ok': False, 'reason': 'missing_required_description_fields', 'fields': required}
    if required['Package'] != expected_package: return {'ok': False, 'reason': 'description_package_mismatch', 'fields': required}
    if fields.get('License_restricts_use', '').strip().lower() == 'yes': return {'ok': False, 'reason': 'license_restricts_use', 'fields': required}
    if fields.get('License_is_FOSS', '').strip().lower() == 'no': return {'ok': False, 'reason': 'license_not_foss', 'fields': required}
    if not parser.allowed_license(required['License']): return {'ok': False, 'reason': 'license_not_in_reviewed_allowed_families', 'fields': required}
    return {'ok': True, 'reason': 'positive_reviewed_allowed_license', 'fields': required,
            'decision_parser_sha256': LICENSE_SHA}

def stable_read(path: Path) -> tuple[bytes, dict[str, Any]]:
    before = path.stat(); identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    data = path.read_bytes(); after = path.stat()
    stable = identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and len(data) == before.st_size
    return data, {'path': str(path), 'bytes': len(data), 'sha256': sha_bytes(data), 'stat_stable': stable,
                  'stat': {'device': before.st_dev, 'inode': before.st_ino, 'size': before.st_size, 'mtime_ns': before.st_mtime_ns}}

def source_window(packet: dict, raw: dict, family: str) -> tuple[bytes, str]:
    if family == 'roxygen_drafting':
        # The normalized upstream source contains the complete documentation
        # target. The prompt is the simulated pre-edit view. Reconstruct the
        # post-edit window from the independently reopened raw scenario to prove
        # source support; this window is never used as prompt evidence.
        lines = list(raw.get('prefix', [])) + list(raw.get('region_new', [])) + list(raw.get('suffix', []))
        return ('\n'.join(lines) + '\n').encode(), 'raw_post_edit_source_support_window'
    selection = packet.get('result', {}).get('selection_source', {})
    text = selection.get('text')
    expected = selection.get('content_sha256')
    if isinstance(text, str) and expected and sha_bytes(text.encode()) == expected:
        return text.encode(), 'packet_selection_source_exact'
    # Fallback is deliberately named and cannot pass a no-op geometry gate.
    lines = list(raw.get('prefix', [])) + list(raw.get('region_old', [])) + list(raw.get('suffix', []))
    return ('\n'.join(lines) + '\n').encode(), 'raw_list_join_fallback_unproven'

def window_occurrence(source: bytes, window: bytes) -> tuple[int, str]:
    exact = source.count(window)
    if exact: return exact, 'exact_bytes'
    # Builders explicitly normalize document EOL to LF. Permit only the exact
    # CRLF->LF transform, never whitespace or content normalization.
    if b'\r\n' in source and b'\r' not in source.replace(b'\r\n', b''):
        normalized = source.replace(b'\r\n', b'\n')
        count = normalized.count(window)
        if count: return count, 'uniform_crlf_to_lf'
    return 0, 'no_match'

def noop_geometry(packet: dict, raw: dict, source: bytes) -> dict[str, Any]:
    window, method = source_window(packet, raw, 'no_op')
    occurrences, occurrence_method = window_occurrence(source, window)
    provenance = packet.get('result', {}).get('provenance', {})
    context = packet.get('result', {}).get('context', {})
    kind = raw.get('kind')
    old = list(raw.get('region_old', [])); target = list(packet.get('result', {}).get('target_body', []))
    unchanged = (raw.get('region_new') == [] and target == ([] if old == [''] else old) and
                 packet.get('result', {}).get('operation') == 'no_op')
    supported_kind = kind in {'after_close_brace', 'blank_between'}
    cursor = context.get('cursor', {}).get('region_line_index')
    rr = context.get('replacement_range', {})
    zero_width = rr.get('start') == rr.get('end') and cursor == -1
    replace_same_line = (kind == 'after_close_brace' and len(old) == 1 and old[0].strip() == '}' and target == old and
                         rr.get('start',{}).get('line') == rr.get('end',{}).get('line') and
                         rr.get('start',{}).get('character') == 0 and rr.get('end',{}).get('character') == len(old[0]))
    geometry = zero_width if kind == 'blank_between' else replace_same_line
    blank_evidence = (kind != 'blank_between' or provenance.get('physical_blank_anchor') is True)
    ok = method == 'packet_selection_source_exact' and occurrences == 1 and unchanged and supported_kind and geometry and blank_evidence
    return {'ok': ok, 'kind': kind, 'window_method': method, 'window_sha256': sha_bytes(window),
            'window_occurrences': occurrences, 'occurrence_method': occurrence_method, 'unchanged_target': unchanged,
            'supported_kind': supported_kind, 'geometry_supported': geometry, 'zero_width_cursor_geometry': zero_width,
            'physical_blank_evidence': blank_evidence}

def input_inventory(shards: Iterable[int]) -> tuple[list[dict], dict[str, Any]]:
    if sha(HOLD_LEDGER) != HOLD_SHA: raise RuntimeError('hold_ledger_hash_mismatch')
    holds = {}
    with HOLD_LEDGER.open() as stream:
        for line in stream:
            item = json.loads(line); holds[item['row_id']] = item['reasons']
    chosen, pins = [], []
    for shard in shards:
        directory = BASE / f'shard-{shard:04d}' / 'structured-materialization-v1'
        manifest_path = directory / 'token-audit-manifest.json'; manifest = json.loads(manifest_path.read_text())
        rows_path = Path(manifest['token_rows']['path']); digest = hashlib.sha256(); total = 0; selected = 0
        with rows_path.open('rb') as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line); total += 1; item = json.loads(raw_line); row = item['row']; rid = row['id']
                include = row['family'] == 'no_op' or (row['family'] == 'roxygen_drafting' and rid not in holds)
                if include:
                    selected += 1; chosen.append({'shard': shard, 'token_line': line_number, 'row_id': rid,
                        'family': row['family'], 'package_id': row['package_id'], 'source_ref': item['source_ref'],
                        'upstream_mechanical_hold_reasons': holds.get(rid, [])})
        if digest.hexdigest() != manifest['token_rows']['sha256'] or total != manifest['token_rows']['rows']:
            raise RuntimeError(f'token_rows_pin_mismatch:{shard}')
        pins.append({'shard': shard, 'path': str(rows_path), 'sha256': digest.hexdigest(), 'rows': total,
                     'selected_rows': selected, 'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path)})
    ids = [x['row_id'] for x in chosen]
    if len(ids) != len(set(ids)): raise RuntimeError('selected_id_duplicate')
    counts = collections.Counter(x['family'] for x in chosen)
    noop_holds = sum(x['family'] == 'no_op' and bool(x['upstream_mechanical_hold_reasons']) for x in chosen)
    if len(pins) == 41 and ({'all': len(chosen), **counts, 'upstream_mechanical_hold_noop': noop_holds} != EXPECTED):
        raise RuntimeError(f'full_denominator_mismatch:{len(chosen)}:{dict(counts)}:{noop_holds}')
    return chosen, {'hold_ledger': {'path': str(HOLD_LEDGER), 'sha256': HOLD_SHA, 'rows': len(holds)},
                    'shard_pins': pins, 'rows': len(chosen), 'families': dict(counts),
                    'upstream_mechanical_hold_noop': noop_holds}

def build_index(out: Path, shards: list[int]) -> dict[str, Any]:
    chosen, inventory = input_inventory(shards); wanted: dict[Path, dict[int, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for item in chosen:
        ref = item['source_ref']; wanted[Path(ref['file'])][int(ref['line'])].append(item)
    stage = out / f'.index-{uuid.uuid4().hex}'; stage.mkdir(parents=True)
    writers = {s: (stage / f'shard-{s:04d}.jsonl').open('x') for s in shards}; source_pins = []
    try:
        for path, lines in sorted(wanted.items(), key=lambda x: str(x[0])):
            before = path.stat(); identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            digest = hashlib.sha256(); found = set()
            with path.open('rb', buffering=4 << 20) as stream:
                for number, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    if number in lines:
                        found.add(number)
                        parsed = json.loads(raw_line)
                        for selected in lines[number]:
                            ref = selected['source_ref']
                            line_ok = sha_bytes(raw_line) == ref['raw_line_sha256'] or sha_bytes(raw_line.rstrip(b'\r\n')) == ref['raw_line_sha256']
                            metadata_ok = parsed.get('family') == selected['family'] and parsed.get('package') == selected['package_id']
                            if not line_ok or not metadata_ok: raise RuntimeError(f'raw_line_or_metadata_mismatch:{selected["row_id"]}')
                            writers[selected['shard']].write(canonical({'selected': selected, 'raw': parsed,
                                'raw_line_sha256_observed': sha_bytes(raw_line.rstrip(b'\r\n'))}) + '\n')
            after = path.stat(); stable = identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            expected_hashes = {item['source_ref']['source_sha256'] for values in lines.values() for item in values}
            if not stable: raise RuntimeError(f'raw_source_stat_changed:{path}')
            if len(found) != len(lines): raise RuntimeError(f'raw_lines_missing:{path}')
            if expected_hashes != {digest.hexdigest()}: raise RuntimeError(f'raw_source_hash_mismatch:{path}')
            source_pins.append({'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size,
                                'stat_stable': True, 'selected_lines': len(lines)})
        for stream in writers.values(): stream.flush(); os.fsync(stream.fileno()); stream.close()
        index_files = []
        for shard in shards:
            path = stage / f'shard-{shard:04d}.jsonl'; index_files.append({'shard': shard, 'path': str(out/'index'/path.name),
                'rows': sum(1 for _ in path.open()), 'sha256': sha(path), 'bytes': path.stat().st_size})
        manifest = {'schema': 'sepalith.dat10.sourcewalk_raw_index.v2', 'status': 'complete',
                    'input_inventory': inventory, 'raw_sources': source_pins, 'index_files': index_files,
                    'driver_sha256': sha(Path(__file__).resolve())}
        atomic_json(stage / 'manifest.json', manifest); fsync_dir(stage)
        target = out / 'index'
        if target.exists():
            old = json.loads((target/'manifest.json').read_text())
            if old != manifest: raise RuntimeError('existing_index_binding_mismatch')
            shutil.rmtree(stage); return old
        stage.replace(target); fsync_dir(out); return manifest
    except BaseException:
        for stream in writers.values():
            if not stream.closed: stream.close()
        raise

def receipt_reusable(path: Path, binding: dict[str, Any]) -> bool:
    if not path.exists(): return False
    try: value = json.loads(path.read_text())
    except Exception: return False
    if value.get('status') != 'complete' or value.get('binding') != binding: return False
    for item in value.get('outputs', []):
        p = Path(item['path'])
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']: return False
    return True

def commit_shard(out: Path, shard: int, binding: dict[str, Any], rows: Iterable[dict], stop_after_write: bool = False) -> str:
    terminal = out / 'shards' / f'shard-{shard:04d}' / 'receipt.json'
    if receipt_reusable(terminal, binding): return 'reused'
    if terminal.parent.exists(): raise RuntimeError(f'nonreusable_terminal_shard:{shard}')
    stage = out / 'shards' / f'.shard-{shard:04d}-{uuid.uuid4().hex}'; stage.mkdir(parents=True)
    count, digest = atomic_lines(stage/'ledger.jsonl', rows)
    if stop_after_write: raise KeyboardInterrupt('synthetic_interrupt_after_write_before_commit')
    receipt = {'schema': 'sepalith.dat10.sourcewalk_provenance_shard.v2', 'status': 'complete', 'shard': shard,
               'binding': binding, 'rows': count, 'outputs': [{'path': str(out/'shards'/f'shard-{shard:04d}'/'ledger.jsonl'),
               'rows': count, 'bytes': (stage/'ledger.jsonl').stat().st_size, 'sha256': digest}]}
    atomic_json(stage/'receipt.json', receipt); fsync_dir(stage); stage.replace(terminal.parent); fsync_dir(terminal.parent.parent)
    return 'committed'

def command_build_index(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    shards = list(range(41)) if args.shards == 'all' else [int(x) for x in args.shards.split(',')]
    result = build_index(args.output, shards); print(canonical({'status': result['status'], 'rows': result['input_inventory']['rows']})); return 0

def replay_one(out: Path, shard: int) -> str:
    index_manifest_path = out/'index/manifest.json'; index_manifest = json.loads(index_manifest_path.read_text())
    if index_manifest.get('status') != 'complete': raise RuntimeError('index_not_complete')
    index_entry = next(x for x in index_manifest['index_files'] if x['shard'] == shard)
    index_path = Path(index_entry['path'])
    if sha(index_path) != index_entry['sha256'] or sum(1 for _ in index_path.open()) != index_entry['rows']:
        raise RuntimeError(f'index_shard_pin_mismatch:{shard}')
    selected = {}; raw_rows = {}
    with index_path.open() as stream:
        for line in stream:
            item=json.loads(line); rid=item['selected']['row_id']; selected[rid]=item['selected']; raw_rows[rid]=item['raw']
    directory = BASE/f'shard-{shard:04d}'/'structured-materialization-v1'
    token_manifest_path=directory/'token-audit-manifest.json'; token_manifest=json.loads(token_manifest_path.read_text())
    token_path=Path(token_manifest['token_rows']['path']); packet_manifest_path=directory/'manifest.json'; packet_manifest=json.loads(packet_manifest_path.read_text()); packet_path=Path(packet_manifest['outputs']['candidate_packets']['path'])
    tokens={}; token_hash=hashlib.sha256(); token_count=0
    with token_path.open('rb') as stream:
        for line in stream:
            token_hash.update(line); token_count+=1; item=json.loads(line); rid=item['row']['id']
            if rid in selected: tokens[rid]=item
    packets={}; packet_hash=hashlib.sha256(); packet_count=0
    with packet_path.open('rb') as stream:
        for line in stream:
            packet_hash.update(line); packet_count+=1; item=json.loads(line); rid=item.get('row_ref',{}).get('row_id')
            if rid in selected: packets[rid]=item
    if token_hash.hexdigest()!=token_manifest['token_rows']['sha256'] or token_count!=token_manifest['token_rows']['rows']: raise RuntimeError(f'token_pin_mismatch:{shard}')
    if packet_hash.hexdigest()!=packet_manifest['outputs']['candidate_packets']['sha256'] or packet_count!=packet_manifest['outputs']['candidate_packets']['rows']: raise RuntimeError(f'packet_pin_mismatch:{shard}')
    if set(selected)!=set(tokens) or set(selected)!=set(packets): raise RuntimeError(f'shard_join_incomplete:{shard}')
    if sha(GLOBAL)!=GLOBAL_SHA or sha(CPT)!=CPT_SHA: raise RuntimeError('registry_hash_mismatch')
    groups={x['group_id']:x for x in json.loads(GLOBAL.read_text())['groups']}; partitions=json.loads(CPT.read_text())['groups']
    strict=load_module('sourcewalk_strict',STRICT,STRICT_SHA); license_parser=load_module('sourcewalk_license',LICENSE,LICENSE_SHA)
    from tree_sitter import Language,Parser
    import tree_sitter_r
    parser=Parser(Language(tree_sitter_r.language())); source_cache={}; description_cache={}; results=[]
    for rid in sorted(selected):
        sel=selected[rid]; item=tokens[rid]; row=item['row']; raw=raw_rows[rid]; packet=packets[rid]; ref=item['source_ref']; validation=packet['validation']
        # The three independently loaded records must identify one source event.
        joined=(sel['source_ref']==ref and packet['row_ref']==ref and ref['row_id']==rid and
                ref['package_id']==row['package_id']==raw.get('package') and ref['family']==row['family']==raw.get('family') and
                raw.get('path')==packet['result']['context']['path'])
        source_path=Path(validation['source_path'])
        if source_path not in source_cache:
            source_bytes,source_ev=stable_read(source_path);source_cache[source_path]=(source_bytes,source_ev,not parser.parse(source_bytes).root_node.has_error)
        source_bytes,source_ev,parse_ok=source_cache[source_path]
        license_path=Path(validation['license_evidence']['path'])
        if license_path not in description_cache:
            description,description_ev=stable_read(license_path);description_cache[license_path]=(description_ev,positive_license(description,row['package_id'],license_parser))
        description_ev,license_decision=description_cache[license_path]
        window,window_method=source_window(packet,raw,row['family']); occurrences,occurrence_method=window_occurrence(source_bytes,window)
        protocol_errors=strict.validate_token_row(row,full_text=True)
        group=groups.get(ref['group_id']); partition=partitions.get(ref['group_id'])
        common_ok=(joined and source_ev['stat_stable'] and description_ev['stat_stable'] and source_ev['sha256']==validation['source_sha256'] and
                   description_ev['sha256']==validation['license_evidence']['sha256'] and license_decision['ok'] and parse_ok and
                   group is not None and group.get('split')=='train_group' and partition!='cpt_validation' and
                   window_method in {'packet_selection_source_exact','raw_post_edit_source_support_window'} and occurrences==1 and not protocol_errors)
        noop=noop_geometry(packet,raw,source_bytes) if row['family']=='no_op' else None
        if not common_ok: status='hold_independent_provenance_failure'
        elif noop is not None and not noop['ok']: status='hold_noop_geometry_failure'
        elif row['family']=='no_op': status='provenance_supported_candidate_root_review_required'
        else: status='provenance_pass_semantic_analyzer_queued'
        results.append({'row_id':rid,'shard':shard,'family':row['family'],'package_id':row['package_id'],'group_id':ref['group_id'],
            'source_line_group_join':joined,'raw_source_line':ref['line'],'source_path_sha256':source_ev['sha256'],'source_stat_stable':source_ev['stat_stable'],
            'description_sha256':description_ev['sha256'],'description_stat_stable':description_ev['stat_stable'],'license_decision':license_decision,
            'global_train':bool(group and group.get('split')=='train_group'),'protected_disjoint':partition!='cpt_validation',
            'source_parse_ok':parse_ok,'source_window_method':window_method,'source_window_occurrences':occurrences,'source_window_occurrence_method':occurrence_method,
            'strict_protocol_ok':not protocol_errors,'strict_protocol_errors':protocol_errors,'noop_geometry':noop,
            'upstream_mechanical_hold_reasons':sel['upstream_mechanical_hold_reasons'],'semantic_analyzer':'not_applicable_noop' if noop else 'queued_not_passed',
            'status':status,'source_or_target_text_written':False})
    binding={'driver_sha256':sha(Path(__file__).resolve()),'index_manifest_sha256':sha(index_manifest_path),'index_shard_sha256':index_entry['sha256'],
             'token_rows_sha256':token_hash.hexdigest(),'candidate_packets_sha256':packet_hash.hexdigest(),'global_sha256':GLOBAL_SHA,'cpt_sha256':CPT_SHA,
             'hold_ledger_sha256':HOLD_SHA,'strict_validator_sha256':STRICT_SHA,'license_parser_sha256':LICENSE_SHA}
    return commit_shard(out,shard,binding,results)

def merge(out: Path) -> dict[str, Any]:
    receipts=[]; total=0; ids=set(); families=collections.Counter(); statuses=collections.Counter(); noop_holds=0
    index=json.loads((out/'index/manifest.json').read_text())
    for entry in index['index_files']:
        shard=entry['shard']; receipt_path=out/'shards'/f'shard-{shard:04d}'/'receipt.json'
        if not receipt_path.exists(): raise RuntimeError(f'missing_shard:{shard}')
        receipt=json.loads(receipt_path.read_text()); receipts.append({'shard':shard,'receipt_sha256':sha(receipt_path)})
        for line in Path(receipt['outputs'][0]['path']).open():
            item=json.loads(line); rid=item['row_id']
            if rid in ids: raise RuntimeError(f'duplicate_output_id:{rid}')
            ids.add(rid);total+=1;families[item['family']]+=1;statuses[item['status']]+=1
            noop_holds += item['family']=='no_op' and bool(item['upstream_mechanical_hold_reasons'])
    observed={'all':total,**families,'upstream_mechanical_hold_noop':noop_holds}
    if len(index['index_files'])==41 and observed!=EXPECTED: raise RuntimeError(f'merged_denominator_mismatch:{observed}')
    result={'schema':'sepalith.dat10.sourcewalk_independent_replay.v2','status':'complete_review_only_no_admission','denominators':observed,
            'status_counts':dict(statuses),'shard_receipts':receipts,'semantic_roxygen_is_queue_not_pass':True,'training_admission':False}
    atomic_json(out/'manifest.json',result);return result

def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-index'); p.add_argument('--output', type=Path, required=True); p.add_argument('--shards', default='all')
    p = sub.add_parser('replay'); p.add_argument('--output', type=Path, required=True); p.add_argument('--shards', default='all')
    p = sub.add_parser('merge'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(os, 'sched_setaffinity'): os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    if args.command == 'build-index': return command_build_index(args)
    if args.command == 'replay':
        shards=range(41) if args.shards=='all' else [int(x) for x in args.shards.split(',')]
        for shard in shards:
            if STOP: break
            print(canonical({'shard':shard,'result':replay_one(args.output,shard)}),flush=True)
        return 130 if STOP else 0
    if args.command == 'merge': print(canonical(merge(args.output)));return 0
    raise AssertionError

if __name__ == '__main__': raise SystemExit(main())
