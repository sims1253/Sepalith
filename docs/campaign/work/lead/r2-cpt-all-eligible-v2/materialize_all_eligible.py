#!/usr/bin/env python3
"""Resumable, uncapped materialization of the remaining global TRAIN groups.

Each seeded group is committed as an immutable directory. Interruption before
rename leaves only an ignored staging directory; interruption after rename is
reconciled from the receipt. No payload, package, group, or wall-time cap is
used. Only regular .R files under the package R tree are the admitted source
category. Unsupported source objects are named in a repair queue.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, fcntl, hashlib, importlib.util, json, os
from pathlib import Path
import resource, signal, stat, subprocess, sys, time, uuid

HERE = Path(__file__).resolve().parent
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE = PLAN / 'docs/campaign/work/r2-corpus-preparation-v1'
GLOBAL = PLAN / 'docs/campaign/work/r2-cpt-global-shard-v1'
ORDER = PLAN / 'docs/campaign/work/r2-cpt-long-stage-review-v1/next-global-package-order.json'
SPLIT = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
TOKENIZER = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
NORMALIZED = Path('/mnt/h/sepalith/normalized')
DEFAULT_OUTPUT = Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1')
SOURCE_MANIFEST = HERE / 'source-manifest.json'
SOURCE_MIGRATION = HERE / 'source-migration.json'
START_INDEX = 775
EXPECTED_GROUPS = 8092
PINS = {
    ORDER: 'c7255cbbb664b7a12322b7bb9acf40c1b23c73dc9e1869531443205dbc9335b8',
    SPLIT: 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09',
    BASE / 'cpt-train-group-partition.json': '6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06',
    BASE / 'known-nontrain-parent-hashes.json': '7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0',
    BASE / 'broader-shard-v1-2k/documents.jsonl': '98b9e4a3df7aa2bda79365a0521ac538628406be0cc9af3a0da1eb3ba5ab5d11',
    BASE / 'profile-shard-v1/documents.jsonl': '674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68',
    GLOBAL / 'shard/documents.jsonl': '8a8cf09059afa42fcc6915d8b0a7645558115cc440d521daeedfa6d2a9305804',
    BASE / 'raw_cpt_broader.py': '84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab',
    TOKENIZER: '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
}
STOP = False
WORKER_SEEN = '.runtime/seen-sha256.txt'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def fields(text: str) -> dict[str, str]:
    result, key = {}, None
    for line in text.splitlines():
        if line[:1].isspace() and key: result[key] += ' ' + line.strip()
        elif ':' in line:
            key, value = line.split(':', 1); result[key] = value.strip()
    return result


def fingerprints(raw: bytes) -> dict[str, str]:
    return {'sha256': hashlib.sha256(raw).hexdigest(),
            'sha1': hashlib.sha1(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}


def inventory_package(entry: dict, registry: dict, parts: dict) -> tuple[list[dict], list[dict], collections.Counter]:
    name, group = entry['name'], entry['group_id']; categories = collections.Counter(); repair = []
    if entry.get('split') != 'train_group' or registry.get(name.lower()) != (group, 'train_group'):
        raise ValueError(f'{name}: frozen TRAIN registry mismatch')
    if parts.get(group) != 'cpt_train':
        raise ValueError(f'{name}: heldout CPT validation group reached payload inventory')
    root = NORMALIZED / name
    try:
        versions = [item for item in os.scandir(root) if item.is_dir(follow_symlinks=False)]
    except OSError as error:
        return [], [{'category':'package_metadata','reason':'metadata_read_failure','detail':type(error).__name__}], categories
    if len(versions) != 1:
        return [], [{'category':'package_layout','reason':'version_count_not_one','count':len(versions)}], categories
    version = versions[0].name; source = root / version / name; description = source / 'DESCRIPTION'
    if source.is_symlink() or description.is_symlink():
        return [], [{'category':'package_layout','reason':'source_or_description_symlink'}], categories
    try:
        before = description.stat()
        if not stat.S_ISREG(before.st_mode):
            return [], [{'category':'package_metadata','reason':'description_not_regular'}], categories
        raw_description = description.read_bytes(); after = description.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('DESCRIPTION changed during read')
        dcf = fields(raw_description.decode('utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        return [], [{'category':'package_metadata','reason':'description_read_or_decode_failure','detail':type(error).__name__}], categories
    license_value = dcf.get('License', '')
    if dcf.get('Package') != name:
        return [], [{'category':'package_metadata','reason':'DESCRIPTION_package_mismatch'}], categories
    if (not license_value or dcf.get('License_restricts_use','').lower() == 'yes'
            or dcf.get('License_is_FOSS','').lower() == 'no'):
        return [], [{'category':'license','reason':'license_requires_review','license':license_value,
                     'disposition':'exclude_by_frozen_provenance'}], categories
    rdir = source / 'R'
    if not rdir.is_dir() or rdir.is_symlink():
        return [], [{'category':'package_layout','reason':'no_regular_R_directory',
                     'disposition':'exclude_no_supported_payload'}], categories
    rows, stack = [], [rdir]
    while stack:
        current = stack.pop()
        try: entries = sorted(os.scandir(current), key=lambda item:item.name)
        except OSError as error:
            repair.append({'category':'R_tree','path':str(current),'reason':'directory_read_failure','detail':type(error).__name__}); continue
        for item in entries:
            path = Path(item.path)
            if item.is_symlink():
                categories['symlink'] += 1
                if path.suffix.lower() == '.r': repair.append({'category':'R_source','path':str(path),'reason':'symlink_R_source_unsupported'})
            elif item.is_dir(follow_symlinks=False): categories['directory'] += 1; stack.append(path)
            elif not item.is_file(follow_symlinks=False):
                categories['nonregular'] += 1
                if path.suffix.lower() == '.r': repair.append({'category':'R_source','path':str(path),'reason':'nonregular_R_source_unsupported'})
            elif path.suffix.lower() != '.r': categories['regular_non_R'] += 1
            else:
                categories['regular_R'] += 1; state = item.stat(follow_symlinks=False)
                rows.append({'package':name,'version':version,'group_id':group,'split':'train_group','cpt_partition':'cpt_train',
                             'path':str(path),'bytes':state.st_size,'mtime_ns':state.st_mtime_ns,'inode':state.st_ino,
                             'device':state.st_dev,'description_path':str(description),'description_sha256':hashlib.sha256(raw_description).hexdigest(),
                             'license':license_value,'source_category':'regular_R_under_package_R','raw_code_hashed':False})
    rows.sort(key=lambda row: hashlib.sha256(('DAT10-all-eligible-file-v1\0'+row['path']).encode()).digest())
    return rows, repair, categories


def read_regular(row: dict) -> tuple[bytes | None, dict | None]:
    path = Path(row['path']); before = path.stat()
    expected = (row['bytes'],row['mtime_ns'],row['inode'],row['device'])
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev) != expected:
        return None, {'category':'R_source','path':str(path),'reason':'source_stat_changed_or_not_regular'}
    if before.st_size == 0:
        return None, {'category':'R_source','path':str(path),'reason':'empty_R_file_degenerate',
                      'disposition':'exclude_no_payload'}
    try:
        with path.open('rb') as stream: raw = stream.read()
    except OSError as error:
        return None, {'category':'R_source','path':str(path),'reason':'source_read_failure','detail':type(error).__name__}
    after = path.stat()
    if len(raw) != before.st_size or (after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev) != expected:
        return None, {'category':'R_source','path':str(path),'reason':'source_changed_during_read'}
    return raw, None


def base_seen() -> tuple[set[str], set[str]]:
    exact, protected = set(), set(json.loads((BASE/'known-nontrain-parent-hashes.json').read_text()))
    for path in (BASE/'broader-shard-v1-2k/documents.jsonl', GLOBAL/'shard/documents.jsonl'):
        with path.open() as stream:
            for row in map(json.loads, stream): exact.add(row['sha256'])
    with (BASE/'profile-shard-v1/documents.jsonl').open() as stream:
        for row in map(json.loads, stream):
            if row['cpt_partition'] == 'cpt_validation': exact.add(row['sha256'])
    return exact, protected


def committed(output: Path, seen: set[str]) -> tuple[dict[int, dict], collections.Counter]:
    receipts, totals = {}, collections.Counter()
    groups = output/'groups'
    if not groups.exists(): return receipts, totals
    for folder in sorted(path for path in groups.iterdir() if path.is_dir()):
        receipt_path = folder/'receipt.json'
        if not receipt_path.is_file(): raise ValueError(f'committed group lacks receipt: {folder}')
        receipt = json.loads(receipt_path.read_text()); index = receipt['seeded_index']
        if index in receipts: raise ValueError(f'duplicate committed seeded index: {index}')
        for name, record in receipt['artifacts'].items():
            path=folder/name
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError(f'committed group artifact changed: {path}')
        with (folder/'documents.jsonl').open() as stream:
            for row in map(json.loads, stream):
                if row['sha256'] in seen: raise ValueError(f'duplicate document in committed ledger: {row["sha256"]}')
                seen.add(row['sha256'])
        receipts[index]=receipt; totals.update(receipt['counts'])
    return receipts, totals


def process_group(output: Path, index: int, entry: dict, tokenizer, raw_cpt, registry: dict, parts: dict,
                  seen: set[str], protected: set[str]) -> dict:
    name=f'{index:06d}-{entry["group_id"]}'; final=output/'groups'/name
    if final.exists(): raise ValueError(f'group destination already exists: {final}')
    stage=output/'.staging'/(name+'.'+uuid.uuid4().hex); stage.mkdir(parents=True)
    paths={key:stage/key for key in ('cpt_train.jsonl','documents.jsonl','source-inventory.jsonl','exclusions.jsonl','repair-queue.jsonl')}
    streams={key:path.open('x') for key,path in paths.items()}; counts=collections.Counter(); started=time.monotonic()
    local_seen=set(); rows, repairs, categories=inventory_package(entry,registry,parts)
    try:
        for repair in repairs:
            value={'package':entry['name'],'group_id':entry['group_id'],**repair}
            disposition=value.pop('disposition',None)
            if disposition and disposition.startswith('exclude_'):
                streams['exclusions.jsonl'].write(canonical({**value,'disposition':disposition})+'\n');counts['excluded_documents']+=1;counts['excluded_'+value['reason']]+=1
            else:
                streams['repair-queue.jsonl'].write(canonical(value)+'\n');counts['repair_items']+=1
        for row in rows:
            streams['source-inventory.jsonl'].write(canonical(row)+'\n'); counts['regular_R_inventory']+=1; counts['source_bytes_inventory']+=row['bytes']
            # Apply the frozen provenance/license gate before opening payload.
            if not raw_cpt.allowed_license(row['license']):
                streams['exclusions.jsonl'].write(canonical({'package':entry['name'],'group_id':entry['group_id'],'path':row['path'],'reason':'license_not_in_frozen_recognized_families','license':row['license']})+'\n');counts['excluded_documents']+=1;counts['excluded_license']+=1;continue
            raw, failure=read_regular(row)
            if failure:
                value={'package':entry['name'],'group_id':entry['group_id'],**failure};disposition=value.pop('disposition',None)
                if disposition and disposition.startswith('exclude_'):
                    streams['exclusions.jsonl'].write(canonical({**value,'disposition':disposition})+'\n');counts['excluded_documents']+=1;counts['excluded_'+value['reason']]+=1
                else:
                    streams['repair-queue.jsonl'].write(canonical(value)+'\n');counts['repair_items']+=1
                continue
            hashes=fingerprints(raw); reason=None
            if protected.intersection(hashes.values()): reason='known_nontrain_parent_hash_match'
            elif hashes['sha256'] in seen: reason='exact_duplicate_of_prior_admitted_or_heldout_document'
            elif hashes['sha256'] in local_seen: reason='exact_duplicate_within_group'
            if reason:
                streams['exclusions.jsonl'].write(canonical({'package':entry['name'],'group_id':entry['group_id'],'path':row['path'],'reason':reason,'sha256':hashes['sha256']})+'\n');counts['excluded_documents']+=1;counts['excluded_'+reason]+=1;continue
            try: text=raw.decode('utf-8')
            except UnicodeDecodeError:
                streams['repair-queue.jsonl'].write(canonical({'package':entry['name'],'group_id':entry['group_id'],'path':row['path'],'reason':'non_UTF8_R_source_requires_repair','bytes':len(raw),'sha256':hashes['sha256']})+'\n');counts['repair_items']+=1;continue
            if '\0' in text:
                streams['repair-queue.jsonl'].write(canonical({'package':entry['name'],'group_id':entry['group_id'],'path':row['path'],'reason':'NUL_R_source_requires_repair','bytes':len(raw),'sha256':hashes['sha256']})+'\n');counts['repair_items']+=1;continue
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            if not ids or tokenizer.decode(ids,skip_special_tokens=False) != text:
                streams['repair-queue.jsonl'].write(canonical({'package':entry['name'],'group_id':entry['group_id'],'path':row['path'],'reason':'tokenizer_empty_or_roundtrip_failure','bytes':len(raw),'sha256':hashes['sha256']})+'\n');counts['repair_items']+=1;continue
            chunk_count=0
            for chunk_index,chunk in enumerate(raw_cpt.chunks(ids,2048)):
                record={'schema':1,'row_id':hashes['sha256']+':'+str(chunk_index),'document_id':hashes['sha256'],'package':row['package'],
                        'group_id':row['group_id'],'cpt_partition':'cpt_train','source_path':row['path'],'source_sha256':hashes['sha256'],
                        'chunk_index':chunk_index,**chunk}
                streams['cpt_train.jsonl'].write(canonical(record)+'\n');counts['rows']+=1;counts['input_tokens']+=len(chunk['input_ids']);counts['loss_tokens']+=chunk['supervised_tokens'];chunk_count+=1
            streams['documents.jsonl'].write(canonical({**row,**hashes,'source_code_tokens':len(ids),'source_utf8_bytes':len(raw),'document_id':hashes['sha256'],'chunks':chunk_count})+'\n')
            local_seen.add(hashes['sha256']);counts['documents']+=1;counts['payload_tokens']+=len(ids);counts['raw_bytes']+=len(raw)
            del ids,text,raw
    finally:
        for stream in streams.values(): stream.flush();os.fsync(stream.fileno());stream.close()
    artifacts={path.name:{'sha256':sha(path),'bytes':path.stat().st_size} for path in paths.values()}
    receipt={'schema':'sepalith.cpt.all-eligible-group.v1','status':'complete_with_repairs_pending' if counts['repair_items'] else 'complete',
             'seeded_index':index,'package':entry['name'],'group_id':entry['group_id'],'source_categories':dict(categories),
             'counts':dict(counts),'artifacts':artifacts,'seconds':time.monotonic()-started}
    write_json(stage/'receipt.json',receipt);fsync_dir(stage);stage.rename(final);fsync_dir(output/'groups')
    for value in local_seen: seen.add(value)
    return receipt


def summarize(receipts: dict[int, dict], totals: collections.Counter, expected: int) -> dict:
    repairs=sum(r['counts'].get('repair_items',0) for r in receipts.values())
    return {'schema':'sepalith.cpt.all-eligible-progress.v1','status':'complete' if len(receipts)==expected and repairs==0 else ('all_groups_inventoried_repairs_pending' if len(receipts)==expected else 'in_progress'),
            'groups_expected':expected,'groups_committed':len(receipts),'groups_remaining':expected-len(receipts),'repair_items_pending':repairs,
            'counts':dict(totals),'first_uncommitted_seeded_index':next((i for i in range(START_INDEX,START_INDEX+expected) if i not in receipts),None)}


def preflight(output: Path) -> tuple[list[dict],dict,dict]:
    source_manifest=json.loads(SOURCE_MANIFEST.read_text())
    if (source_manifest.get('schema')!='sepalith.cpt.all-eligible-source-manifest.v1'
            or source_manifest.get('materializer_sha256')!=sha(Path(__file__))):
        raise ValueError('materializer source manifest differs')
    for path,pin in PINS.items():
        if sha(path)!=pin: raise ValueError(f'input pin mismatch: {path}')
    order=json.loads(ORDER.read_text()); entries=order['packages'];
    if len(entries)!=8867 or len(entries[START_INDEX:])!=EXPECTED_GROUPS: raise ValueError('frozen global order extent differs')
    split=json.loads(SPLIT.read_text());registry={}
    for group in split['groups']:
        for form in group['identity_forms']:
            if form.startswith('pkg:'):
                key=form[4:].lower();value=(group['group_id'],group['split'])
                if key in registry and registry[key]!=value: raise ValueError('ambiguous package registry identity')
                registry[key]=value
    parts=json.loads((BASE/'cpt-train-group-partition.json').read_text())['groups']
    if sum(v=='cpt_validation' for v in parts.values())!=556: raise ValueError('CPT validation partition count differs')
    for entry in entries[START_INDEX:]:
        if entry['split']!='train_group' or registry.get(entry['name'].lower())!=(entry['group_id'],'train_group') or parts.get(entry['group_id'])!='cpt_train':
            raise ValueError(f'non-TRAIN or heldout entry in materialization slice: {entry}')
    source={'schema':'sepalith.cpt.all-eligible-source.v1','status':'preflight_pass','order_start':START_INDEX,'groups':EXPECTED_GROUPS,
            'supported_payload_category':'regular .R files recursively under the sole normalized package/version/package/R tree',
            'unsupported_categories':'symlink/nonregular .R, unreadable metadata, non-UTF8, NUL, or tokenizer-roundtrip failures enter explicit repair queues; no payload truncation',
            'caps':{'wall_time':None,'group_tokens':None,'package_tokens':None,'file_bytes':None},
            'heldout':{'cpt_validation_groups':556,'non_TRAIN_groups_rejected_before_payload_read':True},
            'materializer_sha256':source_manifest['materializer_sha256'],'pins':{str(k):v for k,v in PINS.items()}}
    return entries[START_INDEX:],registry,parts,source


def validate_source_migration(prior: dict, current: dict, output: Path) -> tuple[dict[int, dict], collections.Counter]:
    """Validate the exact approved predecessor transition and every prior commit."""
    migration=json.loads(SOURCE_MIGRATION.read_text())
    required={'schema','status','from_materializer_sha256','to_materializer_sha256','reason'}
    if (not isinstance(migration,dict) or set(migration)!=required
            or migration['schema']!='sepalith.cpt.all-eligible-source-migration.v1'
            or migration['status']!='admitted'
            or migration['from_materializer_sha256']!=prior.get('materializer_sha256')
            or migration['to_materializer_sha256']!=current.get('materializer_sha256')):
        raise ValueError('source migration is not exactly admitted')
    allowed={
        'move frozen recognized-license check before payload open; no row or token semantics change',
        'add exclusive output lock; no row or token semantics change',
        'classify frozen-license and empty-file outcomes as exclusions; no row or token semantics change',
        'recycle the tokenizer process after every atomic group; no row, token, split, license, provenance, dedup, or chunk semantics change',
    }
    if migration['reason'] not in allowed: raise ValueError('source migration reason is not admitted')
    comparison=dict(prior);comparison['materializer_sha256']=current['materializer_sha256']
    if comparison!=current: raise ValueError('source migration changed more than materializer identity')
    seen,_=base_seen();receipts,totals=committed(output,seen)
    if (migration['reason'].startswith('move frozen recognized-license')
            and totals.get('excluded_license',0)!=0):
        raise ValueError('predecessor receipts contain a license exclusion')
    return receipts,totals


def admit_source_migration(prior: dict, current: dict, output: Path) -> None:
    receipts,_=validate_source_migration(prior,current,output)
    migration=json.loads(SOURCE_MIGRATION.read_text())
    applied=output/f'source-migration-{migration["from_materializer_sha256"][:12]}-{migration["to_materializer_sha256"][:12]}.json'
    write_json(applied,{'migration':migration,'committed_groups_validated':len(receipts)})
    write_json(output/'run-manifest.json',current)


def write_seen_snapshot(output: Path, seen: set[str]) -> tuple[Path,str]:
    runtime=output/'.runtime';runtime.mkdir(exist_ok=True)
    path=runtime/'seen-sha256.txt';temporary=runtime/f'seen-sha256.{os.getpid()}.tmp'
    with temporary.open('x') as stream:
        for value in sorted(seen): stream.write(value+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);fsync_dir(runtime)
    return path,sha(path)


def load_seen_snapshot(path: Path, expected_sha: str) -> set[str]:
    if sha(path)!=expected_sha: raise ValueError('worker seen snapshot hash mismatch')
    values=set(path.read_text().splitlines())
    if any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in values):
        raise ValueError('worker seen snapshot contains a non-SHA256 identity')
    return values


def verify_inherited_lock(fd: int, output: Path) -> None:
    if fd < 0: raise ValueError('worker requires inherited parent lock')
    inherited=os.fstat(fd);actual=(output/'process.lock').stat()
    if (inherited.st_dev,inherited.st_ino)!=(actual.st_dev,actual.st_ino):
        raise ValueError('worker lock does not identify output process.lock')
    if os.environ.get('SEPALITH_CPT_PARENT_PID')!=str(os.getppid()):
        raise ValueError('worker parent identity mismatch')


def run_worker(args, entries: list[dict], registry: dict, parts: dict, source: dict) -> None:
    output=args.output;verify_inherited_lock(args.worker_lock_fd,output)
    if json.loads((output/'run-manifest.json').read_text())!=source:
        raise ValueError('worker source identity differs from admitted run manifest')
    index=args.worker_index
    if index<START_INDEX or index>=START_INDEX+EXPECTED_GROUPS: raise ValueError('worker index outside frozen order')
    entry=entries[index-START_INDEX]
    final=output/'groups'/f'{index:06d}-{entry["group_id"]}'
    if final.exists(): raise ValueError('worker refuses an already committed group')
    seen=load_seen_snapshot(args.seen_snapshot,args.seen_snapshot_sha256)
    protected=set(json.loads((BASE/'known-nontrain-parent-hashes.json').read_text()))
    raw_cpt=load_module('sepalith_all_eligible_raw_cpt',BASE/'raw_cpt_broader.py')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(TOKENIZER));tokenizer.encode_special_tokens=True
    receipt=process_group(output,index,entry,tokenizer,raw_cpt,registry,parts,seen,protected)
    print(canonical({'worker_receipt':receipt,'worker_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)


def invoke_worker(output: Path, index: int, lock_fd: int, seen: set[str]) -> tuple[dict,int]:
    snapshot,snapshot_sha=write_seen_snapshot(output,seen)
    command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker-index',str(index),
             '--worker-lock-fd',str(lock_fd),'--seen-snapshot',str(snapshot),'--seen-snapshot-sha256',snapshot_sha]
    environment=dict(os.environ);environment['SEPALITH_CPT_PARENT_PID']=str(os.getpid())
    result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,check=False,pass_fds=(lock_fd,),env=environment)
    if result.returncode:
        raise RuntimeError(f'group worker {index} failed rc={result.returncode}: {result.stderr[-4000:]}')
    lines=[line for line in result.stdout.splitlines() if line.strip()]
    if len(lines)!=1: raise RuntimeError(f'group worker {index} emitted unexpected stdout')
    envelope=json.loads(lines[0]);return envelope['worker_receipt'],envelope['worker_peak_rss_kib']


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--validate-resume-only',action='store_true');parser.add_argument('--max-groups',type=int)
    parser.add_argument('--worker-index',type=int);parser.add_argument('--worker-lock-fd',type=int,default=-1)
    parser.add_argument('--seen-snapshot',type=Path);parser.add_argument('--seen-snapshot-sha256');args=parser.parse_args()
    if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    try: os.nice(10)
    except OSError: pass
    os.environ.update(TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    entries,registry,parts,source=preflight(args.output)
    if args.preflight_only: print(json.dumps(source,sort_keys=True));return
    if args.worker_index is not None:
        if args.seen_snapshot is None or args.seen_snapshot_sha256 is None: raise ValueError('worker seen snapshot arguments required')
        run_worker(args,entries,registry,parts,source);return
    if args.max_groups is not None and args.max_groups<1: raise ValueError('--max-groups must be positive when set')
    output=args.output
    if args.validate_resume_only:
        validation_lock=(output/'process.lock').open('r')
        try: fcntl.flock(validation_lock.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('cannot validate resume while a materializer owns this output')
        prior=json.loads((output/'run-manifest.json').read_text());receipts,totals=validate_source_migration(prior,source,output)
        print(canonical({'status':'resume_validation_pass','committed_groups':len(receipts),'counts':dict(totals),
                         'from_source':prior['materializer_sha256'],'to_source':source['materializer_sha256'],'output_mutated':False}));return
    output.mkdir(parents=True,exist_ok=True)
    process_lock=(output/'process.lock').open('a')
    try: fcntl.flock(process_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise ValueError('another materializer owns this output')
    if (output/'run-manifest.json').exists():
        prior=json.loads((output/'run-manifest.json').read_text())
        if prior!=source: admit_source_migration(prior,source,output)
    else:
        (output/'groups').mkdir();(output/'.staging').mkdir();write_json(output/'run-manifest.json',source);fsync_dir(output)
    exact,protected=base_seen();receipts,totals=committed(output,exact)
    processed=0
    def stop(*_):
        global STOP; STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for offset,entry in enumerate(entries):
        index=START_INDEX+offset
        if index in receipts: continue
        if STOP or (args.max_groups is not None and processed>=args.max_groups): break
        receipt,worker_peak_rss_kib=invoke_worker(output,index,process_lock.fileno(),exact);processed+=1
        receipts[index]=receipt;totals.update(receipt['counts'])
        with (output/'groups'/f'{index:06d}-{entry["group_id"]}'/'documents.jsonl').open() as stream:
            for row in map(json.loads,stream):
                if row['sha256'] in exact: raise ValueError('worker committed a duplicate document')
                exact.add(row['sha256'])
        progress=summarize(receipts,totals,EXPECTED_GROUPS);write_json(output/'progress.json',progress)
        with (output/'progress.jsonl').open('a') as stream: stream.write(canonical({'at':dt.datetime.now(dt.timezone.utc).isoformat(),**progress,'last_receipt':receipt,'worker_peak_rss_kib':worker_peak_rss_kib})+'\n');stream.flush();os.fsync(stream.fileno())
        print(canonical({'seeded_index':index,'package':entry['name'],'status':receipt['status'],'counts':receipt['counts'],'worker_peak_rss_kib':worker_peak_rss_kib,'progress':{k:progress[k] for k in ('groups_committed','groups_remaining','repair_items_pending')}}),flush=True)
    result=summarize(receipts,totals,EXPECTED_GROUPS);result['stopped_by_signal']=STOP;result['invocation_groups_processed']=processed;write_json(output/'terminal.json',result);print(canonical(result))

if __name__=='__main__': main()
