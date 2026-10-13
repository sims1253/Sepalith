#!/usr/bin/env python3
"""Provisional, resumable recovery of aliases of already-admitted license families."""
from __future__ import annotations
import argparse, collections, datetime as dt, fcntl, hashlib, importlib.util, json, os, shutil, stat, sys, time, uuid
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
MAIN=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1')
BASE=PLAN/'docs/campaign/work/r2-corpus-preparation-v1'
GLOBAL=PLAN/'docs/campaign/work/r2-cpt-global-shard-v1'
SPLIT=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
HERE=Path(__file__).resolve().parent
PINS={
 HERE/'license_aliases.py':'57dcde114ff95fea0ef97423d2d7e5f2f865a4949bf32ea8c62158ca9bcf7975',
 HERE/'raw_cpt_broader.py':'84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab',
 HERE/'campaign_cpt_data.py':'8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa',
 PLAN/'docs/campaign/work/lead/r2-cpt-license-exclusion-census-v1/census.json':'5216d2ba5e105456b5a570295119ea0a5217a20fdc460b441fbd1c59d99b0e5a',
 BASE/'cpt-train-group-partition.json':'6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06',
 BASE/'known-nontrain-parent-hashes.json':'7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0',
 BASE/'broader-shard-v1-2k/documents.jsonl':'98b9e4a3df7aa2bda79365a0521ac538628406be0cc9af3a0da1eb3ba5ab5d11',
 BASE/'profile-shard-v1/documents.jsonl':'674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68',
 GLOBAL/'shard/documents.jsonl':'8a8cf09059afa42fcc6915d8b0a7645558115cc440d521daeedfa6d2a9305804',
 SPLIT:'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09',
 TOKENIZER:'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
}
RECOVERABLE=frozenset({
 'BSD_2_clause + file LICENSE','BSD_2_clause + file LICENCE',
 'BSD_3_clause + file LICENSE','BSD_3_clause + file LICENCE',
 'GNU General Public License','Mozilla Public License 2.0','Mozilla Public License Version 2.0',
})

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 assert spec.loader is not None;spec.loader.exec_module(m);return m

def write_json(path,value):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def fingerprints(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'sha1':hashlib.sha1(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
def fields(text):
 result={};key=None
 for line in text.splitlines():
  if line[:1].isspace() and key:result[key]+=' '+line.strip()
  elif ':' in line:key,value=line.split(':',1);result[key]=value.strip()
 return result

def base_seen():
 exact=set();protected=set(json.loads((BASE/'known-nontrain-parent-hashes.json').read_text()))
 for path in (BASE/'broader-shard-v1-2k/documents.jsonl',GLOBAL/'shard/documents.jsonl'):
  for row in map(json.loads,path.open()):exact.add(row['sha256'])
 for row in map(json.loads,(BASE/'profile-shard-v1/documents.jsonl').open()):
  if row['cpt_partition']=='cpt_validation':exact.add(row['sha256'])
 return exact,protected

def recoverable_license(original,aliases,raw):
 if not isinstance(original,str) or original not in RECOVERABLE:return None
 normalized=aliases.normalize_recognized_alias(original)
 if normalized==original or raw.allowed_license(original) or not raw.allowed_license(normalized):return None
 return normalized

def capture_frontier(main,aliases,raw):
 groups=[]
 for p in main.joinpath('groups').iterdir():
  if p.is_dir() and len(p.name)>=6 and p.name[:6].isdigit():groups.append(p)
 groups.sort(key=lambda p:int(p.name[:6]))
 indexes=[int(p.name[:6]) for p in groups]
 if indexes!=list(range(775,775+len(indexes))):raise ValueError('main committed frontier is not contiguous from 775')
 seen,protected=base_seen();candidates=collections.defaultdict(list);main_docs=0
 for folder in groups:
  receipt=json.loads((folder/'receipt.json').read_text())
  if receipt['seeded_index']!=int(folder.name[:6]) or receipt['group_id'] not in folder.name:raise ValueError(f'group receipt identity mismatch: {folder}')
  for name in ('documents.jsonl','exclusions.jsonl'):
   path=folder/name;record=receipt['artifacts'][name]
   if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError(f'changed committed artifact: {path}')
  for row in map(json.loads,(folder/'documents.jsonl').open()):
   digest=row['sha256']
   if digest in seen:raise ValueError(f'duplicate document inside main committed frontier: {digest}')
   seen.add(digest);main_docs+=1
  for row in map(json.loads,(folder/'exclusions.jsonl').open()):
   normalized=recoverable_license(row.get('license'),aliases,raw)
   if row.get('reason')=='license_not_in_frozen_recognized_families' and normalized:
    candidates[folder.name].append({**row,'canonical_license':normalized})
 return groups,seen,protected,candidates,main_docs

def verify_split(group_id,package,registry,partitions):
 if registry.get(package.lower())!=(group_id,'train_group'):raise ValueError(f'{package}: not in exact TRAIN group')
 if partitions.get(group_id)!='cpt_train':raise ValueError(f'{package}: protected CPT validation group')

def read_candidate(candidate,inventory):
 path=Path(candidate['path']);meta=inventory.get(str(path))
 if meta is None:raise ValueError(f'missing source inventory row: {path}')
 for key in ('package','group_id','path','license'):
  if meta.get(key)!=candidate.get(key):raise ValueError(f'{path}: exclusion/inventory {key} mismatch')
 before=path.stat();expected=(meta['bytes'],meta['mtime_ns'],meta['inode'],meta['device'])
 if path.is_symlink() or not stat.S_ISREG(before.st_mode) or (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)!=expected:raise ValueError(f'{path}: source stat changed')
 with path.open('rb') as f:raw=f.read()
 after=path.stat()
 if len(raw)!=before.st_size or (after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)!=expected:raise ValueError(f'{path}: source changed during read')
 desc=Path(meta['description_path']);descraw=desc.read_bytes()
 if hashlib.sha256(descraw).hexdigest()!=meta['description_sha256']:raise ValueError(f'{path}: DESCRIPTION changed')
 dcf=fields(descraw.decode('utf-8'))
 if dcf.get('Package')!=meta['package'] or dcf.get('License')!=meta['license']:raise ValueError(f'{path}: package/license provenance mismatch')
 if dcf.get('License_restricts_use','').lower()=='yes' or dcf.get('License_is_FOSS','').lower()=='no':raise ValueError(f'{path}: package restrictions prohibit use')
 try:text=raw.decode('utf-8')
 except UnicodeDecodeError:return None,'non_UTF8_R_source_requires_repair',raw,meta
 if '\0' in text:return None,'NUL_R_source_requires_repair',raw,meta
 return text,None,raw,meta

def commit_group(output,folder,candidates,seen,protected,registry,partitions,tokenizer,rawmod,validator,alias_sha):
 final=output/'groups'/folder.name
 if final.exists():
  receipt=json.loads((final/'receipt.json').read_text())
  for name,record in receipt['artifacts'].items():
   p=final/name
   if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError(f'changed recovery commit: {p}')
  return receipt
 source_receipt=json.loads((folder/'receipt.json').read_text());invpath=folder/'source-inventory.jsonl';record=source_receipt['artifacts']['source-inventory.jsonl']
 if invpath.stat().st_size!=record['bytes'] or sha(invpath)!=record['sha256']:raise ValueError(f'changed source inventory: {invpath}')
 inventory={row['path']:row for row in map(json.loads,invpath.open())}
 stage=output/'.staging'/(folder.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
 names=('cpt_train.jsonl','documents.jsonl','dedup-exclusions.jsonl','repair-queue.jsonl');streams={n:(stage/n).open('x') for n in names};counts=collections.Counter();local=set();started=time.monotonic()
 try:
  for candidate in candidates:
   verify_split(candidate['group_id'],candidate['package'],registry,partitions);text,failure,raw,meta=read_candidate(candidate,inventory);hashes=fingerprints(raw)
   if failure:
    streams['repair-queue.jsonl'].write(canonical({**candidate,'reason':failure,'source_sha256_observed':hashes['sha256']})+'\n');counts['repair']+=1;continue
   reason=None
   if protected.intersection(hashes.values()):reason='known_nontrain_parent_hash_match'
   elif hashes['sha256'] in seen:reason='exact_duplicate_prior_or_current_main'
   elif hashes['sha256'] in local:reason='exact_duplicate_within_alias_recovery'
   if reason:
    streams['dedup-exclusions.jsonl'].write(canonical({**candidate,'reason':reason,'source_sha256_observed':hashes['sha256']})+'\n');counts['dedup_excluded']+=1;continue
   encoded=tokenizer.encode(text,add_special_tokens=False);ids=encoded.ids
   if not ids or tokenizer.decode(ids,skip_special_tokens=False)!=text:
    streams['repair-queue.jsonl'].write(canonical({**candidate,'reason':'tokenizer_empty_or_roundtrip_failure','source_sha256_observed':hashes['sha256']})+'\n');counts['repair']+=1;continue
   rows=[]
   for index,chunk in enumerate(rawmod.chunks(ids,2048)):
    row={'schema':1,'row_id':hashes['sha256']+':'+str(index),'document_id':hashes['sha256'],'package':candidate['package'],'group_id':candidate['group_id'],'cpt_partition':'cpt_train','source_path':candidate['path'],'source_sha256':hashes['sha256'],'chunk_index':index,**chunk}
    rows.append(row);streams['cpt_train.jsonl'].write(canonical(row)+'\n');counts['rows']+=1;counts['input_tokens']+=len(row['input_ids']);counts['loss_tokens']+=row['supervised_tokens']
   checked=validator.validate_materialized_rows(rows,max_sequence_tokens=2048,require_complete_documents=True)
   if checked['payload_tokens']!=len(ids):raise ValueError('frozen validator token coverage mismatch')
   streams['documents.jsonl'].write(canonical({**hashes,'document_id':hashes['sha256'],'package':candidate['package'],'group_id':candidate['group_id'],'cpt_partition':'cpt_train','source_path':candidate['path'],'source_utf8_bytes':len(raw),'source_code_tokens':len(ids),'chunks':len(rows),'original_license':candidate['license'],'canonical_license_for_existing_family':candidate['canonical_license'],'alias_normalizer_sha256':alias_sha,'source_inventory_sha256':record['sha256'],'provisional_pending_terminal_main_dedup':True})+'\n')
   local.add(hashes['sha256']);counts['documents']+=1;counts['payload_tokens']+=len(ids);counts['raw_bytes']+=len(raw)
  for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
  artifacts={n:{'bytes':(stage/n).stat().st_size,'sha256':sha(stage/n)} for n in names}
  receipt={'schema':'sepalith.cpt.license-alias-recovery-group.v1','status':'provisional_pending_terminal_main_dedup','main_group':folder.name,'main_group_receipt_sha256':sha(folder/'receipt.json'),'candidate_exclusion_records':len(candidates),'counts':dict(counts),'artifacts':artifacts,'seconds':time.monotonic()-started}
  write_json(stage/'receipt.json',receipt);fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);stage.rename(final);fd=os.open(final.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  seen.update(local);return receipt
 except BaseException:
  for stream in streams.values():
   if not stream.closed:stream.close()
  shutil.rmtree(stage);raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 os.environ['TOKENIZERS_PARALLELISM']='false'
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError(f'input pin mismatch: {p}')
 aliases=load('alias_recovery_aliases',HERE/'license_aliases.py');raw=load('alias_recovery_raw',HERE/'raw_cpt_broader.py');validator=load('alias_recovery_validator',HERE/'campaign_cpt_data.py')
 census=json.loads((PLAN/'docs/campaign/work/lead/r2-cpt-license-exclusion-census-v1/census.json').read_text())
 if sum(r['records'] for r in census['rows'] if r['license'] in RECOVERABLE)!=1661:raise ValueError('frozen census alias count differs')
 args.output.mkdir(parents=True,exist_ok=True);(args.output/'groups').mkdir(exist_ok=True);(args.output/'.staging').mkdir(exist_ok=True)
 lock=(args.output/'process.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binding={'schema':'sepalith.cpt.license-alias-recovery-binding.v1','status':'provisional_only','pins':{str(p):h for p,h in PINS.items()},'main':str(MAIN),'recoverable_original_licenses':sorted(RECOVERABLE)}
 bp=args.output/'binding.json'
 if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('output binding differs')
 if not bp.exists():write_json(bp,binding)
 groups,seen,protected,candidates,main_docs=capture_frontier(MAIN,aliases,raw)
 split=json.loads(SPLIT.read_text());registry={}
 for g in split['groups']:
  for form in g['identity_forms']:
   if form.startswith('pkg:'):registry[form[4:].lower()]=(g['group_id'],g['split'])
 partitions=json.loads((BASE/'cpt-train-group-partition.json').read_text())['groups']
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(str(TOKENIZER));tokenizer.encode_special_tokens=True
 counts=collections.Counter();processed=[]
 for folder in groups:
  if folder.name not in candidates:continue
  receipt=commit_group(args.output,folder,candidates[folder.name],seen,protected,registry,partitions,tokenizer,raw,validator,PINS[HERE/'license_aliases.py']);processed.append(folder.name);counts.update(receipt['counts'])
 main_progress=json.loads((MAIN/'progress.json').read_text())
 progress={'schema':'sepalith.cpt.license-alias-recovery-progress.v1','status':'provisional_pending_main_completion_and_terminal_dedup','captured_at':dt.datetime.now(dt.timezone.utc).isoformat(),'captured_main_groups':len(groups),'captured_first_uncommitted_seeded_index':775+len(groups),'main_progress_observed':main_progress,'main_documents_in_dedup_snapshot':main_docs,'base_and_main_sha256_seen':len(seen),'protected_hash_identities':len(protected),'recoverable_exclusion_records_at_frontier':sum(map(len,candidates.values())),'recovery_groups_at_frontier':len(candidates),'recovery_groups_committed':len(processed),'counts':dict(counts),'terminal_rescan_required':True,'admitted':False}
 write_json(args.output/'progress.json',progress);print(canonical(progress))

if __name__=='__main__':main()
