#!/usr/bin/env python3
"""Bounded streaming raw TRAIN materialization in the frozen global package order."""
import argparse,collections,concurrent.futures,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'r2-corpus-preparation-v1'
ORDER=HERE.parent/'r2-cpt-long-stage-review-v1/next-global-package-order.json'
PINS={ORDER:'c7255cbbb664b7a12322b7bb9acf40c1b23c73dc9e1869531443205dbc9335b8',BASE/'inventory_raw_train_v2.py':'7271bfaaa01eb0da485c221581e60a4e91ed91555c3fa8f4d4ee74da0367fce2',BASE/'raw_cpt_broader.py':'84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab',BASE/'raw-train-package-candidates.json':'44493229f5cea86798e1a256fdde679a87ed32f75591041614a2a00755e87841',BASE/'cpt-train-group-partition.json':'6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06',BASE/'known-nontrain-parent-hashes.json':'7210a342559278c2bd82f3514da899d6cfbbb89104c4d9c8af0df61fbb8133b0',BASE/'broader-shard-v1-2k/documents.jsonl':'98b9e4a3df7aa2bda79365a0521ac538628406be0cc9af3a0da1eb3ba5ab5d11',BASE/'profile-shard-v1/documents.jsonl':'674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68',Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json'):'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while chunk:=f.read(4*1024*1024):h.update(chunk)
 return h.hexdigest()
def load(name,path):
 if sha(path)!=PINS[path]:raise ValueError('source pin mismatch')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
c=load('r2_global_existing_cpt',BASE/'raw_cpt_broader.py')
inv=load('r2_global_existing_inventory',BASE/'inventory_raw_train_v2.py')

def admit_group(entry,registry,parts,used_groups):
 g=entry['group_id'];name=entry['name']
 if entry['split']!='train_group' or registry.get(name.lower())!=(g,'train_group'):raise ValueError('registry TRAIN mismatch')
 if parts.get(g)!='cpt_train' or c.partition(g)!='cpt_train':raise ValueError('reserved CPT validation group')
 if g in used_groups:raise ValueError('already emitted broad group')
 if not name or name in ('.','..') or '/' in name or '\\' in name:raise ValueError('invalid package name')
 return name,{'group_id':g}

def reject_reason(hashes,parent_hashes,broad_hashes,validation_hashes,seen):
 if parent_hashes.intersection(hashes.values()):return 'known_nontrain_parent_hash_match'
 if hashes['sha256'] in validation_hashes:return 'reserved_CPT_validation_document_duplicate'
 if hashes['sha256'] in broad_hashes:return 'already_emitted_broad_document_duplicate'
 if hashes['sha256'] in seen:return 'duplicate_within_global_shard'
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-seconds',type=int,default=720);ap.add_argument('--target-code-tokens',type=int,default=30000000);ap.add_argument('--group-code-token-cap',type=int,default=100000);a=ap.parse_args()
 assert 30<=a.max_seconds<=900 and 1000000<=a.target_code_tokens<=40000000 and 10000<=a.group_code_token_cap<=400000
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.environ.update(TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
 for path,pin in PINS.items():
  if sha(path)!=pin:raise ValueError('input pin mismatch: '+str(path))
 if sha(c.TOKENIZER)!=c.TOKENIZER_SHA:raise ValueError('tokenizer changed')
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(c.TOKENIZER));tok.encode_special_tokens=True
 order=json.loads(ORDER.read_text());packages=order['packages'];assert len(packages)==8867
 assert packages==sorted(packages,key=lambda x:hashlib.sha256(('DAT10-next-global-v1\0'+x['group_id']).encode()).digest())
 split=json.loads(inv.SPLIT.read_text());registry={}
 for g in split['groups']:
  for form in g['identity_forms']:
   if form.startswith('pkg:'):
    key=form[4:].lower();value=(g['group_id'],g['split']);assert key not in registry or registry[key]==value;registry[key]=value
 del split
 parts=json.loads((BASE/'cpt-train-group-partition.json').read_text())['groups'];assert sum(v=='cpt_validation' for v in parts.values())==556
 used_groups=set();broad_hashes=set();validation_hashes=set()
 for row in map(json.loads,(BASE/'broader-shard-v1-2k/documents.jsonl').open()):used_groups.add(row['group_id']);broad_hashes.add(row['sha256'])
 for row in map(json.loads,(BASE/'profile-shard-v1/documents.jsonl').open()):
  if row['cpt_partition']=='cpt_validation':validation_hashes.add(row['sha256'])
 assert len(used_groups)==1296 and len(broad_hashes)==14098 and len(validation_hashes)==294
 parent_hashes=set(json.loads((BASE/'known-nontrain-parent-hashes.json').read_text()))
 # Validate all package permissions before the first DESCRIPTION or R-directory read.
 entries=[admit_group(e,registry,parts,used_groups) for e in packages]
 assert len({e['group_id'] for e in packages})==len(packages)
 out=HERE/'shard';out.mkdir(exist_ok=False);counts=collections.Counter();exclusions=collections.Counter();group_tokens=collections.Counter();initials=collections.Counter();seen=set();started=time.monotonic();stop='target_code_tokens';completed_packages=0;last_index=-1
 handles={name:(out/name).open('x') for name in ['cpt_train.jsonl','cpt_validation.jsonl','documents.jsonl','selected-source-metadata.jsonl','exclusions.jsonl','package-receipts.jsonl']}
 def emit(name,value):handles[name].write(json.dumps(value,separators=(',',':'))+'\n')
 def exclude(row,reason):exclusions[reason]+=1;emit('exclusions.jsonl',{'package':row.get('package'),'path':row.get('path'),'reason':reason})
 # Two metadata lookups may be in flight. File prefetch is separately bounded to two files.
 metapool=concurrent.futures.ThreadPoolExecutor(max_workers=2);filepool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
 pending=collections.deque();cursor=0
 try:
  while cursor<min(2,len(entries)):pending.append((cursor,metapool.submit(inv.package,entries[cursor])));cursor+=1
  while pending:
   if time.monotonic()-started>=a.max_seconds:stop='time_budget_before_next_package';break
   index,future=pending.popleft();rows,rejects=future.result();last_index=index
   if cursor<len(entries):pending.append((cursor,metapool.submit(inv.package,entries[cursor])));cursor+=1
   for r in rejects:exclude(r,r['reason'])
   rows.sort(key=lambda r:hashlib.sha256(('DAT10-global-file-v1\0'+r['path']).encode()).digest())
   for row in rows:row['cpt_partition']='cpt_train'
   file_pending=collections.deque();rpos=0;before_docs=counts['documents'];before_tokens=counts['code_tokens'];partial=False
   while rpos<min(2,len(rows)):file_pending.append(filepool.submit(c.read_file,rows[rpos]));rpos+=1
   while file_pending:
    row,result,reason=file_pending.popleft().result()
    if rpos<len(rows):file_pending.append(filepool.submit(c.read_file,rows[rpos]));rpos+=1
    emit('selected-source-metadata.jsonl',row);counts['files_read']+=1
    if reason:exclude(row,reason);continue
    raw,text,hashes=result;reason=reject_reason(hashes,parent_hashes,broad_hashes,validation_hashes,seen)
    if reason:exclude(row,reason);continue
    ids=tok.encode(text,add_special_tokens=False).ids
    if not ids or tok.decode(ids,skip_special_tokens=False)!=text:raise ValueError('empty tokenization or byte roundtrip mismatch')
    if group_tokens[row['group_id']]+len(ids)>a.group_code_token_cap:exclude(row,'whole_document_exceeds_remaining_group_cap');continue
    chunk_count=0
    for k,chunk in enumerate(c.chunks(ids,2048)):
     record={'schema':1,'row_id':hashes['sha256']+':'+str(k),'document_id':hashes['sha256'],'package':row['package'],'group_id':row['group_id'],'cpt_partition':'cpt_train','source_path':row['path'],'source_sha256':hashes['sha256'],'chunk_index':k,**chunk};emit('cpt_train.jsonl',record);counts['rows']+=1;counts['input_tokens']+=len(chunk['input_ids']);counts['supervised_tokens']+=chunk['supervised_tokens'];chunk_count+=1
    emit('documents.jsonl',{**row,**hashes,'source_code_tokens':len(ids),'source_utf8_bytes':len(raw),'document_id':hashes['sha256'],'chunks':chunk_count})
    seen.add(hashes['sha256']);counts['documents']+=1;counts['code_tokens']+=len(ids);counts['raw_bytes']+=len(raw);group_tokens[row['group_id']]+=len(ids);initials[row['package'][0].upper()]+=len(ids)
    if counts['code_tokens']>=a.target_code_tokens or time.monotonic()-started>=a.max_seconds:partial=True;stop='target_code_tokens' if counts['code_tokens']>=a.target_code_tokens else 'time_budget_after_complete_document';break
   # Already-started reads are bounded; consume them only as observations, never partial token rows.
   for fut in file_pending:
    row,result,reason=fut.result();emit('selected-source-metadata.jsonl',row);counts['prefetched_not_emitted_files']+=1
   emit('package-receipts.jsonl',{'seeded_index':index,'package':packages[index]['name'],'group_id':packages[index]['group_id'],'metadata_files':len(rows),'emitted_documents':counts['documents']-before_docs,'emitted_code_tokens':counts['code_tokens']-before_tokens,'package_complete':not partial})
   completed_packages+=not partial
   if (index+1)%25==0:
    for f in handles.values():f.flush()
    print(json.dumps({'package_index':index,'counts':dict(counts),'groups':len(group_tokens),'seconds':time.monotonic()-started}),flush=True)
   if partial:break
  else:stop='entire_seeded_order_completed'
 finally:
  for _,future in pending:future.cancel()
  metapool.shutdown(wait=True,cancel_futures=True);filepool.shutdown(wait=True,cancel_futures=True)
  for f in handles.values():f.close()
 assert counts['supervised_tokens']==counts['code_tokens']+counts['documents']
 counts['packages']=len(group_tokens)
 manifest={'schema':1,'status':'CPU_materialized_candidate_not_training_admission','counts':{'cpt_train':dict(counts)},'max_length':2048,'tokenizer_path':str(c.TOKENIZER),'tokenizer_sha256':c.TOKENIZER_SHA,'bos_id':0,'eos_id':1,'pad_id':1,'source_sha256':sha(__file__),'reused_source_pins':{str(p):v for p,v in PINS.items()},'order_seed':order['seed'],'order_sha256':PINS[ORDER],'eligible_remaining_packages':len(packages),'last_seeded_index':last_index,'completed_packages':completed_packages,'metadata_packages_started':cursor,'stop_reason':stop,'requested_target_code_tokens':a.target_code_tokens,'max_seconds':a.max_seconds,'elapsed_seconds':time.monotonic()-started,'group_code_token_cap':a.group_code_token_cap,'maximum_group_code_tokens':max(group_tokens.values(),default=0),'maximum_group_fraction':max(group_tokens.values(),default=0)/max(1,counts['code_tokens']),'code_tokens_by_package_initial':dict(initials),'group_code_token_counts':dict(group_tokens),'exclusions':dict(exclusions),'existing_broad_group_overlap':len(set(group_tokens)&used_groups),'existing_broad_document_overlap':len(seen&broad_hashes),'reserved_validation_document_overlap':len(seen&validation_hashes),'reserved_validation_groups_excluded':556,'boundaries':'Exact raw_cpt_broader.chunks:2K,BOS and overlap masked,nonterminal EOS masked,one actual EOS supervised;complete retained documents only','sampling_limit':'Prefix of a seeded order over all8867remainingTRAINpackages;not whole-corpus coverage;per-group whole-document cap','io_bound':'Two metadata futures plus two file futures;no unbounded executor.map of raw file results','model_framework_imported':any(x in sys.modules for x in ['torch','transformers']),'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['group_code_token_counts','reused_source_pins','artifacts']}),flush=True)

if __name__=='__main__':main()
