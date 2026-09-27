#!/usr/bin/env python3
"""Deterministically materialize TRAIN-only SPEC panels; no model or target load."""
import hashlib,json,os
from pathlib import Path
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
SHORT=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
LONG=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-full-file-token-rows.jsonl')
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')
PINS={'short':'65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7','long':'752131a0ade6c1bd30248b9663046e027cd8ac9bdb38170a2b680f9dbae6d897','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def key(band,row):return hashlib.sha256(f'sepalith-spec-v2|{band}|{row["id"]}'.encode()).hexdigest()
def choose(path,nested,lo,hi,count):
 candidates=[]
 with path.open() as f:
  for line_no,line in enumerate(f,1):
   outer=json.loads(line);row=outer['token_row'] if nested else outer
   if row.get('split')!='train' or row.get('tokenizer_json_sha256')!=PINS['tokenizer']:continue
   n=row.get('prompt_token_count')
   if type(n) is int and lo<=n<=hi:
    candidates.append((key(f'{lo}-{hi}',row),line_no,row,outer.get('admission') if nested else 'admitted_15006'))
 candidates.sort(key=lambda x:x[0]);assert len(candidates)>=count,(path,len(candidates),count)
 return candidates[:count]
def normalize(row,tokenizer):
 ids=row['input_ids'];start=row['target_start'];prompt_ids=ids[1:start];target_ids=ids[start:-1]
 prompt=row.get('prompt_text') or tokenizer.decode(prompt_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 target=row.get('target_text') or tokenizer.decode(target_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 assert tokenizer.encode(prompt,add_special_tokens=False)==prompt_ids
 assert tokenizer.encode(target,add_special_tokens=False)==target_ids
 out=dict(row);out['prompt_text']=prompt;out['target_text']=target;out['target_body_text']=out.get('target_body_text') or tokenizer.decode(out['target_body_tokens'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
 assert out['bos_token_id']==0 and out['eos_token_id']==1 and ids[0]==0 and ids[-1]==1
 assert len(prompt_ids)==out['prompt_token_count'];return out
def emit(name,selected,profile,tokenizer):
 rows=[];records=[]
 out=HERE/f'{name}-panel.jsonl'
 with out.open('wb') as f:
  for band,source,line_no,row,admission,selection_key in selected:
   x=normalize(row,tokenizer);raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode();f.write(raw);rows.append(x)
   records.append({'row_id':x['id'],'split':'train','band':band,'family':x['family'],'package_id':x['package_id'],'prompt_token_count':x['prompt_token_count'],'row_sha256':hashlib.sha256(raw).hexdigest(),'source':source,'source_line':line_no,'source_admission':admission,'selection_key':selection_key})
 manifest={'schema_version':'sepalith.r2.serving.spec-train-panel.v2','status':'frozen_train_only','profile':profile,'panel':{'path':str(out),'sha256':sha(out),'row_count':len(rows),'records':records,'band_counts':{b:sum(r['band']==b for r in records) for b in ('2k','8k')},'prompt_min':min(x['prompt_token_count'] for x in rows),'prompt_max':max(x['prompt_token_count'] for x in rows)},'sources':{'short':{'path':str(SHORT),'sha256':PINS['short']},'long':{'path':str(LONG),'sha256':PINS['long']},'tokenizer_json_sha256':PINS['tokenizer']},'policy':{'split':'train','contains_dev_or_final':False,'no_authored_target_tail_at_inference':True,'selection':'stable_sha256_without_model_outputs','training_admission_claim':False}}
 (HERE/f'{name}-panel.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 return manifest
assert sha(SHORT)==PINS['short'] and sha(LONG)==PINS['long'] and sha(TOKENIZER/'tokenizer.json')==PINS['tokenizer']
short=choose(SHORT,False,1536,2048,32);long=choose(LONG,True,4097,8192,32)
tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,trust_remote_code=False)
def records(s,l):
 return [('2k',str(SHORT),n,r,a,k) for k,n,r,a in s]+[('8k',str(LONG),n,r,a,k) for k,n,r,a in l]
ng=emit('ngram',records(short[:20],long[:20]),{'id':'SPEC-CPU-NGRAM','context':10240,'cap':64,'repetitions':1,'expected_requests':40},tok)
dr=emit('draft',records(short[20:32],long[20:32]),{'id':'SPEC-DRAFT-PAIR','context':10240,'cap':64,'repetitions':2,'expected_requests':48},tok)
assert not ({r['row_id'] for r in ng['panel']['records']} & {r['row_id'] for r in dr['panel']['records']})
print(json.dumps({'ngram':ng['panel']['band_counts'],'draft':dr['panel']['band_counts'],'disjoint':True}))
