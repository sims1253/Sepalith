#!/usr/bin/env python3
"""Materialize the exact eligible-15006 TRAIN PromptContext sidecar."""
from __future__ import annotations
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PROTOCOL=PLAN/'docs/campaign/work/lead/r2-expanded-rl-reward-coverage-v2/source/packages/sepalith/src'
sys.path.insert(0,str(PROTOCOL))
from sepalith.campaign_protocol import PromptContext,build_training_row,render_prompt,validate_training_row
from transformers import AutoTokenizer

ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
OLD=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-context-v1/context-sidecar.jsonl')
NEW=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
CANDIDATE_ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl')
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged')
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1')
EXPECTED={
 'rows':'65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7',
 'old':'378ef3aaaa13017ff83b27c36c5e04f0c5a2e8f5598eb3891a85dd5d0de00cee',
 'new':'e499c07d6da2ef325f7d9f3156b6c2e70361d3b1bc140198b6c3563b4471e7a0',
 'candidate_rows':'7887686e022e520e746c04c6197a9a3c2484fba2668d09a650d0627ff267187b',
 'tokenizer_json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
}
HELD={'8451310ff8e0c5d9f3e77bbc','dd65bd2cd11f38e729a9c712'}
FORBIDDEN={'reward','score','advantage','return','target','target_text','target_body','target_tokens','region_new','model_target','completion','gold','reference_answer'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def leak(x):
 if isinstance(x,Mapping):
  return any(str(k).casefold() in FORBIDDEN or leak(v) for k,v in x.items())
 if isinstance(x,list):return any(leak(v) for v in x)
 return False
def index_jsonl(path,key):
 result={};h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   off=f.tell();raw=f.readline()
   if not raw:break
   h.update(raw);x=json.loads(raw);rid=x[key]
   if rid in result:raise ValueError(f'duplicate:{path}:{rid}')
   result[rid]=(off,len(raw))
 return result,h.hexdigest()
def read_at(f,item):
 f.seek(item[0]);raw=f.read(item[1]);return raw,json.loads(raw)
def new_sidecar(row,source,line_no):
 rid=row['id'];ctx=source['context'];selection=source['selection'];ref=source['source_ref'];prov=source['source_provenance']
 if source['row_id']!=rid or source['prompt_sha256']!=hashlib.sha256(row['prompt_text'].encode()).hexdigest():raise ValueError(f'new_identity:{rid}')
 geometry={k:selection.get(k) for k in ('availability','budget_utf16_units','document_sha256','policy_id','policy_id_combined','required_utf16_units','used_utf16_units','overflow','required_overflow','spans','region')}
 geometry['context_range']=ctx['replacement_range'];geometry['document_version_policy']='offline_static_source; zero is valid; no live-editor freshness asserted'
 identity={'candidate_file':str(CANDIDATE_ROWS),'candidate_file_sha256':EXPECTED['candidate_rows'],'candidate_line':line_no,
  'registry_provenance_id':rid,'registry_provenance_decision':'admitted','group_id':ref['group_id'],'package_id':row['package_id'],
  'source_ref':ref,'source_provenance':prov}
 return {'row_id':rid,'context':ctx,'source_identity':identity,'selection_geometry':geometry,'family':row['family'],
  'package_id':row['package_id'],'split':'train','prompt_sha256':source['prompt_sha256'],
  'context_has_target_or_reward_keys':False,'offline_static_source':True}

def main():
 if OUT.exists():raise FileExistsError(OUT)
 for p,k in [(ROWS,'rows'),(OLD,'old'),(NEW,'new'),(CANDIDATE_ROWS,'candidate_rows')]:
  if sha(p)!=EXPECTED[k]:raise ValueError(f'hash:{k}')
 if sha(TOKENIZER/'tokenizer.json')!=EXPECTED['tokenizer_json']:raise ValueError('tokenizer_json_hash')
 old_idx,old_sha=index_jsonl(OLD,'row_id');new_idx,new_sha=index_jsonl(NEW,'row_id');cand_idx,cand_sha=index_jsonl(CANDIDATE_ROWS,'id')
 if old_sha!=EXPECTED['old'] or new_sha!=EXPECTED['new'] or cand_sha!=EXPECTED['candidate_rows']:raise ValueError('indexed_hash')
 if len(old_idx)!=11505 or len(new_idx)!=len(cand_idx)!=3503 or set(old_idx)&set(new_idx):raise ValueError('source_counts')
 if not HELD<=set(old_idx):raise ValueError('held_not_in_old')
 tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,trust_remote_code=False)
 attempt=OUT.with_name(OUT.name+f'.attempt-{os.getpid()}');attempt.mkdir(parents=True)
 side=attempt/'context-sidecar.jsonl';ids_path=attempt/'selected-train-ids.json';ledger=attempt/'source-order-manifest.jsonl'
 counts=Counter();ids=[];seen=set();prompt_seen=set();maxlens=Counter()
 with ROWS.open('rb') as rf,OLD.open('rb') as of,NEW.open('rb') as nf,side.open('wb') as sf,ledger.open('w') as lf:
  for pos,rawrow in enumerate(rf):
   row=json.loads(rawrow);rid=row['id'];validate_training_row(row)
   if rid in seen or rid in HELD or row['split']!='train':raise ValueError(f'row_identity:{rid}')
   seen.add(rid);ids.append(rid)
   if rid in old_idx:
    raw,sc=read_at(of,old_idx[rid]);origin='root_verified_11505_byte_exact';source_line=None;sf.write(raw);counts['old']+=1
   elif rid in new_idx:
    source_line=list(new_idx).index(rid)+1 if False else 0
    raw,source=read_at(nf,new_idx[rid]);candidate_line=read_at_index_line(new_idx,rid)
    sc=new_sidecar(row,source,candidate_line);sf.write(canonical(sc)+b'\n');origin='admitted_new_3503';source_line=candidate_line;counts['new']+=1
   else:raise ValueError(f'context_missing:{rid}')
   if sc['row_id']!=rid or sc['family']!=row['family'] or sc['package_id']!=row['package_id'] or sc['split']!='train':raise ValueError(f'sidecar_join:{rid}')
   if leak(sc['context']) or sc['context_has_target_or_reward_keys'] is not False:raise ValueError(f'context_leak:{rid}')
   context=PromptContext.from_mapping(sc['context']);rendered=render_prompt(context)
   if rendered!=row['prompt_text'] or hashlib.sha256(rendered.encode()).hexdigest()!=sc['prompt_sha256']:raise ValueError(f'render:{rid}')
   region_new=(context.region_old if row['target_operation']=='no_op' else () if row['target_operation']=='delete' else tuple(row['target_body_text'].split('\n')))
   rebuilt=build_training_row(context,operation=row['target_operation'],region_new=region_new,tokenizer=tokenizer,row_id=rid,family=row['family'],package_id=row['package_id'],split='train')
   if rebuilt!=row:raise ValueError(f'tokenizer_or_row_parity:{rid}')
   repl=context.replacement_range;geom=sc['selection_geometry']
   if geom['document_sha256']!=repl.content_sha256 or geom['context_range']!=repl.to_dict() or geom['overflow'] is not False or geom['required_overflow'] is not False:raise ValueError(f'geometry:{rid}')
   prompt_sha=sc['prompt_sha256']
   if prompt_sha in prompt_seen:raise ValueError(f'duplicate_prompt:{rid}')
   prompt_seen.add(prompt_sha);maxlens['max_sequence']=max(maxlens['max_sequence'],len(row['input_ids']));maxlens['max_target']=max(maxlens['max_target'],len(row['input_ids'])-row['target_start'])
   lf.write(json.dumps({'position':pos,'row_id':rid,'origin':origin,'source_line':source_line,'prompt_sha256':prompt_sha,'context_sha256':hashlib.sha256(canonical(sc['context'])).hexdigest(),'sidecar_record_sha256':hashlib.sha256(canonical(sc)).hexdigest()},sort_keys=True,separators=(',',':'))+'\n')
 if len(ids)!=len(seen)!=15006 or counts!=Counter(old=11503,new=3503):raise ValueError((len(ids),counts))
 ids_path.write_text(json.dumps({'schema_version':'sepalith.prm07.selected-train-ids.v1','split':'train','row_ids':ids},separators=(',',':'))+'\n')
 outputs={p.name:{'path':str(OUT/p.name),'sha256':sha(p),'bytes':p.stat().st_size} for p in (side,ids_path,ledger)}
 report={'schema':'sepalith.rl11.context15006-materialization.v1','status':'prepared_root_review_required_no_launch','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'inputs':{'eligible_rows':{'path':str(ROWS),'sha256':EXPECTED['rows']},'existing_context':{'path':str(OLD),'sha256':EXPECTED['old'],'rows':11505},'new_context':{'path':str(NEW),'sha256':EXPECTED['new'],'rows':3503},'candidate_rows':{'path':str(CANDIDATE_ROWS),'sha256':EXPECTED['candidate_rows']},'tokenizer_json_sha256':EXPECTED['tokenizer_json']},
  'coverage':{'rows':15006,'distinct_ids':15006,'existing_byte_exact':11503,'new_contexts':3503,'held_ids':sorted(HELD),'other_omissions':0,'duplicate_prompts':0,'max_sequence_tokens':maxlens['max_sequence'],'max_target_tokens':maxlens['max_target']},
  'checks':{'all_prompt_renders_exact':True,'all_training_rows_rebuilt_exact_with_pinned_tokenizer':True,'all_replacement_geometry_exact':True,'recursive_context_target_reward_leak_scan':True,'existing_records_raw_byte_exact':True,'new_source_provenance_preserved':True,'exact_order':True},
  'outputs':outputs,'scope':{'cpu_threads_max':2,'gpu':False,'cloud':False,'dev_final':False,'training_launch':False}}
 (attempt/'materialization.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');os.replace(attempt,OUT);print(json.dumps(report['coverage'],sort_keys=True))

def read_at_index_line(index,rid):
 # dict insertion order is exact JSONL line order.
 for number,key in enumerate(index,1):
  if key==rid:return number
 raise KeyError(rid)
if __name__=='__main__':main()
