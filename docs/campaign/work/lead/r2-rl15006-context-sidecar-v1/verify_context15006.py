#!/usr/bin/env python3
"""Independent full streaming verification of the exact15006 context sidecar."""
from __future__ import annotations
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
from typing import Mapping
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PROTOCOL=PLAN/'docs/campaign/work/lead/r2-expanded-rl-reward-coverage-v2/source/packages/sepalith/src';sys.path.insert(0,str(PROTOCOL))
from sepalith.campaign_protocol import PromptContext,build_training_row,render_prompt
from transformers import AutoTokenizer
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1')
ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
OLD=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-expanded-context-v1/context-sidecar.jsonl')
NEW=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
TOK=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-merged')
HELD={'8451310ff8e0c5d9f3e77bbc','dd65bd2cd11f38e729a9c712'}
FORBIDDEN={'reward','score','advantage','return','target','target_text','target_body','target_tokens','region_new','model_target','completion','gold','reference_answer'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def leak(x):
 if isinstance(x,Mapping):return any(str(k).casefold() in FORBIDDEN or leak(v) for k,v in x.items())
 if isinstance(x,list):return any(leak(v) for v in x)
 return False
def raw_index(p):
 d={}
 with p.open('rb') as f:
  for n,raw in enumerate(f,1):
   x=json.loads(raw);d[x['row_id']]=(hashlib.sha256(raw).hexdigest(),x,n)
 return d
def main():
 m=json.loads((OUT/'materialization.json').read_text())
 for name in ('context-sidecar.jsonl','selected-train-ids.json','source-order-manifest.jsonl'):
  x=m['outputs'][name];assert sha(OUT/name)==x['sha256'] and (OUT/name).stat().st_size==x['bytes']
 ids=json.loads((OUT/'selected-train-ids.json').read_text());assert ids['schema_version']=='sepalith.prm07.selected-train-ids.v1' and ids['split']=='train'
 old=raw_index(OLD);new=raw_index(NEW);tok=AutoTokenizer.from_pretrained(str(TOK),local_files_only=True,trust_remote_code=False)
 seen=set();counts=Counter();ordered=[]
 with ROWS.open('rb') as rf,(OUT/'context-sidecar.jsonl').open('rb') as sf,(OUT/'source-order-manifest.jsonl').open() as lf:
  for pos,(rr,sr,lr) in enumerate(zip(rf,sf,lf,strict=True)):
   row=json.loads(rr);side=json.loads(sr);ledger=json.loads(lr);rid=row['id'];ordered.append(rid)
   assert rid==side['row_id']==ledger['row_id'] and ledger['position']==pos and rid not in seen and rid not in HELD;seen.add(rid)
   assert row['split']==side['split']=='train' and side['family']==row['family'] and side['package_id']==row['package_id']
   assert side['context_has_target_or_reward_keys'] is False and side['offline_static_source'] is True and not leak(side['context'])
   context=PromptContext.from_mapping(side['context']);rendered=render_prompt(context)
   assert rendered==row['prompt_text'] and hashlib.sha256(rendered.encode()).hexdigest()==side['prompt_sha256']
   region_new=(context.region_old if row['target_operation']=='no_op' else () if row['target_operation']=='delete' else tuple(row['target_body_text'].split('\n')))
   assert build_training_row(context,operation=row['target_operation'],region_new=region_new,tokenizer=tok,row_id=rid,family=row['family'],package_id=row['package_id'],split='train')==row
   geom=side['selection_geometry'];repl=context.replacement_range
   assert geom['document_sha256']==repl.content_sha256 and geom['context_range']==repl.to_dict() and geom['overflow'] is False and geom['required_overflow'] is False
   if rid in old:
    assert hashlib.sha256(sr).hexdigest()==old[rid][0] and ledger['origin']=='root_verified_11505_byte_exact';counts['old']+=1
   else:
    source=new[rid][1];assert side['context']==source['context'] and side['source_identity']['source_ref']==source['source_ref'] and side['source_identity']['source_provenance']==source['source_provenance'] and side['source_identity']['candidate_line']==new[rid][2];counts['new']+=1
 assert ordered==ids['row_ids'] and len(seen)==15006 and counts==Counter(old=11503,new=3503) and not (seen&HELD)
 result={'schema':'sepalith.rl11.context15006-independent-verification.v1','status':'pass','rows':15006,'distinct_ids':15006,'counts':dict(counts),'held_ids_absent':sorted(HELD),'other_omissions':0,
  'checks':{'output_hashes':True,'rows_sidecar_ledger_selected_ids_exact_order':True,'all_prompt_renders_exact':True,'all_rows_retokenized_and_rebuilt_exact':True,'all_context_geometry_exact':True,'recursive_context_leak_scan':True,'existing_raw_records_byte_exact':True,'new_context_and_source_provenance_exact':True},
  'generated_r_executed':False,'dev_final_used':False}
 p=PLAN/'docs/campaign/work/lead/r2-rl15006-context-sidecar-v1/independent-verification.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
