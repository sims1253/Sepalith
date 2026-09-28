#!/usr/bin/env python3
"""Bind a terminal no-op4100 reconstruction to two provider lanes."""
import argparse,collections,hashlib,json,os,tempfile
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PROVIDER=PLAN/'docs/campaign/work/lead/r2-noop4100-provider-preparation-v1'
PROVIDER_SOURCE_SHA='6e6dfd87116bb8743fa1d593c857c7e2108a4d5e87f3a0814ff36e998b3cd424'
RUN_SHARD_SHA='9fd8d8ef6730871ffa1a9703e3af4b92dd4e265c07976fa547803e23e6c3e3fb'
RECONSTRUCTION_SOURCE=PLAN/'docs/campaign/work/lead/r2-noop4100-reconstruction-preparation-v1/source-manifest.json'
RECONSTRUCTION_SOURCE_SHA='b81bfe2ca7e4e1eb503775b8150d54d7cdf9d5602eea154f53f6adffe2d413b3'
COVERAGE_SHA='8c11cf68cf1d2a91dd92cbe29327b887fe0efc46943d8302760b5f3a80c16df8';CANDIDATES_SHA='59e4d0b57fa2185503e1ab4a1bad1b6e43acf4bc70e239508b800204a31a7ca7'
RUNTIME_FILES={'render_shard.ts':'f5c5e57e9f33cb6503586ee51c9aa0fd1e38d017f6f74edef8b4e45423e9c2a6','source/namespace_runtime.ts':'e11868caa0d96a216f399d7772098d89ac415fe4db36ba69ea9e3c0d96140b12','source/source_imports.R':'3f140b69f9ff75d48d3305a5f27d6d2b8d8a49a4f97eaa9b0a2effe963801925'}
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def rows(path):
 with Path(path).open()as f:return [json.loads(x)for x in f if x.strip()]
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def target_free(row):
 bad=[k for k in row if 'target' in k.lower()or'gold'in k.lower()]
 req(not bad,f'provider input target/gold keys:{row.get("row_id")}:{bad}')

def build(input_root,manifest_sha,coverage_path,candidates_path,output):
 input_root=Path(input_root);manifest_path=input_root/'manifest.json';req(sha(manifest_path)==manifest_sha,'terminal reconstruction manifest hash differs');manifest=json.loads(manifest_path.read_text());coverage=json.loads(Path(coverage_path).read_text());candidates=rows(candidates_path)
 req(sha(RECONSTRUCTION_SOURCE)==RECONSTRUCTION_SOURCE_SHA and sha(coverage_path)==COVERAGE_SHA and sha(candidates_path)==CANDIDATES_SHA,'reconstruction source/census differs')
 req(sha(PROVIDER/'provider-runtime-manifest.json')==PROVIDER_SOURCE_SHA and sha(PROVIDER/'run_shard.sh')==RUN_SHARD_SHA and all(sha(PROVIDER/p)==d for p,d in RUNTIME_FILES.items()) and sha(TOKENIZER)==TOKENIZER_SHA,'provider runtime differs')
 req(manifest.get('schema')=='sepalith.dat10.noop4100.reconstruction.v1' and manifest.get('status')=='complete_review_only' and manifest.get('full_41_shard_closure') is True,'reconstruction not terminal full41')
 req((manifest.get('candidate_rows'),manifest.get('prediction_inputs'),manifest.get('holds'))==(4227,4100,127),'reconstruction denominator')
 req(manifest.get('hold_accounting',{}).get('provenance')==121 and manifest['hold_accounting'].get('reviewed_mixed_eol')==6,'hold classes')
 req(coverage.get('completed_shards')==41 and coverage.get('pending_shards')==[] and coverage.get('partial')is False,'coverage not full41')
 req(len(candidates)==len({x['row_id']for x in candidates})==4227,'candidate IDs')
 census=collections.Counter(x['shard']for x in candidates);bound={x['shard']:x for x in manifest['shards']};req(set(bound)==set(range(41))and len(manifest['shards'])==41,'manifest must enumerate all 41 shards')
 entries=[];prediction_by_id={};hold_total=0
 for shard in range(41):
  item=bound[shard];req(item['candidates']==census[shard] and item['prediction_inputs']+item['holds']==item['candidates'],'shard accounting')
  paths={kind:input_root/f'shard-{shard:04d}{suffix}'for kind,suffix in [('prediction','.jsonl'),('sidecar','.sidecar.jsonl'),('holds','.holds.jsonl')]}
  req(all(x.is_file()for x in paths.values()),'all shard artifacts required including empty')
  req(sha(paths['prediction'])==item['prediction_sha256'] and sha(paths['sidecar'])==item['sidecar_sha256'] and sha(paths['holds'])==item['holds_sha256'],'shard artifact hash')
  pred=rows(paths['prediction']);side=rows(paths['sidecar']);held=rows(paths['holds']);req((len(pred),len(side),len(held))==(item['prediction_inputs'],item['prediction_inputs'],item['holds']),'shard row count')
  req([x.get('row_id')for x in pred]==[x.get('row_id')for x in side],'sidecar order/join')
  for row in pred:
   rid=row.get('row_id');req(isinstance(rid,str)and rid not in prediction_by_id,'prediction identity');target_free(row);req(row.get('schema')in('sepalith.dat10.sourcewalk-noop.prediction_input.v1','sepalith.dat10.sourcewalk-noop.prediction_input.v2'),'prediction schema');prediction_by_id[rid]=row
  hold_total+=len(held);entries.append({'shard':shard,'candidate_rows':item['candidates'],'rows':len(pred),'upstream_holds':len(held),'path':paths['prediction'].name,'sha256':sha(paths['prediction']),'bytes':paths['prediction'].stat().st_size})
 req(len(prediction_by_id)==4100 and hold_total==127,'global 4100/127 closure')
 duplicate=manifest['duplicate_geometry'];duplicate_path=input_root/duplicate['artifact'];req(duplicate['policy']=='retain_all_until_provider_prompt_target_dedup'and sha(duplicate_path)==duplicate['sha256'],'duplicate geometry pin')
 members=[]
 for group in rows(duplicate_path):
  ids=group['row_ids'];req(group['disposition']=='retain_all_until_provider_prompt_target_dedup'and len(ids)==group['count']>=2,'duplicate group');members+=ids
  for rid in ids:
   row=prediction_by_id.get(rid);g=group['geometry'];req(row and row['preedit_sha256']==g['preedit_sha256']and row['cursor']==g['cursor']and row['path']==g['path'],'duplicate member')
 req(len(members)==len(set(members))==duplicate['rows_in_groups'],'duplicate accounting')
 lane_rows=[0,0]
 for item in sorted((x for x in entries if x['rows']),key=lambda x:(-x['rows'],x['shard'])):
  lane=0 if lane_rows[0]<=lane_rows[1]else 1;item['lane']=lane;item['core']=(4,6)[lane];lane_rows[lane]+=item['rows']
 for item in entries:
  if not item['rows']:item['lane']=None;item['core']=None
 result={'schema':'sepalith.dat10.noop4100.provider-run-plan.v1','status':'prepared_no_launch','phase':'render16','context_size':16384,'generation_reserve':2048,'source_candidates':4227,'rows':4100,'upstream_holds':127,'upstream_hold_classes':{'provenance':121,'mixed_eol_geometry':6},'shards':41,'nonempty_shards':sum(x['rows']>0 for x in entries),'input_root':str(input_root.resolve()),'input_manifest_sha256':manifest_sha,'reconstruction_source_manifest_sha256':RECONSTRUCTION_SOURCE_SHA,'duplicate_geometry_sha256':duplicate['sha256'],'duplicate_geometry_groups':duplicate['groups'],'duplicate_geometry_rows':duplicate['rows_in_groups'],'coverage_sha256':sha(coverage_path),'candidate_ids_sha256':sha(candidates_path),'provider_source_manifest_sha256':PROVIDER_SOURCE_SHA,'run_shard_sha256':RUN_SHARD_SHA,'provider_runtime_sha256':RUNTIME_FILES,'tokenizer_sha256':TOKENIZER_SHA,'lane_rows':lane_rows,'entries':entries,'prediction_inputs_target_free':True,'fixed_training_target':'NO_EDIT','execution_authorized':False,'training_admission':False}
 write(output,result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--input-manifest-sha256',required=True);p.add_argument('--coverage',required=True);p.add_argument('--candidate-ids',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.inputs,a.input_manifest_sha256,a.coverage,a.candidate_ids,a.output),sort_keys=True))
