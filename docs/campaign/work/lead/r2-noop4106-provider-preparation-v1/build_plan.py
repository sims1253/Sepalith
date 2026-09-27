#!/usr/bin/env python3
"""Bind completed no-op reconstruction to two provider lanes."""
import argparse,collections,hashlib,json,os,tempfile
from pathlib import Path

PROVIDER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2')
PROVIDER_SOURCE_SHA='7546d5457c299edfbf69bc0681327f08bad5041e2a146fe60fd66278dd58c770'
RUN_SHARD_SHA='2d9dcf249571ae66c75b10bec732162e99b6d53cc5c969186f0e52292593f309'
RECONSTRUCTION_SOURCE=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-sourcewalk-noop-expansion-preparation-v3/source-manifest.json');RECONSTRUCTION_SOURCE_SHA='b7b03e63f533c8035d896352dc1416d4304b56744d8683a53b2b443fb734a842'
COVERAGE_SHA='8c11cf68cf1d2a91dd92cbe29327b887fe0efc46943d8302760b5f3a80c16df8';CANDIDATES_SHA='59e4d0b57fa2185503e1ab4a1bad1b6e43acf4bc70e239508b800204a31a7ca7'
RUNTIME_FILES={'render_shard.ts':'352a2976b235d5604af9fe2a931208fda53728a53d3cf027583aa4e237d2fcd5','tokenize_bridge.py':'57ef8859a2e9294acd8f1abb56def91b1bf2fd864eb64394dc89e016051e9db9','source/namespace_evidence.R':'97a56820bb237b4d6e40216f932b2514a306985c0b67cbb3afe6b91b22bf1847'}
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def req(value,message):
 if not value:raise ValueError(message)
def rows(path):
 with Path(path).open() as stream:return [json.loads(line) for line in stream]
def row_count(path):
 with Path(path).open() as stream:return sum(1 for _ in stream)
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def build(input_root,coverage_path,candidates_path,output):
 input_root=Path(input_root);manifest_path=input_root/'manifest.json';manifest=json.loads(manifest_path.read_text());coverage=json.loads(Path(coverage_path).read_text());candidates=rows(candidates_path)
 req(sha(RECONSTRUCTION_SOURCE)==RECONSTRUCTION_SOURCE_SHA and sha(coverage_path)==COVERAGE_SHA and sha(candidates_path)==CANDIDATES_SHA,'reconstruction source or census binding differs')
 req(sha(PROVIDER/'source-manifest.json')==PROVIDER_SOURCE_SHA and sha(PROVIDER/'run_shard.sh')==RUN_SHARD_SHA and all(sha(PROVIDER/path)==digest for path,digest in RUNTIME_FILES.items()) and sha(TOKENIZER)==TOKENIZER_SHA,'frozen provider runtime differs')
 req(manifest.get('schema')=='sepalith.dat10.sourcewalk-noop-expansion-preparation.v3' and manifest.get('status')=='complete_review_only' and manifest.get('full_41_shard_closure') is True,'reconstruction is not complete full41 v3')
 req((manifest.get('candidate_rows'),manifest.get('prediction_inputs'),manifest.get('holds'))==(4227,4106,121),'reconstruction denominator differs')
 req(coverage.get('completed_shards')==41 and coverage.get('pending_shards')==[] and coverage.get('partial') is False,'coverage is not complete41')
 req(len(candidates)==len({x['row_id'] for x in candidates})==4227,'candidate IDs differ')
 counts=collections.defaultdict(collections.Counter)
 for item in candidates:counts[item['shard']][item['status']]+=1
 bound={x['shard']:x for x in manifest['shards']};req(len(bound)==len(manifest['shards']),'duplicate reconstruction shard')
 entries=[];prediction_by_id={}
 for shard in range(41):
  total=sum(counts[shard].values());supported=counts[shard]['provenance_supported_candidate_root_review_required'];holds=counts[shard]['hold_independent_provenance_failure'];item=bound.get(shard)
  if total==0:
   req(item is None and not (input_root/f'shard-{shard:04d}.jsonl').exists(),'empty shard unexpectedly materialized');entries.append({'shard':shard,'candidate_rows':0,'rows':0,'upstream_holds':0,'path':None,'sha256':hashlib.sha256(b'').hexdigest(),'bytes':0});continue
  req(item and (item['candidates'],item['prediction_inputs'],item['holds'])==(total,supported,holds),'reconstruction shard count differs')
  path=input_root/f'shard-{shard:04d}.jsonl';side=input_root/f'shard-{shard:04d}.sidecar.jsonl';held=input_root/f'shard-{shard:04d}.holds.jsonl'
  req(path.is_file() and side.is_file() and held.is_file(),'reconstruction shard artifact missing')
  req(sha(path)==item['prediction_sha256'] and sha(side)==item['sidecar_sha256'] and sha(held)==item['holds_sha256'],'reconstruction shard hash differs')
  prediction_rows=rows(path);req(len(prediction_rows)==supported and row_count(side)==supported and row_count(held)==holds,'reconstruction shard rows differ')
  for prediction in prediction_rows:
   rid=prediction.get('row_id');req(isinstance(rid,str) and rid not in prediction_by_id,'duplicate reconstruction row ID');prediction_by_id[rid]=prediction
  entries.append({'shard':shard,'candidate_rows':total,'rows':supported,'upstream_holds':holds,'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
 req(sum(x['rows'] for x in entries)==4106 and sum(x['upstream_holds'] for x in entries)==121 and len(prediction_by_id)==4106,'global provider denominator differs')
 duplicate=manifest.get('duplicate_geometry',{});duplicate_path=input_root/duplicate.get('artifact','')
 req(duplicate.get('policy')=='retain_all_until_provider_prompt_target_dedup' and duplicate_path.is_file() and sha(duplicate_path)==duplicate.get('sha256'),'duplicate geometry evidence differs')
 duplicate_rows=rows(duplicate_path);member_ids=[]
 for group in duplicate_rows:
  ids=group.get('row_ids');req(group.get('disposition')=='retain_all_until_provider_prompt_target_dedup' and isinstance(ids,list) and len(ids)==group.get('count') and len(ids)>=2 and len(ids)==len(set(ids)),'duplicate geometry group invalid');member_ids+=ids
  for rid in ids:
   prediction=prediction_by_id.get(rid);req(prediction is not None,'duplicate geometry member absent');geometry=group['geometry'];req(prediction['preedit_sha256']==geometry['preedit_sha256'] and prediction['cursor']==geometry['cursor'] and prediction['path']==geometry['path'],'duplicate geometry member mismatch')
 req(len(member_ids)==len(set(member_ids))==duplicate.get('rows_in_groups',0) and len(duplicate_rows)==duplicate.get('groups',0) and sum(x['count']-1 for x in duplicate_rows)==duplicate.get('excess_rows',0),'duplicate geometry accounting differs')
 lane_rows=[0,0]
 for item in sorted((x for x in entries if x['rows']),key=lambda x:(-x['rows'],x['shard'])):
  lane=0 if lane_rows[0]<=lane_rows[1] else 1;item['lane']=lane;item['core']=(4,6)[lane];lane_rows[lane]+=item['rows']
 for item in entries:
  if not item['rows']:item['lane']=None;item['core']=None
 result={'schema':'sepalith.dat10.noop4106.provider-run-plan.v1','status':'prepared_no_launch','phase':'render16','context_size':16384,'generation_reserve':2048,'source_candidates':4227,'rows':4106,'upstream_holds':121,'shards':41,'nonempty_shards':sum(x['rows']>0 for x in entries),'input_root':str(input_root.resolve()),'input_manifest_sha256':sha(manifest_path),'reconstruction_source_manifest_sha256':RECONSTRUCTION_SOURCE_SHA,'duplicate_geometry_sha256':duplicate['sha256'],'duplicate_geometry_groups':duplicate['groups'],'duplicate_geometry_rows':duplicate['rows_in_groups'],'coverage_sha256':sha(coverage_path),'candidate_ids_sha256':sha(candidates_path),'provider_source_manifest_sha256':PROVIDER_SOURCE_SHA,'run_shard_sha256':RUN_SHARD_SHA,'provider_runtime_sha256':RUNTIME_FILES,'tokenizer_sha256':TOKENIZER_SHA,'lane_rows':lane_rows,'entries':entries,'target_or_gold_used':False,'execution_authorized':False,'training_admission':False}
 write(output,result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--coverage',required=True);p.add_argument('--candidate-ids',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build(a.inputs,a.coverage,a.candidate_ids,a.output),sort_keys=True))
