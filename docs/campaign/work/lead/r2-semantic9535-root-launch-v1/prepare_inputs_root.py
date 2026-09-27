#!/usr/bin/env python3
"""Prepare exact target-free provider inputs for reviewed semantic shards 27-40."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

HERE=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic9535-provider-preparation-v1')
ROOT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards27to40-v1')
REVIEW=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic27to40-terminal-review-v1/review-result.json')
REVIEW_SHA='8bbb1e9c635deebe370ad638c642b5aca1db1dc2a280502a299e605ab7183b1a'
QUEUE_SHA='6e63cf554caf26298d99ba1fd76797eed3d1daa3fe205c965788abb5ed4a06a8'
SUPPORTED='semantic_supported_context_closure_root_review_required'
SHARDS=list(range(27,41)); DENOMINATOR=9535

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_rows(p):
 out={}
 with Path(p).open() as f:
  for line in f:
   x=json.loads(line);rid=x['row_id']
   if rid in out:raise ValueError('duplicate:'+rid)
   out[rid]=x
 return out
def write_rows(p,values):
 with p.open('x') as f:
  for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 return {'path':p.name,'rows':len(values),'bytes':p.stat().st_size,'sha256':sha(p)}
def no_target_keys(value):
 if isinstance(value,dict):
  for key,item in value.items():
   if any(word in key.lower() for word in ('target','gold','completion','reward')):return False
   if not no_target_keys(item):return False
 if isinstance(value,list):return all(no_target_keys(x) for x in value)
 return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh_output_required')
 if sha(REVIEW)!=REVIEW_SHA or sha(ROOT/'streaming-manifest.json')!=QUEUE_SHA:raise ValueError('review_or_queue_pin')
 review=json.loads(REVIEW.read_text());queue=json.loads((ROOT/'streaming-manifest.json').read_text())
 if review['status']!='verified_partial_review_only_not_training_admitted' or review['semantic_rows']!=24330 or review['status_counts'].get(SUPPORTED)!=DENOMINATOR:raise ValueError('review_contract')
 if queue['requested_shards']!=SHARDS or queue['training_admission'] is not False or queue['status']!='partial_review_only':raise ValueError('queue_contract')
 tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent));entries=[];all_ids=set()
 try:
  (tmp/'sidecars').mkdir();(tmp/'preparation-holds').mkdir();(tmp/'base-reports').mkdir()
  by={x['shard']:x for x in review['shards']}
  for shard in SHARDS:
   bound=by[shard];folder=ROOT/f'shard-{shard:04d}';manifest=folder/'manifest.json';sem=folder/'semantic-ledger.jsonl';m=json.loads(manifest.read_text());binding=m['streaming_binding'];base=tmp/f'.base-{shard:04d}'
   packet=binding['candidate_packet'];prov=binding['provenance_ledger']
   command=['python3','-B',str(HERE/'prepare_shard_base.py'),'--semantic-manifest',str(manifest),'--expected-semantic-manifest-sha256',bound['manifest']['sha256'],'--semantic-ledger',str(sem),'--expected-semantic-ledger-sha256',bound['semantic_output']['sha256'],'--provenance-ledger',prov['path'],'--expected-provenance-ledger-sha256',prov['sha256'],'--candidate-packets',packet['path'],'--expected-candidate-packets-sha256',packet['sha256'],'--output',str(base)]
   env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
   subprocess.run(command,check=True,env=env)
   report=json.loads((base/'preparation-manifest.json').read_text());pred=load_rows(base/'prediction-inputs.jsonl');side=load_rows(base/'training-sidecar.jsonl')
   if set(pred)!=set(side) or report['semantic_supported']!=m['status_counts'].get(SUPPORTED,0) or len(pred)+report['preparation_holds']!=report['semantic_supported']:raise ValueError('prepared_closure:'+str(shard))
   rendered=[]
   for rid in sorted(pred):
    x=pred[rid];identity=side[rid]['identity'];source=Path(identity['source_path']);expected=x['external_import_dependencies']
    value={**x,'absolute_document_path':str(source),'workspace_root':str(source.parent.parent),'expected_dependencies':expected,'source_sha256':identity['source_sha256'],'package_id':identity['package_id'],'group_id':identity['group_id']}
    value.pop('external_import_dependencies',None)
    if not no_target_keys(value):raise ValueError('target_shaped_provider_input:'+rid)
    if hashlib.sha256(value['preedit_text'].encode()).hexdigest()!=value['preedit_sha256']:raise ValueError('preedit_hash:'+rid)
    rendered.append(value);all_ids.add(rid)
   provider=tmp/f'shard-{shard:04d}.jsonl';sideout=tmp/'sidecars'/f'shard-{shard:04d}.jsonl';holdout=tmp/'preparation-holds'/f'shard-{shard:04d}.jsonl'
   provider_pin=write_rows(provider,rendered);side_pin=write_rows(sideout,[side[x] for x in sorted(side)])
   shutil.copy2(base/'preparation-holds.jsonl',holdout);hold_pin={'path':str(holdout.relative_to(tmp)),'rows':report['preparation_holds'],'bytes':holdout.stat().st_size,'sha256':sha(holdout)}
   shutil.copy2(base/'preparation-manifest.json',tmp/'base-reports'/f'shard-{shard:04d}.json')
   shutil.rmtree(base)
   entries.append({'shard':shard,'semantic_rows':m['rows'],'upstream_semantic_holds':m['status_counts'].get('hold_semantic_evidence',0),'supported_denominator':report['semantic_supported'],'provider_rows':len(rendered),'preparation_holds':report['preparation_holds'],'provider':provider_pin,'sidecar':{**side_pin,'path':str(sideout.relative_to(tmp))},'preparation_hold':hold_pin,'source_manifest_sha256':bound['manifest']['sha256'],'semantic_output_sha256':bound['semantic_output']['sha256'],'provenance_sha256':prov['sha256'],'candidate_packet_sha256':packet['sha256']})
  if sum(x['supported_denominator'] for x in entries)!=DENOMINATOR or sum(x['provider_rows']+x['preparation_holds'] for x in entries)!=DENOMINATOR or len(all_ids)!=sum(x['provider_rows'] for x in entries):raise ValueError('global_denominator')
  manifest_out={'schema':'sepalith.dat10.semantic9535.provider_inputs.v1','status':'complete_target_free_review_only','training_admission':False,'upstream':{'streaming_manifest_sha256':QUEUE_SHA,'terminal_review_sha256':REVIEW_SHA,'provenance_rows':28753,'semantic_rows':24330,'semantic_supported':9535,'semantic_evidence_holds_preserved':14795,'provenance_outside_semantic_preserved':4423},'provider_rows':len(all_ids),'geometry_preparation_holds':sum(x['preparation_holds'] for x in entries),'exact_supported_denominator_closure':len(all_ids)+sum(x['preparation_holds'] for x in entries)==DENOMINATOR,'ids_sha256':hashlib.sha256(('\n'.join(sorted(all_ids))+'\n').encode()).hexdigest(),'target_or_gold_copied_to_provider_inputs':False,'shards':entries}
  with (tmp/'manifest.json').open('x') as f:json.dump(manifest_out,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,a.output);fd=os.open(a.output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  print(json.dumps(manifest_out,sort_keys=True))
 except BaseException:
  shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
