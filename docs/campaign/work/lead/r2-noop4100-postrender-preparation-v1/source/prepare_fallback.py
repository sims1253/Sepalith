#!/usr/bin/env python3
"""Create target-free 32K fallback inputs from complete 16K results."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def keyed(path):
 out={}
 with Path(path).open() as stream:
  for line in stream:
   x=json.loads(line);req(x['row_id'] not in out,'duplicate row');out[x['row_id']]=x
 return out
def write_rows(path,values):
 with path.open('x') as stream:
  for x in values:stream.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  stream.flush();os.fsync(stream.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--render16',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();req(sha(a.plan)==a.plan_sha256,'plan hash differs');plan=json.loads(a.plan.read_text());req(plan['phase']=='render16' and plan['rows']==4100 and plan['upstream_holds']==127,'16K plan differs');req(not a.output.exists(),'fresh output required');a.output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent));entries=[];all_ids=set();hold_ids=set()
 try:
  for item in plan['entries']:
   shard=item['shard'];values=[]
   if item['rows']:
    source=Path(plan['input_root'])/item['path'];result=a.render16/f'shard-{shard:04d}.jsonl';terminal=a.render16/f'shard-{shard:04d}.terminal.json';t=json.loads(terminal.read_text());req(t.get('schema')=='sepalith.dat10.noop4100.parse_retry.render_shard_terminal.v1' and t['status']=='complete' and t['context_size']==16384 and t['generation_reserve']==2048 and t['input']['sha256']==item['sha256'] and t['output']['sha256']==sha(result),'16K terminal differs');src=keyed(source);render=keyed(result);req(set(src)==set(render) and len(src)==item['rows'] and all(x.get('status')in ('supported','hold') for x in render.values()),'16K ID/status closure differs');all_ids.update(src);values=[src[rid] for rid in sorted(src) if render[rid]['status']=='hold'];hold_ids.update(x['row_id'] for x in values)
   path=tmp/f'shard-{shard:04d}.jsonl';write_rows(path,values);entries.append({'shard':shard,'candidate_rows':item['candidate_rows'],'rows':len(values),'upstream_holds':item['upstream_holds'],'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
  req(len(all_ids)==4100,'16K provider denominator differs');lane_rows=[0,0]
  for item in sorted((x for x in entries if x['rows']),key=lambda x:(-x['rows'],x['shard'])):
   lane=0 if lane_rows[0]<=lane_rows[1] else 1;item['lane']=lane;item['core']=(4,6)[lane];lane_rows[lane]+=item['rows']
  for item in entries:
   if not item['rows']:item['lane']=None;item['core']=None
  manifest={'schema':'sepalith.dat10.noop4100.provider-run-plan.v1','status':'prepared_no_launch','phase':'render32','context_size':32768,'generation_reserve':2048,'source_candidates':4227,'rows':len(hold_ids),'provider_denominator':4100,'upstream_holds':127,'upstream_hold_classes':plan['upstream_hold_classes'],'shards':41,'nonempty_shards':sum(x['rows']>0 for x in entries),'input_root':str(a.output.resolve()),'source_plan_sha256':a.plan_sha256,'render16_source_manifest_sha256':'114b0a410cf15c821ce83c8d569b16d831640f744f3e53d303d849929e503919','provider_source_manifest_sha256':'9725d968382c98fe5313985f5407fd6aa01a3e34063393b88c3bfb45193572e8','run_shard_sha256':'839818844ee35dbf43fcaf99af544dda4e40f79d71917a192976aff5e675d668','provider_runtime_sha256':plan['provider_runtime_sha256'],'tokenizer_sha256':plan['tokenizer_sha256'],'lane_rows':lane_rows,'entries':entries,'prediction_inputs_target_free':True,'fixed_training_target':'NO_EDIT','execution_authorized':False,'training_admission':False};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(tmp,a.output);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
