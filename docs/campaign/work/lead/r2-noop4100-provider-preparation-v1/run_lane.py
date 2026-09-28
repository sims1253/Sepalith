#!/usr/bin/env python3
"""Execute one root-authorized lane from a hash-bound provider plan."""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
PROVIDER=Path(__file__).resolve().parent;SOURCE_SHA='6e6dfd87116bb8743fa1d593c857c7e2108a4d5e87f3a0814ff36e998b3cd424';RUN_SHA='9fd8d8ef6730871ffa1a9703e3af4b92dd4e265c07976fa547803e23e6c3e3fb'
RUNTIME_FILES={'render_shard.ts':'f5c5e57e9f33cb6503586ee51c9aa0fd1e38d017f6f74edef8b4e45423e9c2a6','source/namespace_runtime.ts':'e11868caa0d96a216f399d7772098d89ac415fe4db36ba69ea9e3c0d96140b12','source/source_imports.R':'3f140b69f9ff75d48d3305a5f27d6d2b8d8a49a4f97eaa9b0a2effe963801925'}
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def row_count(path):
 with Path(path).open() as stream:return sum(1 for _ in stream)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--lane',type=int,choices=(0,1),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();req(sha(a.plan)==a.plan_sha256,'plan hash differs');plan=json.loads(a.plan.read_text());req(plan['schema']=='sepalith.dat10.noop4100.provider-run-plan.v1' and plan['status']=='prepared_no_launch','provider plan differs');req(plan['provider_source_manifest_sha256']==sha(PROVIDER/'provider-runtime-manifest.json')==SOURCE_SHA and plan['run_shard_sha256']==sha(PROVIDER/'run_shard.sh')==RUN_SHA and plan.get('provider_runtime_sha256')==RUNTIME_FILES and plan.get('tokenizer_sha256')==TOKENIZER_SHA and all(sha(PROVIDER/path)==digest for path,digest in RUNTIME_FILES.items()) and sha(TOKENIZER)==TOKENIZER_SHA,'provider source differs');a.output.mkdir(parents=True,exist_ok=True);terminal=a.output/f'lane-{a.lane}.terminal.json';req(not terminal.exists(),'lane terminal already exists');done=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for item in plan['entries']:
  if item['lane']!=a.lane:continue
  req(item['core']==(4,6)[a.lane] and Path(item['path']).name==item['path'],'lane core or shard path differs');source=Path(plan['input_root'])/item['path'];req(source.is_file() and sha(source)==item['sha256'] and row_count(source)==item['rows'],'planned shard input differs');command=['bash',str(PROVIDER/'run_shard.sh'),f"{item['shard']:04d}",str(item['core']),str(plan['context_size']),str(plan['generation_reserve']),plan['input_root'],str(a.output)];result=subprocess.run(command,env=env);req(result.returncode==0,f"shard {item['shard']} failed");output=a.output/f"shard-{item['shard']:04d}.jsonl";record=json.loads((a.output/f"shard-{item['shard']:04d}.terminal.json").read_text());req(record['status']=='complete' and record['shard']==item['shard'] and record['core']==item['core'] and record['context_size']==plan['context_size'] and record['generation_reserve']==plan['generation_reserve'] and record['input']['sha256']==item['sha256'] and record['output']['rows']==item['rows'] and record['output']['sha256']==sha(output),'provider shard terminal differs');done.append(item['shard'])
 value={'schema':'sepalith.dat10.noop4100.provider-lane-terminal.v1','status':'complete','phase':plan['phase'],'lane':a.lane,'core':(4,6)[a.lane],'shards':done,'rows':sum(x['rows'] for x in plan['entries'] if x['lane']==a.lane),'plan_sha256':a.plan_sha256}
 with terminal.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
