#!/usr/bin/env python3
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1')
INPUT=ROOT/'input-shards';RENDER=ROOT/'render-04';OUT=ROOT/'render-16k-input'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path):
 out={}
 for line in path.read_text().splitlines():
  x=json.loads(line);rid=x['row_id']
  if rid in out:raise ValueError('duplicate:'+rid)
  out[rid]=x
 return out
inputs={}
for shard in ('0000','0001'):inputs.update(load(INPUT/f'shard-{shard}.jsonl'))
if len(inputs)!=4551:raise ValueError('input denominator')
rendered={};terminal=[]
for shard in ('0000','0001'):
 t=json.loads((RENDER/f'shard-{shard}.terminal.json').read_text());p=RENDER/f'shard-{shard}.jsonl'
 if t['status']!='complete' or t['output']['sha256']!=sha(p) or t['output']['rows']!=len(load(p)):raise ValueError('terminal binding:'+shard)
 terminal.append(t);rendered.update(load(p))
if set(rendered)!=set(inputs):raise ValueError('render/input closure')
rerun=[];reuse=[];holds=[]
for rid in sorted(inputs):
 r=rendered[rid]
 if r['status']!='supported':holds.append(rid)
 elif r['mode']=='full_document' and r['prompt_tokens']+1+1024<=16384:reuse.append(rid)
 else:rerun.append(inputs[rid])
if OUT.exists():raise ValueError('fresh 16k input required')
OUT.mkdir(parents=True);entries=[]
for shard in range(2):
 values=[x for i,x in enumerate(rerun) if i%2==shard];p=OUT/f'shard-{shard:04d}.jsonl'
 with p.open('x') as f:
  for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 entries.append({'shard':shard,'rows':len(values),'sha256':sha(p),'bytes':p.stat().st_size})
m={'schema':'sepalith.dat10.semantic4551.fallback16_inputs.v1','status':'complete_target_free','denominator':4551,'reuse_16k':len(reuse),'rerun_16k':len(rerun),'holds_32k':len(holds),'reuse_ids_sha256':hashlib.sha256(('\n'.join(reuse)+'\n').encode()).hexdigest(),'hold_ids':holds,'shards':entries,'render32_terminals_sha256':[sha(RENDER/f'shard-{s}.terminal.json') for s in ('0000','0001')],'target_used':False}
(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(json.dumps(m,sort_keys=True))
