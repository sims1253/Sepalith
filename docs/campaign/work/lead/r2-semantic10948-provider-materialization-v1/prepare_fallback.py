#!/usr/bin/env python3
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 out={}
 for l in Path(p).open():
  x=json.loads(l);rid=x['row_id']
  if rid in out:raise ValueError('duplicate:'+rid)
  out[rid]=x
 return out
ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--render16',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();im=json.loads((a.inputs/'manifest.json').read_text());assert im['status']=='complete_target_free' and im['rows']==10948
if a.output.exists():raise ValueError('fresh output required')
tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent));entries=[];all_ids=set();hold_ids=set()
try:
 for bound in im['shards']:
  shard=int(bound['shard']);inp=a.inputs/bound['path'];terminal=json.loads((a.render16/f'shard-{shard:04d}.terminal.json').read_text());render=a.render16/f'shard-{shard:04d}.jsonl'
  if terminal['status']!='complete' or terminal['input']['sha256']!=sha(inp) or terminal['output']['sha256']!=sha(render):raise ValueError('terminal:'+str(shard))
  source=rows(inp);result=rows(render)
  if set(source)!=set(result) or len(result)!=bound['rows']:raise ValueError('shard closure:'+str(shard))
  held=[source[rid] for rid in sorted(source) if result[rid]['status']!='supported'];hold_ids.update(x['row_id'] for x in held);all_ids.update(source)
  out=tmp/f'shard-{shard:04d}.jsonl'
  with out.open('x') as f:
   for x in held:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   f.flush();os.fsync(f.fileno())
  entries.append({'shard':shard,'rows':len(held),'path':out.name,'sha256':sha(out),'bytes':out.stat().st_size,'render16_terminal_sha256':sha(a.render16/f'shard-{shard:04d}.terminal.json')})
 assert len(all_ids)==10948
 m={'schema':'sepalith.dat10.semantic10948.fallback32_inputs.v1','status':'complete_target_free','denominator':10948,'rerun32_rows':len(hold_ids),'shards':entries,'hold_ids_sha256':hashlib.sha256(('\n'.join(sorted(hold_ids))+('\n' if hold_ids else '')).encode()).hexdigest(),'source_input_manifest_sha256':sha(a.inputs/'manifest.json'),'target_or_gold_used':False}
 with (tmp/'manifest.json').open('x') as f:json.dump(m,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.rename(tmp,a.output);fd=os.open(a.output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(m,sort_keys=True))
except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
