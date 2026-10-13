#!/usr/bin/env python3
import argparse,collections,hashlib,json,os,shutil,tempfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 out={}
 for l in Path(p).open():
  x=json.loads(l);rid=x['row_id']
  if rid in out:raise ValueError('duplicate:'+rid)
  out[rid]=x
 return out
ap=argparse.ArgumentParser();
for name in ('inputs','render16','fallback_inputs','render32','output'):ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args();im=json.loads((a.inputs/'manifest.json').read_text());fm=json.loads((a.fallback_inputs/'manifest.json').read_text());assert im['rows']==10948 and fm['denominator']==10948
if a.output.exists():raise ValueError('fresh output required')
tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent));selected=[];counts=collections.Counter();followup={int(x['shard']):[] for x in im['shards']};all_ids=set()
try:
 for bound in im['shards']:
  shard=int(bound['shard']);source_path=a.inputs/bound['path'];fallback_path=a.fallback_inputs/f'shard-{shard:04d}.jsonl';render16_path=a.render16/f'shard-{shard:04d}.jsonl';render32_path=a.render32/f'shard-{shard:04d}.jsonl';t16=json.loads((a.render16/f'shard-{shard:04d}.terminal.json').read_text());t32=json.loads((a.render32/f'shard-{shard:04d}.terminal.json').read_text())
  if t16['status']!='complete' or t16['input']['sha256']!=sha(source_path) or t16['output']['sha256']!=sha(render16_path) or t32['status']!='complete' or t32['input']['sha256']!=sha(fallback_path) or t32['output']['sha256']!=sha(render32_path):raise ValueError('terminal binding:'+str(shard))
  source=rows(source_path);r16=rows(render16_path);r32=rows(render32_path);expected32=rows(fallback_path)
  if set(source)!=set(r16) or set(r32)!=set(expected32):raise ValueError('stage closure:'+str(shard))
  for rid in sorted(source):
   x16=r16[rid]
   if x16['status']=='supported':value={**x16,'policy_resolution':'selected_16k'};counts['selected_16k']+=1
   else:
    x32=r32[rid]
    if x32['status']=='supported':value={**x32,'policy_resolution':'fallback_supported_32k'};counts['fallback_supported_32k']+=1
    else:
     value={**x32,'policy_resolution':'hold_after_32k'};counts['hold_after_32k']+=1
     if 'complete_span_context_budget' in x32.get('reasons',[]):followup[shard].append(source[rid]);counts['context_only_followup_64k_128k']+=1
   selected.append(value);all_ids.add(rid)
 assert len(selected)==len(all_ids)==10948
 out=tmp/'selected-contexts.jsonl'
 with out.open('x') as f:
  for x in sorted(selected,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 follow=tmp/'followup-context-inputs';follow.mkdir();entries=[]
 for shard,values in sorted(followup.items()):
  q=follow/f'shard-{shard:04d}.jsonl'
  with q.open('x') as f:
   for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   f.flush();os.fsync(f.fileno())
  entries.append({'shard':shard,'rows':len(values),'sha256':sha(q),'bytes':q.stat().st_size})
 m={'schema':'sepalith.dat10.semantic10948.context_policy.v1','status':'complete_review_only','denominator':10948,'counts':dict(counts),'supported':sum(x['status']=='supported' for x in selected),'holds':sum(x['status']!='supported' for x in selected),'selected':{'path':'selected-contexts.jsonl','sha256':sha(out),'bytes':out.stat().st_size,'rows':10948},'followup_context_inputs':{'profiles':[65536,131072],'reason':'32K context budget only; no exclusion ceiling','shards':entries},'target_or_gold_used':False,'training_admission':False}
 with (tmp/'manifest.json').open('x') as f:json.dump(m,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.rename(tmp,a.output);fd=os.open(a.output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(m,sort_keys=True))
except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
