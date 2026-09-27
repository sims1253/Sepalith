#!/usr/bin/env python3
import hashlib,json,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
def terminal(p,source,output):p.write_text(json.dumps({'status':'complete','input':{'sha256':sha(source)},'output':{'sha256':sha(output)}})+'\n')
with tempfile.TemporaryDirectory(prefix='semantic9535-policy-',dir='/mnt/e/sepalith/campaign-20260915/data-work') as td:
 root=Path(td);inputs=root/'inputs';r16=root/'r16';fallback=root/'fallback';r32=root/'r32';selected=root/'selected';entries=[];cursor=0
 for shard,count in ((27,3),(28,2),(40,1)):
  values=[{'row_id':f'r{i}'} for i in range(cursor,cursor+count)];cursor+=count;p=inputs/f'shard-{shard:04d}.jsonl';write(p,values);entries.append({'shard':shard,'provider_rows':count,'provider':{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'rows':count}})
  out=r16/p.name;write(out,[{'row_id':x['row_id'],'status':'supported' if x['row_id'] not in ('r1','r5') else 'hold','reasons':['complete_span_context_budget']} for x in values]);terminal(r16/f'shard-{shard:04d}.terminal.json',p,out)
 inputs.mkdir(exist_ok=True);(inputs/'manifest.json').write_text(json.dumps({'status':'complete_target_free_review_only','training_admission':False,'provider_rows':6,'shards':entries})+'\n')
 subprocess.run(['python3',str(HERE/'prepare_fallback.py'),'--inputs',str(inputs),'--render16',str(r16),'--output',str(fallback)],check=True)
 fm=json.loads((fallback/'manifest.json').read_text());assert fm['denominator']==6 and fm['rerun32_rows']==2 and fm['training_admission'] is False
 for x in fm['shards']:
  source=fallback/x['path'];vals=[json.loads(y) for y in source.read_text().splitlines()];out=r32/source.name;write(out,[{'row_id':y['row_id'],'status':'supported','reasons':[]} for y in vals]);terminal(r32/f"shard-{x['shard']:04d}.terminal.json",source,out)
 subprocess.run(['python3',str(HERE/'finalize_policy.py'),'--inputs',str(inputs),'--render16',str(r16),'--fallback_inputs',str(fallback),'--render32',str(r32),'--output',str(selected)],check=True,capture_output=True,text=True)
 sm=json.loads((selected/'manifest.json').read_text());assert sm['denominator']==6 and sm['supported']==6 and sm['training_admission'] is False
 (r32/'shard-0028.jsonl').write_text((r32/'shard-0028.jsonl').read_text()+'{}\n')
 bad=subprocess.run(['python3',str(HERE/'finalize_policy.py'),'--inputs',str(inputs),'--render16',str(r16),'--fallback_inputs',str(fallback),'--render32',str(r32),'--output',str(root/'bad')],capture_output=True,text=True);assert bad.returncode!=0
print('PASS variable denominator, 16K->32K fallback, exact joins, corrupt output rejection')
