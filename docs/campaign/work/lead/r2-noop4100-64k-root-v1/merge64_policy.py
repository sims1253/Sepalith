"""Merge a root-reviewed 64K render into the 4,100-row context policy."""
import argparse,hashlib,json,os
from pathlib import Path
POLICY=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/selected-policy-postrender-v2')
POLICY_MANIFEST_SHA='f3563a42c2a42ea7cb6125eab8ea3dfe16294cf587d940834f3af081df0759ca'
SELECTED_SHA='be797af50a32d76820ab0b5377e8d0c116e8bf9d9ac40189128c886c4fd58c5b'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadlines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f]
def merge(queue_root,render_root,output):
 queue_root=Path(queue_root);render_root=Path(render_root);output=Path(output);assert not output.exists()
 qm=json.loads((queue_root/'manifest.json').read_text());assert qm['rows']==38 and qm['closure']=='3995+38+67=4100'
 queued={json.loads(x)['row_id'] for x in (queue_root/'queued-ids.jsonl').read_text().splitlines()};assert len(queued)==38
 rendered=[]
 for item in qm['entries']:
  if item['rows']:
   p=render_root/f"shard-{item['shard']:04d}.jsonl";terminal=render_root/f"shard-{item['shard']:04d}.terminal.json";t=json.loads(terminal.read_text())
   assert t['schema']=='sepalith.dat10.noop4100.fallback32.render_shard_terminal.v1' and t['status']=='complete' and t['exit_code']==0
   assert t['context_size']==65536 and t['generation_reserve']==2048 and t['input']['sha256']==item['sha256'] and t['output']['sha256']==sha(p) and t['output']['rows']==item['rows']
   rendered.extend(loadlines(p))
 assert len(rendered)==38 and {x['row_id'] for x in rendered}==queued and all(x['status'] in {'supported','hold'} for x in rendered)
 by_id={x['row_id']:x for x in rendered}
 assert sha(POLICY/'manifest.json')==POLICY_MANIFEST_SHA and sha(POLICY/'selected-contexts.jsonl')==SELECTED_SHA
 prior=loadlines(POLICY/'selected-contexts.jsonl');assert len(prior)==4100 and len({x['row_id'] for x in prior})==4100
 merged=[]
 for row in prior:
  if row['row_id'] in queued:
   row=by_id[row['row_id']];row['policy_resolution']='selected_64k' if row['status']=='supported' else 'hold_after_64k'
  merged.append(row)
 assert len(merged)==4100
 output.mkdir(parents=True);selected=output/'selected-contexts.jsonl'
 with selected.open('x') as f:
  for row in merged:f.write(json.dumps(row,sort_keys=True)+'\n')
 supported=sum(x['status']=='supported' for x in merged);holds=4100-supported
 manifest={'schema':'sepalith.dat10.noop4100.context-policy.v2','status':'complete_review_only','contexts':[16384,32768,65536],'provider_denominator':4100,'provider_supported':supported,'provider_holds_after_64k':holds,'prior_supported':3995,'retry_rows':38,'retry_supported':sum(x['status']=='supported' for x in rendered),'retry_holds':sum(x['status']=='hold' for x in rendered),'selected':{'path':'selected-contexts.jsonl','rows':4100,'bytes':selected.stat().st_size,'sha256':sha(selected)},'queue_manifest_sha256':sha(queue_root/'manifest.json'),'prior_policy_manifest_sha256':POLICY_MANIFEST_SHA,'provider_source_manifest_sha256':qm['provider_source_manifest_sha256'],'target_or_gold_used_for_selection':False,'training_admission':False}
 prior_manifest=json.loads((POLICY/'manifest.json').read_text());manifest={**prior_manifest,**manifest};manifest['counts']={key:sum(x.get('policy_resolution')==key for x in merged) for key in sorted({x.get('policy_resolution') for x in merged})};manifest['generation_reserve']=2048
 with (output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--render',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(merge(a.queue,a.render,a.output),sort_keys=True))
