#!/usr/bin/env python3
import hashlib,json,os,shutil,tempfile
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');SRC=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10952-preparation-v1');OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/render-inputs-v1')
PINS={'terminal':'0f2e2d321dce5ba4355a1eb7b304e97a2bfc093da27e46106dda3d41d0a63f78','root_receipt':'556cf6149900dd5ef3a2274ee6a9e243c151f4334fb88cc7bd8789da48047619'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path,key='row_id'):
 out={}
 for line in path.open():
  x=json.loads(line);rid=x[key]
  if rid in out:raise ValueError('duplicate:'+rid)
  out[rid]=x
 return out
if sha(SRC/'terminal.json')!=PINS['terminal'] or sha(PLAN/'docs/campaign/receipts/DAT-10-semantic10952-root-review.json')!=PINS['root_receipt']:raise ValueError('root input pin')
terminal=json.loads((SRC/'terminal.json').read_text());assert terminal['status']=='complete' and terminal['prepared_rows']==10948 and terminal['hold_rows']==4 and terminal['exact_denominator_closure'] is True
if OUT.exists():raise ValueError('fresh output required')
tmp=Path(tempfile.mkdtemp(prefix='.render-inputs-v1.',dir=OUT.parent));entries=[];all_ids=set()
try:
 for bound in terminal['shards']:
  shard=int(bound['shard']);assert 12<=shard<=26
  d=SRC/f'shard-{shard:04d}';manifest=d/'preparation-manifest.json'
  assert sha(manifest)==bound['preparation_manifest']['sha256'];m=json.loads(manifest.read_text());assert m['status']=='complete'
  pred=d/'prediction-inputs.jsonl';side=d/'training-sidecar.jsonl';assert sha(pred)==bound['outputs']['prediction-inputs.jsonl']['sha256'] and sha(side)==bound['outputs']['training-sidecar.jsonl']['sha256']
  identities={rid:x['identity'] for rid,x in rows(side).items()};prepared=[]
  for rid,x in rows(pred).items():
   if rid not in identities or rid in all_ids:raise ValueError('ID closure:'+rid)
   if any(any(word in key.lower() for word in ('target','gold','completion','reward')) for key in x):raise ValueError('target-shaped prediction key:'+rid)
   if hashlib.sha256(x['preedit_text'].encode()).hexdigest()!=x['preedit_sha256']:raise ValueError('preedit hash:'+rid)
   identity=identities[rid];source=Path(identity['source_path']);expected=x.pop('external_import_dependencies')
   prepared.append({**x,'absolute_document_path':str(source),'workspace_root':str(source.parent.parent),'expected_dependencies':expected,'source_sha256':identity['source_sha256'],'package_id':identity['package_id'],'group_id':identity['group_id']});all_ids.add(rid)
  assert set(identities)=={x['row_id'] for x in prepared} and len(prepared)==bound['prepared_rows']
  output=tmp/f'shard-{shard:04d}.jsonl'
  with output.open('x') as f:
   for x in sorted(prepared,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   f.flush();os.fsync(f.fileno())
  entries.append({'shard':shard,'rows':len(prepared),'path':output.name,'sha256':sha(output),'bytes':output.stat().st_size,'preparation_manifest_sha256':sha(manifest),'prediction_sha256':sha(pred),'sidecar_sha256':sha(side)})
 assert len(all_ids)==10948 and len(entries)==15
 manifest={'schema':'sepalith.dat10.semantic10948.render_inputs.v1','status':'complete_target_free','rows':10948,'shards':entries,'ids_sha256':hashlib.sha256(('\n'.join(sorted(all_ids))+'\n').encode()).hexdigest(),'source_terminal_sha256':PINS['terminal'],'root_review_receipt_sha256':PINS['root_receipt'],'upstream_geometry_holds':4,'target_or_gold_copied':False}
 with (tmp/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,OUT);fd=os.open(OUT.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(manifest,sort_keys=True))
except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
