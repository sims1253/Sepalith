#!/usr/bin/env python3
import argparse,hashlib,json,time
from pathlib import Path
PINS={'terminal':'0f2e2d321dce5ba4355a1eb7b304e97a2bfc093da27e46106dda3d41d0a63f78','root_receipt':'556cf6149900dd5ef3a2274ee6a9e243c151f4334fb88cc7bd8789da48047619','root_v4_preparer':'d8bd4a472dbe764081a20de57d829ce5de507eb01eb2be6ef9a02167043a6e5a','render_manifest':'89ccbe0226124fc64e460c34595d7adac1d94a976e5c455bdf3ffc963e830619'}
ALLOWED={'complete','complete_with_explicit_holds'}
BAD_KEYS=('target','gold','completion','reward')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read_rows(path):
 out={}
 with Path(path).open() as f:
  for line in f:
   x=json.loads(line);rid=x.get('row_id')
   if not isinstance(rid,str) or not rid or rid in out:raise ValueError('invalid_or_duplicate_id:'+str(rid))
   out[rid]=x
 return out
def verify_file(path,pin,label):
 if sha(path)!=pin['sha256'] or Path(path).stat().st_size!=pin['bytes']:raise ValueError('file_binding:'+label)
def validate_id_join(predictions,sidecars,held,rendered,label):
 if set(predictions)!=set(sidecars) or set(predictions)!=set(rendered) or set(predictions)&set(held):raise ValueError('id_join:'+label)
def assert_no_target_keys(value,path='row'):
 if isinstance(value,dict):
  for k,v in value.items():
   if any(x in k.lower() for x in BAD_KEYS):raise ValueError('target_shaped_key:'+path+'.'+k)
   assert_no_target_keys(v,path+'.'+k)
 elif isinstance(value,list):
  for i,v in enumerate(value):assert_no_target_keys(v,f'{path}[{i}]')
def validate_metadata(terminal,render_manifest,prep_manifests,expected_shards=range(12,27),expected_prepared=10948,expected_holds=4):
 shards=list(expected_shards);bound=terminal['shards']
 if terminal['status']!='prepared_not_training_admitted' or terminal['training_admission'] is not False:raise ValueError('terminal_status')
 if terminal['prepared_rows']!=expected_prepared or terminal['hold_rows']!=expected_holds or terminal['accepted_scope_rows']!=expected_prepared+expected_holds or terminal['exact_denominator_closure'] is not True:raise ValueError('terminal_denominator')
 if [int(x['shard']) for x in bound]!=shards or len({int(x['shard']) for x in bound})!=len(shards):raise ValueError('terminal_shard_set')
 if render_manifest['status']!='complete_target_free' or render_manifest['rows']!=expected_prepared or render_manifest['target_or_gold_copied'] is not False:raise ValueError('render_manifest_contract')
 if [int(x['shard']) for x in render_manifest['shards']]!=shards:raise ValueError('render_shard_set')
 prepared=holds=0
 for b,r in zip(bound,render_manifest['shards']):
  s=int(b['shard']);m=prep_manifests[s]
  if m['status'] not in ALLOWED:raise ValueError('preparation_status:'+str(s))
  if m['preparation_holds']==0 and m['status']!='complete':raise ValueError('status_hold_mismatch:'+str(s))
  if m['preparation_holds']>0 and m['status']!='complete_with_explicit_holds':raise ValueError('status_hold_mismatch:'+str(s))
  if m['prediction_inputs']!=m['training_sidecar_rows'] or m['semantic_supported']!=m['prediction_inputs']+m['preparation_holds'] or m['exact_supported_id_accounting'] is not True:raise ValueError('per_shard_denominator:'+str(s))
  if r['rows']!=m['prediction_inputs'] or r['preparation_manifest_sha256']!=b['preparation_manifest']['sha256']:raise ValueError('render_binding:'+str(s))
  prepared+=m['prediction_inputs'];holds+=m['preparation_holds']
 if prepared!=expected_prepared or holds!=expected_holds:raise ValueError('aggregate_denominator')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--render-inputs',type=Path,required=True);ap.add_argument('--root-receipt',type=Path,required=True);ap.add_argument('--root-v4-preparer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
 if a.output.exists():raise ValueError('fresh output')
 if sha(a.source/'terminal.json')!=PINS['terminal'] or sha(a.root_receipt)!=PINS['root_receipt'] or sha(a.root_v4_preparer)!=PINS['root_v4_preparer'] or sha(a.render_inputs/'manifest.json')!=PINS['render_manifest']:raise ValueError('top_level_pin')
 terminal=json.loads((a.source/'terminal.json').read_text());rm=json.loads((a.render_inputs/'manifest.json').read_text());preps={int(x['shard']):json.loads((a.source/f"shard-{int(x['shard']):04d}/preparation-manifest.json").read_text()) for x in terminal['shards']};validate_metadata(terminal,rm,preps)
 all_prepared=set();all_holds=set();audit=[]
 for b,r in zip(terminal['shards'],rm['shards']):
  shard=int(b['shard']);d=a.source/f'shard-{shard:04d}';m=preps[shard];mp=d/'preparation-manifest.json';pred=d/'prediction-inputs.jsonl';side=d/'training-sidecar.jsonl';holds=d/'preparation-holds.jsonl';render=a.render_inputs/r['path']
  if sha(mp)!=b['preparation_manifest']['sha256']:raise ValueError('manifest_hash:'+str(shard))
  for name,path in [('prediction-inputs.jsonl',pred),('training-sidecar.jsonl',side),('preparation-holds.jsonl',holds)]:
   pin=m['outputs'][name]
   verify_file(path,pin,str(shard)+':'+name)
  verify_file(render,r,str(shard)+':render')
  predictions=read_rows(pred);sidecars=read_rows(side);held=read_rows(holds);rendered=read_rows(render)
  if len(predictions)!=m['prediction_inputs'] or len(sidecars)!=m['training_sidecar_rows'] or len(held)!=m['preparation_holds'] or len(rendered)!=r['rows']:raise ValueError('physical_row_count:'+str(shard))
  validate_id_join(predictions,sidecars,held,rendered,str(shard))
  for rid,x in predictions.items():
   y=rendered[rid];identity=sidecars[rid]['identity'];expected=dict(x);deps=expected.pop('external_import_dependencies');source=Path(identity['source_path']);expected.update({'absolute_document_path':str(source),'workspace_root':str(source.parent.parent),'expected_dependencies':deps,'source_sha256':identity['source_sha256'],'package_id':identity['package_id'],'group_id':identity['group_id']})
   if y!=expected:raise ValueError('render_transform:'+rid)
   assert_no_target_keys(y)
   if hashlib.sha256(y['preedit_text'].encode()).hexdigest()!=y['preedit_sha256']:raise ValueError('preedit_hash:'+rid)
  if all_prepared&set(predictions) or all_holds&set(held):raise ValueError('cross_shard_duplicate:'+str(shard))
  all_prepared|=set(predictions);all_holds|=set(held);audit.append({'shard':shard,'status':m['status'],'prepared_rows':len(predictions),'hold_rows':len(held),'render_sha256':sha(render),'render_bytes':render.stat().st_size})
 if len(all_prepared)!=10948 or len(all_holds)!=4 or all_prepared&all_holds:raise ValueError('global_id_closure')
 ids_sha=hashlib.sha256(('\n'.join(sorted(all_prepared))+'\n').encode()).hexdigest()
 if ids_sha!=rm['ids_sha256']:raise ValueError('ids_sha')
 out={'schema':'sepalith.dat10.semantic10948.input_closure_audit.v1','status':'PASS','prepared_rows':10948,'explicit_holds':4,'accepted_scope':10952,'shards':audit,'checks':{'top_level_pins':True,'exact_15_shard_set':True,'status_allowlist_exact':True,'per_shard_prepared_plus_hold':True,'file_hash_bytes_rows':True,'prediction_sidecar_render_id_parity':True,'hold_disjointness':True,'exact_render_transform':True,'recursive_target_shaped_key_absence':True,'preedit_hash':True,'global_id_closure':True},'ids_sha256':ids_sha,'render_manifest_sha256':PINS['render_manifest'],'root_v4_preparer_sha256':PINS['root_v4_preparer'],'elapsed_seconds':time.monotonic()-start,'training_admission':False}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
