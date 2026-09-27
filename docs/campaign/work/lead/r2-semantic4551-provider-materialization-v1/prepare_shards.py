#!/usr/bin/env python3
import hashlib,json,os,tempfile
from pathlib import Path
PRED=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/combined/prediction-inputs.jsonl')
SIDE=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/combined/training-sidecar.jsonl')
COMBINED=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1/combined/manifest.json')
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1/input-shards')
PINS={PRED:'5bba7f3806bddeb58d11209a852aa397ad1b7f55cb368519d95ac5b7358b19aa',SIDE:'e31cc3c910de6fb426a6bda8c35382cc4dd50514c4ea5fd196513943cd0d1da6'}
COMBINED_SHA='976636c06e57ac455c504223b807321dc0870e2baf4e69609058a1e2e3657787'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
for p,h in PINS.items():
 if sha(p)!=h:raise ValueError('input pin differs:'+str(p))
if sha(COMBINED)!=COMBINED_SHA:raise ValueError('combined manifest pin differs')
combined=json.loads(COMBINED.read_text())
if combined['prepared']!=4551 or combined['holds']!=3 or combined['files']['prediction-inputs.jsonl']['sha256']!=PINS[PRED] or combined['files']['training-sidecar.jsonl']['sha256']!=PINS[SIDE]:raise ValueError('combined manifest binding differs')
identities={}
for raw in SIDE.open():
 x=json.loads(raw);rid=x['row_id'];identity=x['identity']
 if rid in identities:raise ValueError('duplicate sidecar ID')
 identities[rid]={'source_path':identity['source_path'],'source_sha256':identity['source_sha256'],'package_id':identity['package_id'],'group_id':identity['group_id']}
rows=[]
for raw in PRED.open():
 x=json.loads(raw);rid=x['row_id']
 if any(any(word in key.lower() for word in ('target','gold','reward','completion')) for key in x):raise ValueError('prediction target-shaped field:'+rid)
 if rid not in identities or hashlib.sha256(x['preedit_text'].encode()).hexdigest()!=x['preedit_sha256']:raise ValueError('prediction identity:'+rid)
 source=Path(identities[rid]['source_path']);root=source.parent.parent
 rows.append({'row_id':rid,'path':x['path'],'absolute_document_path':str(source),'workspace_root':str(root),'preedit_text':x['preedit_text'],'preedit_sha256':x['preedit_sha256'],'cursor':x['cursor'],'document_eol':x['document_eol'],'expected_dependencies':x['external_import_dependencies'],'source_sha256':identities[rid]['source_sha256'],'package_id':identities[rid]['package_id'],'group_id':identities[rid]['group_id']})
if len(rows)!=4551 or len({x['row_id'] for x in rows})!=4551 or set(identities)!=set(x['row_id'] for x in rows):raise ValueError('4551 ID closure')
if OUT.exists():raise ValueError('fresh output required')
OUT.mkdir(parents=True)
entries=[]
for shard in range(2):
 values=[x for i,x in enumerate(sorted(rows,key=lambda y:y['row_id'])) if i%2==shard];path=OUT/f'shard-{shard:04d}.jsonl'
 with path.open('x') as f:
  for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 entries.append({'shard':shard,'path':str(path),'rows':len(values),'bytes':path.stat().st_size,'sha256':sha(path),'ids_sha256':hashlib.sha256(('\n'.join(x['row_id'] for x in values)+'\n').encode()).hexdigest()})
manifest={'schema':'sepalith.dat10.semantic4551.provider_input_shards.v1','status':'complete_target_free','rows':4551,'shards':entries,'inputs':{str(p):h for p,h in PINS.items()},'combined_manifest_sha256':COMBINED_SHA,'upstream_geometry_holds':3,'prediction_keys_target_shaped_absent':True,'target_values_copied':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(manifest,sort_keys=True))
