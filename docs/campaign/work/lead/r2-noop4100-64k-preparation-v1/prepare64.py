"""Materialize the exact 38 context-only no-op inputs for root-reviewed 64K retry."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path

PLAN=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/plan16.json')
PLAN_SHA='093fd437f6cc4a364b7273e7936459c836a18a31bc06fd84ec39ec88faacd725'
SOURCE=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-reconstruction-v1/inputs-full41')
RENDER32=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render32-postrender-v2')
REVIEW=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-noop4100-postrender-root-v2/render32-root-review.json')
REVIEW_SHA='64dec75c81aa702d5b015ee3930bb9ab14c00b5c05684dbf3d2fbf88e6dd7768'
POLICY=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/selected-policy-postrender-v2/manifest.json')
POLICY_SHA='f3563a42c2a42ea7cb6125eab8ea3dfe16294cf587d940834f3af081df0759ca'
SELECTED_CONTEXTS_SHA='be797af50a32d76820ab0b5377e8d0c116e8bf9d9ac40189128c886c4fd58c5b'
PROVIDER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-noop4100-parse-retry-preparation-v1/source-manifest.json')
PROVIDER_SHA='114b0a410cf15c821ce83c8d569b16d831640f744f3e53d303d849929e503919'
QUEUE_REASONS={'complete_span_not_applicable','complete_span_unresolved'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def line_sha(ids):return hashlib.sha256((''.join(x+'\n' for x in ids)).encode()).hexdigest()

def partition(rows,expected_total=105,expected_queue=38):
 assert len(rows)==expected_total and len({x['row_id'] for x in rows})==expected_total
 assert all(x['status']=='hold' and isinstance(x.get('reasons'),list) and x['reasons'] for x in rows)
 queued=[x for x in rows if x['reasons'][0] in QUEUE_REASONS]
 retained=[x for x in rows if x['reasons'][0] not in QUEUE_REASONS]
 assert len(queued)==expected_queue and len(retained)==expected_total-expected_queue
 return queued,retained

def validate_prediction_input(row):
 assert row['schema'] in {'sepalith.dat10.sourcewalk-noop.prediction_input.v1','sepalith.dat10.sourcewalk-noop.prediction_input.v2'}
 assert not any(('target' in key.lower() or 'gold' in key.lower() or 'label' in key.lower()) for key in row)
 return True

def classify():
 assert sha(REVIEW)==REVIEW_SHA and sha(POLICY)==POLICY_SHA and sha(PROVIDER)==PROVIDER_SHA
 policy=read(POLICY);assert policy['status']=='complete_review_only' and policy['provider_denominator']==4100
 assert policy['provider_supported']==3995 and policy['provider_holds_after_32k']==105
 assert policy['selected']['rows']==4100 and policy['selected']['sha256']==SELECTED_CONTEXTS_SHA
 review=read(REVIEW);assert review['status']=='root_verified_32k_retry_outputs' and review['rows']==228 and review['counts']=={'hold':105,'supported':123}
 rows=[]
 for proof in review['proof']:
  path=RENDER32/f"shard-{proof['shard']:04d}.jsonl";assert sha(path)==proof['output_sha256']
  with path.open(encoding='utf-8') as stream:
   for line in stream:
    row=json.loads(line)
    if row['status']=='hold':rows.append(row)
 queued,retained=partition(rows)
 assert sum(x['reasons'][0]=='complete_span_not_applicable' for x in queued)==18
 assert sum(x['reasons'][0]=='complete_span_unresolved' for x in queued)==20
 return queued,retained

def materialize(output):
 output=Path(output);assert not output.exists(),'fresh 64K input root required'
 queued,retained=classify();wanted={x['row_id'] for x in queued};plan=read(PLAN);assert sha(PLAN)==PLAN_SHA
 assert plan['rows']==4100 and Path(plan['input_root'])==SOURCE and len(plan['entries'])==41
 parent=output.parent;parent.mkdir(parents=True,exist_ok=True)
 stage=Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=parent));inputs=stage/'inputs';inputs.mkdir()
 found={};entries=[]
 try:
  for item in plan['entries']:
   src=SOURCE/item['path'];raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'] and len(raw)==item['bytes']
   selected=[]
   for line in raw.splitlines(keepends=True):
    row=json.loads(line);rid=row['row_id']
    if rid in wanted:
     validate_prediction_input(row)
     assert rid not in found;found[rid]=line;selected.append(line)
   dst=inputs/item['path'];dst.write_bytes(b''.join(selected));payload=dst.read_bytes()
   entries.append({'shard':item['shard'],'path':str(Path('inputs')/item['path']),'rows':len(selected),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'core':4 if item['shard']%2==0 else 6})
  assert set(found)==wanted and len(found)==38
  queued_ids=sorted(wanted);retained_ids=sorted(x['row_id'] for x in retained)
  with (stage/'queued-ids.jsonl').open('x') as f:
   for x in queued_ids:f.write(json.dumps({'row_id':x})+'\n')
  with (stage/'retained-holds.jsonl').open('x') as f:
   for x in sorted(retained,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True)+'\n')
  manifest={'schema':'sepalith.dat10.noop4100.context64-queue.v1','status':'prepared_inputs_only_not_rendered','context_size':65536,'generation_reserve':2048,'rows':38,'shards':41,'nonempty_shards':sum(x['rows']>0 for x in entries),'entries':entries,'queued_ids_sha256':line_sha(queued_ids),'retained_hold_ids_sha256':line_sha(retained_ids),'retained_holds':67,'prior_supported':3995,'provider_denominator':4100,'closure':'3995+38+67=4100','selection':{'first_reasons':{'complete_span_not_applicable':18,'complete_span_unresolved':20}},'source_plan_sha256':PLAN_SHA,'selected_policy_manifest_sha256':POLICY_SHA,'selected_contexts_sha256':SELECTED_CONTEXTS_SHA,'render32_review_sha256':REVIEW_SHA,'provider_source_manifest_sha256':PROVIDER_SHA,'prediction_inputs_target_free':True,'fixed_target_joined':False,'training_admitted':False,'render_authorized':False}
  dump(stage/'manifest.json',manifest)
  os.rename(stage,output);stage=None
  return manifest
 finally:
  if stage is not None:shutil.rmtree(stage,ignore_errors=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(materialize(a.output),sort_keys=True))
