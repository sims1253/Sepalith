"""Metadata review binding exact panel rows and recomputing causal NLL."""
import argparse,hashlib,json,math
from pathlib import Path
FIXTURES={
 'anchor2k':(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl'),'efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8',499,661360,'71217c0470d87a5dcfaee45b914743ba0dc4acc75c980879e05e671a45ca4232'),
 '8k':(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-long-holdout-preparation-v1/output/cpt_validation_8k.jsonl'),'3e6625f075af7639e5e99d76b2fc0b89578c4011408133a32be1a6479b6b8d74',20,126464,'ccb0a781d36cb8949663ccfcd9a0c0b0debfbae7d2a4b4fb511428efbb23c9a8'),
 '16k':(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-long-holdout-preparation-v1/output/cpt_validation_16k.jsonl'),'ac0adc964592488e2e345ad16f4c832e94e9772b16129ada6896da36c7b496ac',6,69137,'77c87f26f18837f45b24beeebfd6c52d10ca5df9bde359619efb7556f5ca7fc3')}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def fixture_identity(label):
 path,expected,rows,tokens,ids_sha=FIXTURES[label];assert sha(path)==expected
 ids=[];docs=[]
 with path.open() as f:
  for line in f:
   row=json.loads(line);rid=row.get('row_id') or row.get('id');did=row.get('document_id') or row.get('source_document_id')
   assert isinstance(rid,str) and rid;ids.append(rid)
   if isinstance(did,str) and did:docs.append(did)
 assert len(ids)==rows and len(set(ids))==rows
 assert hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()==ids_sha
 return ids,sorted(set(docs)),tokens
def validate_panel(panel,label,binding_sha):
 ids,docs,tokens=fixture_identity(label);assert panel['step']==706 and panel['binding_sha256']==binding_sha and panel['length_stratum']==label
 assert panel['denominators']['validation_rows']==len(ids) and panel['denominators']['validation_loss_tokens']==tokens
 metrics=panel['row_metrics'];assert [x['row_id'] for x in metrics]==ids
 assert panel['case_ids']==docs
 loss_tokens=0;loss_sum=0.0
 for row in metrics:
  assert type(row['loss_tokens']) is int and row['loss_tokens']>0 and math.isfinite(row['loss_sum']) and math.isfinite(row['mean_causal_nll'])
  assert math.isclose(row['mean_causal_nll'],row['loss_sum']/row['loss_tokens'],rel_tol=0,abs_tol=1e-12)
  loss_tokens+=row['loss_tokens'];loss_sum+=row['loss_sum']
 assert loss_tokens==tokens
 computed=loss_sum/tokens;assert math.isclose(panel['metrics']['mean_causal_nll'],computed,rel_tol=0,abs_tol=1e-15)
 return computed
def review(binding_path,output):
 binding_sha=sha(binding_path);result=json.loads((output/'result.json').read_text());assert result['status']=='evaluations_complete' and result['binding_sha256']==binding_sha and result['training_performed'] is False and result['promotion_authorized'] is False
 values={label:validate_panel(json.loads((output/f'{label}.json').read_text()),label,binding_sha) for label in FIXTURES}
 print(json.dumps({'schema':'sepalith.sft11.cpt706-matched-review.v2','status':'exact_fixture_metrics_verified_no_promotion','checkpoint_step':706,'binding_sha256':binding_sha,'mean_causal_nll':values},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();review(a.binding,a.output)
