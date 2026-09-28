import copy,hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_plan,prepare_fallback,run_lane

def dump(path,values):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w')as f:
  for x in values:f.write(json.dumps(x,sort_keys=True)+'\n')
def js(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Provider4100(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp');self.root=Path(self.tmp.name);self.inputs=self.root/'inputs';self.inputs.mkdir();self.coverage=self.root/'coverage.json';self.candidates=self.root/'candidates.jsonl'
  js(self.coverage,{'partial':False,'completed_shards':41,'pending_shards':[]})
  candidates=[];per=[];duplicate_ids=[]
  for shard in range(41):
   count=4100//41+(1 if shard<4100%41 else 0);pred=[];side=[]
   for i in range(count):
    rid=f'p-{shard:02d}-{i:03d}';pre='a'*64 if len(duplicate_ids)<2 else hashlib.sha256(rid.encode()).hexdigest();row={'schema':'sepalith.dat10.sourcewalk-noop.prediction_input.v2','row_id':rid,'preedit_text':'x','preedit_sha256':pre,'cursor':{'line':0,'character':0},'path':'R/shared.R'if len(duplicate_ids)<2 else f'R/{rid}.R','absolute_document_path':f'/w{len(duplicate_ids)}/{rid}.R','workspace_root':f'/w{len(duplicate_ids)}','expected_dependencies':[],'document_eol':'lf'};pred.append(row);side.append({'row_id':rid});candidates.append({'row_id':rid,'shard':shard,'status':'provenance_supported_candidate_root_review_required'})
    if len(duplicate_ids)<2:duplicate_ids.append(rid)
   held=[]
   if shard==0:
    for i in range(121):rid=f'hp-{i}';held.append({'row_id':rid,'reason':'provenance_not_supported'});candidates.append({'row_id':rid,'shard':shard,'status':'hold_independent_provenance_failure'})
    for i in range(6):rid=f'hm-{i}';held.append({'row_id':rid,'reason':'mixed_eol'});candidates.append({'row_id':rid,'shard':shard,'status':'provenance_supported_candidate_root_review_required'})
   dump(self.inputs/f'shard-{shard:04d}.jsonl',pred);dump(self.inputs/f'shard-{shard:04d}.sidecar.jsonl',side);dump(self.inputs/f'shard-{shard:04d}.holds.jsonl',held)
   per.append({'shard':shard,'candidates':len(pred)+len(held),'prediction_inputs':len(pred),'holds':len(held),'prediction_sha256':sha(self.inputs/f'shard-{shard:04d}.jsonl'),'sidecar_sha256':sha(self.inputs/f'shard-{shard:04d}.sidecar.jsonl'),'holds_sha256':sha(self.inputs/f'shard-{shard:04d}.holds.jsonl')})
  dump(self.candidates,candidates)
  dup=[{'geometry':{'preedit_sha256':'a'*64,'cursor':{'line':0,'character':0},'path':'R/shared.R'},'count':2,'row_ids':duplicate_ids,'disposition':'retain_all_until_provider_prompt_target_dedup'}];dump(self.inputs/'duplicate-geometries.jsonl',dup)
  js(self.inputs/'manifest.json',{'schema':'sepalith.dat10.noop4100.reconstruction.v1','status':'complete_review_only','full_41_shard_closure':True,'candidate_rows':4227,'prediction_inputs':4100,'holds':127,'hold_accounting':{'provenance':121,'reviewed_mixed_eol':6},'duplicate_geometry':{'policy':'retain_all_until_provider_prompt_target_dedup','artifact':'duplicate-geometries.jsonl','sha256':sha(self.inputs/'duplicate-geometries.jsonl'),'groups':1,'rows_in_groups':2},'shards':per})
 def tearDown(self):self.tmp.cleanup()
 def plan(self):
  out=self.root/'plan.json'
  with mock.patch.object(build_plan,'COVERAGE_SHA',sha(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',sha(self.candidates)):
   result=build_plan.build(self.inputs,sha(self.inputs/'manifest.json'),self.coverage,self.candidates,out)
  return result,out
 def test_plan_binds_terminal_manifest_and_all_41_shards(self):
  plan,_=self.plan();self.assertEqual(len(plan['entries']),41);self.assertEqual(sum(plan['lane_rows']),4100);self.assertEqual(plan['upstream_hold_classes'],{'provenance':121,'mixed_eol_geometry':6});self.assertTrue(plan['prediction_inputs_target_free']);self.assertEqual(plan['fixed_training_target'],'NO_EDIT')
 def test_wrong_terminal_manifest_hash_fails(self):
  with mock.patch.object(build_plan,'COVERAGE_SHA',sha(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',sha(self.candidates)):
   with self.assertRaisesRegex(ValueError,'terminal reconstruction'):build_plan.build(self.inputs,'0'*64,self.coverage,self.candidates,self.root/'bad.json')
 def test_target_key_self_rejection_and_missing_shard_are_fatal(self):
  p=self.inputs/'shard-0000.jsonl';values=[json.loads(x)for x in p.read_text().splitlines()];values[0]['selection_target_or_gold_used']=False;dump(p,values);m=json.loads((self.inputs/'manifest.json').read_text());m['shards'][0]['prediction_sha256']=sha(p);js(self.inputs/'manifest.json',m)
  with mock.patch.object(build_plan,'COVERAGE_SHA',sha(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',sha(self.candidates)):
   with self.assertRaisesRegex(ValueError,'target/gold'):build_plan.build(self.inputs,sha(self.inputs/'manifest.json'),self.coverage,self.candidates,self.root/'bad.json')
  p.unlink()
  with mock.patch.object(build_plan,'COVERAGE_SHA',sha(self.coverage)),mock.patch.object(build_plan,'CANDIDATES_SHA',sha(self.candidates)):
   with self.assertRaisesRegex(ValueError,'all shard artifacts'):build_plan.build(self.inputs,sha(self.inputs/'manifest.json'),self.coverage,self.candidates,self.root/'bad2.json')
 def test_32k_plan_preserves_runtime_binding(self):
  plan,path=self.plan();render=self.root/'render16';render.mkdir()
  for item in plan['entries']:
   source=[json.loads(x)for x in (self.inputs/item['path']).read_text().splitlines()];out=render/f"shard-{item['shard']:04d}.jsonl";dump(out,[{'row_id':x['row_id'],'status':'hold'if x['row_id']==next(iter(source),{}).get('row_id')else'supported'}for x in source]);js(render/f"shard-{item['shard']:04d}.terminal.json",{'status':'complete','context_size':16384,'generation_reserve':2048,'input':{'sha256':item['sha256']},'output':{'sha256':sha(out)}})
  fallback=self.root/'fallback'
  with mock.patch.object(sys,'argv',['prepare_fallback.py','--plan',str(path),'--plan-sha256',sha(path),'--render16',str(render),'--output',str(fallback)]):prepare_fallback.main()
  m=json.loads((fallback/'manifest.json').read_text());self.assertEqual(m['provider_runtime_sha256'],plan['provider_runtime_sha256']);self.assertEqual(m['tokenizer_sha256'],plan['tokenizer_sha256']);self.assertEqual(m['provider_denominator'],4100)
 def test_lane_infrastructure_failure_is_not_a_provider_hold(self):
  plan,path=self.plan();output=self.root/'render'
  with mock.patch.object(sys,'argv',['run_lane.py','--plan',str(path),'--plan-sha256',sha(path),'--lane','0','--output',str(output)]),mock.patch.object(run_lane.subprocess,'run',return_value=mock.Mock(returncode=70)):
   with self.assertRaisesRegex(ValueError,'shard .* failed'):run_lane.main()
 def test_real_target_free_noop_reaches_frozen_provider(self):
  source=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop-expansion-v3/inputs-full41/shard-0010.jsonl')
  with source.open()as stream:row=next(json.loads(x)for x in stream if '43b24d15b32aae89fdf72245' in x)
  row.pop('selection_target_or_gold_used',None);inp=self.root/'real.jsonl';out=self.root/'real.out.jsonl';dump(inp,[row]);provider=build_plan.PROVIDER;base_provider=build_plan.PLAN/'docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2'
  command=[shutil.which('node'),'--no-warnings=ExperimentalWarning','--experimental-strip-types',str(provider/'render_shard.ts'),str(inp),str(out),'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(base_provider/'tokenize_bridge.py'),str(build_plan.TOKENIZER),str(base_provider/'source/namespace_evidence.R'),'16384','2048']
  result=subprocess.run(command,capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(result.returncode,0,result.stderr);rendered=json.loads(out.read_text());self.assertEqual((rendered['row_id'],rendered['status'],rendered['mode']),('43b24d15b32aae89fdf72245','supported','full_document'));self.assertEqual(rendered['generation_reserve'],2048);self.assertEqual(rendered['source_inventory_status'],'source_import_inventory_complete')
  old=self.root/'old.out.jsonl';old_command=[*command];old_command[3]=str(build_plan.PLAN/'docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2/render_shard.ts');old_command[5]=str(old);old_result=subprocess.run(old_command,capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(old_result.returncode,0,old_result.stderr);baseline=json.loads(old.read_text());rendered_without_status={k:v for k,v in rendered.items()if k!='source_inventory_status'};self.assertEqual(rendered_without_status,baseline)
 def test_real_terminal_nonempty_noop_uses_explicit_no_following_evidence(self):
  recon=Path('docs/campaign/work/lead/r2-noop4100-reconstruction-preparation-v1/source').resolve();sys.path.insert(0,str(recon));import prepare_noop4100 as reconstruction
  data=Path('/mnt/e/sepalith/campaign-20260915/data-work');rid='47a96cc61e9860688ea9114e';decision=reconstruction.base.rows(data/'Sourcewalk-noop-expansion-v4/recovery-decisions.jsonl')[rid];shard=decision['shard'];ledger=reconstruction.base.rows(data/f'Sourcewalk-independent-replay-v3/full-01/shards/shard-{shard:04d}/ledger.jsonl')[rid];packet=reconstruction.base.rows(data/f'DAT10-novel-v1/source-walk-shards-v1/shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl')[rid];row,_=reconstruction.recovered_output(packet,ledger,decision)
  inp=self.root/'terminal.jsonl';out=self.root/'terminal.out.jsonl';dump(inp,[row]);provider=build_plan.PROVIDER
  base_provider=build_plan.PLAN/'docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2';command=[shutil.which('node'),'--no-warnings=ExperimentalWarning','--experimental-strip-types',str(ROOT/'render_shard.ts'),str(inp),str(out),'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(base_provider/'tokenize_bridge.py'),str(build_plan.TOKENIZER),str(base_provider/'source/namespace_evidence.R'),'16384','2048']
  result=subprocess.run(command,capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(result.returncode,0,result.stderr);rendered=json.loads(out.read_text());self.assertEqual(rendered['status'],'supported');self.assertEqual(rendered['source_inventory_status'],'source_import_inventory_no_following_function')
  missing=self.root/'missing-r.out.jsonl';bad=[*command];bad[5]=str(missing);failed=subprocess.run(bad,capture_output=True,text=True,timeout=30,env={**os.environ,'PATH':'','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertNotEqual(failed.returncode,0);self.assertIn('prediction_namespace_infrastructure:Error',failed.stderr)
 def test_real_crlf_blank_geometry_reaches_provider_with_raw_preedit(self):
  recon=Path('docs/campaign/work/lead/r2-noop4100-reconstruction-preparation-v1/source').resolve();sys.path.insert(0,str(recon));import prepare_noop4100 as reconstruction
  data=Path('/mnt/e/sepalith/campaign-20260915/data-work');rid='49da0198cb6f200c0214a9ee';decision=reconstruction.base.rows(data/'Sourcewalk-noop-expansion-v4/recovery-decisions.jsonl')[rid];shard=decision['shard'];ledger=reconstruction.base.rows(data/f'Sourcewalk-independent-replay-v3/full-01/shards/shard-{shard:04d}/ledger.jsonl')[rid];packet=reconstruction.base.rows(data/f'DAT10-novel-v1/source-walk-shards-v1/shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl')[rid];row,_=reconstruction.recovered_output(packet,ledger,decision);self.assertIn('\r\n',row['preedit_text'])
  inp=self.root/'crlf.jsonl';out=self.root/'crlf.out.jsonl';dump(inp,[row]);base_provider=build_plan.PLAN/'docs/campaign/work/lead/r2-semantic10948-provider-materialization-v2';command=[shutil.which('node'),'--no-warnings=ExperimentalWarning','--experimental-strip-types',str(ROOT/'render_shard.ts'),str(inp),str(out),'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(base_provider/'tokenize_bridge.py'),str(build_plan.TOKENIZER),str(base_provider/'source/namespace_evidence.R'),'16384','2048'];result=subprocess.run(command,capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(result.returncode,0,result.stderr);rendered=json.loads(out.read_text());self.assertEqual((rendered['row_id'],rendered['status']), (rid,'supported'));self.assertEqual(rendered['source_inventory_status'],'source_import_inventory_complete')

if __name__=='__main__':unittest.main(verbosity=2)
