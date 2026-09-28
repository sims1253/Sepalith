import hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');PACKET=PLAN/'docs/campaign/work/lead/r2-native-varlen-canary-preparation-v1';TRAINING=PACKET/'source/experiments/training';sys.path.insert(0,str(TRAINING))
RESUME=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-v1/full/checkpoint-194')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Contract(unittest.TestCase):
 def test_actual_prepared_recipes_bind_exact_next_window_and_source_frontdoor(self):
  runner=load('runner',TRAINING/'native_varlen_canary.py')
  for arm,file in [('ordinary_reference','ordinary-reference.recipe.prepared.json'),('varlen_candidate','varlen-candidate.recipe.prepared.json')]:
   recipe=json.loads((PACKET/'prepared'/file).read_text());runner.verify_source(recipe);c=recipe['varlen_canary']
   self.assertEqual((c['arm'],c['source_global_step'],c['target_global_step'],c['source_cursor'],c['target_cursor'],c['logical_rows'],c['updates']),(arm,194,202,2048,2176,16,8))
   self.assertFalse(c['production_admitted'])
 def test_actual_canary_admission_contract_rejects_wrong_cursor(self):
  runner=load('runner_admit',TRAINING/'native_varlen_canary.py');recipe_path=PACKET/'prepared/varlen-candidate.recipe.prepared.json';recipe=json.loads(recipe_path.read_text())
  value={'schema':'sepalith.sft11.native-varlen-canary-admission.v1','status':'admitted','launch_authorized':True,'bound_recipe_sha256':sha(recipe_path),'arm':'varlen_candidate','logical_rows':16,'updates':8,'source_checkpoint':str(RESUME),'source_checkpoint_manifest_sha256':'2ad012580f0a7dccfdfd7f546ad5431893ee56c3df2e4adf01ddc4bab8e4b072','source_global_step':194,'target_global_step':202,'source_cursor':2048,'target_cursor':2176}
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   p=Path(td)/'admission.json';p.write_text(json.dumps(value));result=runner.validate_canary_admission(recipe_path,recipe,RESUME,p);self.assertEqual(result['target_cursor'],2176)
   value['target_cursor']=2192;p.write_text(json.dumps(value))
   with self.assertRaisesRegex(ValueError,'cursor differs'):runner.validate_canary_admission(recipe_path,recipe,RESUME,p)
 def test_startup_creates_parent_and_refuses_existing_evidence(self):
  runner=load('runner_start',TRAINING/'native_varlen_canary.py')
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   archive=Path(td)/'nested/archive';recipe={'outputs':{'archive':str(archive)}};actual,telemetry=runner.prepare_fresh_canary_archive(recipe);self.assertEqual(actual,archive);self.assertTrue(archive.is_dir());telemetry.write_text('{}\n')
   with self.assertRaisesRegex(ValueError,'not fresh'):runner.prepare_fresh_canary_archive(recipe)
 def test_comparator_accepts_metadata_only_mechanical_pass_and_rejects_loss_drift(self):
  compare=load('compare',TRAINING/'compare_native_varlen_canary.py')
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td)
   def arm(name,kind,loss,calls):
    a=root/name;cp=a/'full/checkpoint-195';cp.mkdir(parents=True);files={x:{'bytes':1,'sha256':'a'*64}for x in ('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','campaign-state.json','trainer_state.json')};(cp/'campaign-manifest.json').write_text(json.dumps({'full':True,'checkpoint_kind':'full_weights','step':195,'files':files}));r={'canary_arm':kind,'source_global_step':194,'source_checkpoint_manifest_sha256':'s'*64,'global_optimizer_step_offset':66,'global_step':195,'initial_cursor':2048,'observed_draws':16,'last_draw_position':2063,'loss_denominators':[99],'logical_update_losses':[loss],'logical_rows':16,'logical_updates':1,'draw_ids_sha256':'r'*64,'scientific_bindings':{'tokenizer_sha256':'t','rows_sha256':'d','cache_manifest_sha256':'c','schedule_sha256':'q','optimizer_sha256':'o'},'optimizer_dispatch_sha256':'d'*64,'physical_forward_backward_calls':calls,'train_loss':loss,'terminal_checkpoint':str(cp)};(a/'run-result.json').write_text(json.dumps(r));events=[{'event':'pre_optimizer','finite_gradient_tensors':381,'nonzero_gradient_tensors':381},{'event':'canary_update_timing','step':195,'ordinal':1,'seconds':2.0,'full_step_seconds':2.0,'forward_backward_seconds':1.2,'gradient_audit_seconds':.2,'optimizer_scheduler_to_step_end_seconds':.6,'synchronized':True,'excludes_save_and_evaluation':True,'warmup':True,'timed':False}];(a/'telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in events));return a
   ordinary=arm('ordinary','ordinary_reference',1.0,16);packed=arm('packed','varlen_candidate',1.0005,3);result=compare.compare(ordinary,packed,root/'comparison.json');self.assertEqual(result['packed_forward_backward_calls'],3);self.assertFalse(result['production_admitted'])
   value=json.loads((packed/'run-result.json').read_text());value['physical_forward_backward_calls']=16;(packed/'run-result.json').write_text(json.dumps(value));self.assertEqual(compare.compare(ordinary,packed,root/'all-long.json')['physical_call_reduction_ratio'],1)
   value['draw_ids_sha256']='x'*64;(packed/'run-result.json').write_text(json.dumps(value))
   with self.assertRaisesRegex(ValueError,'binding differs'):compare.compare(ordinary,packed,root/'wrong-draws.json')
   value['draw_ids_sha256']='r'*64;(packed/'run-result.json').write_text(json.dumps(value))
   (packed/'run-result.json').write_text((packed/'run-result.json').read_text().replace('1.0005','1.01'))
   with self.assertRaisesRegex(ValueError,'0.1%'):compare.compare(ordinary,packed,root/'bad.json')
 def test_native_attestation_preserves_reviewed_cuda_descriptor_handoff(self):
  self.assertEqual(sha(TRAINING/'native_launch_attestation.py'),'c611138a53400d49b79d0a7889deac93cfc969e0d1273f5fe100726ef878f05b')
if __name__=='__main__':unittest.main(verbosity=2)
