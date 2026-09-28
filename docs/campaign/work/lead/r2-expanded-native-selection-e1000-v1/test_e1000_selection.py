"""CPU metadata tests for the E full1000 merge/export/native packet."""
import ast,copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
H=Path(__file__).resolve().parent;P=H/'packet';sys.path[:0]=[str(P),str(P/'native_evaluator')]
from bind_native_profile import build
from selection_contract import PARENT,RECIPE,SOURCE,check_task_recipe,export_commands

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.recipe_path=H.parent/'r2-expanded-sft-e/recipe.json';cls.recipe=json.loads(cls.recipe_path.read_text())
 def test_exact_e_recipe_passes(self):
  self.assertEqual(hashlib.sha256(self.recipe_path.read_bytes()).hexdigest(),RECIPE);self.assertEqual(check_task_recipe(self.recipe,RECIPE)['weights_sha256'],PARENT)
 def test_step500_recipe_and_wrong_recipe_hash_reject(self):
  d=H.parent/'r2-expanded-sft-d/recipe.json'
  with self.assertRaises(ValueError):check_task_recipe(json.loads(d.read_text()),hashlib.sha256(d.read_bytes()).hexdigest())
  with self.assertRaises(ValueError):check_task_recipe(self.recipe,'1'*64)
 def test_wrong_source_rejects(self):
  bad=copy.deepcopy(self.recipe);bad['identity']['source']='1'*64
  with self.assertRaises(ValueError):check_task_recipe(bad,RECIPE)
 def test_exact_resume_and_terminal_cursor(self):
  check_task_recipe(self.recipe,RECIPE);self.assertEqual(self.recipe['recovery_binding']['terminal_consumed_draws'],16000)
  self.assertNotIn('resume_identity_compatibility',self.recipe)
 def test_merge_requires_full1000_and_cursor16000(self):
  text=ast.unparse(ast.parse((P/'merge_expanded_sft_cpu.py').read_text()))
  for value in ("checkpoint['step'] == 1000","checkpoint['full'] is True","campaign_state['full'] is True","== 16000","check_task_recipe(recipe, digest(args.recipe))"):self.assertIn(value,text)
  self.assertNotIn("checkpoint['step'] == 500",text)
 def parent(self,step=1000,source=SOURCE,recipe=RECIPE):
  identity=copy.deepcopy(self.recipe['identity']);identity['source']=source;checkpoint={'identity':identity,'full':True,'step':step}
  return {'schema_version':'sepalith.r2-task-sft.parent-manifest.v1','kind':'merged_task_sft','tokenizer_original_bytes_restored':True,'sft_identity':identity,'sft_checkpoint_manifest':checkpoint,'source_cursor':step*16,'recipe_sha256':recipe,'checkpoint_manifest_sha256':'4'*64,'merged_model_path':'/mnt/e/synthetic-merged','merge_verification':{'accumulation_dtype':'float32','rounding':'one final BF16 cast','independent_lora_matrix_merges_exact':294}}
 def integrity(self):return {'status':'Q8_integrity_verified_pending_native_quality','parent_manifest_sha256':'5'*64,'q8':{'path':'/mnt/e/synthetic-quant/model-Q8_0.gguf','bytes':2679710496,'sha256':'6'*64}}
 def test_profile_accepts_only_e1000(self):
  p=build(self.parent(),self.integrity(),'5'*64,'7'*64);self.assertEqual(p['selection']['checkpoint_step'],1000)
  for parent in (self.parent(step=500),self.parent(source='1'*64),self.parent(recipe='2'*64)):
   with self.assertRaises(ValueError):build(parent,self.integrity(),'5'*64,'7'*64)
 def test_export_v2_and_v4_guard_commands(self):
  root=json.loads((H/'root-commands.json').read_text());self.assertIn('export_candidate_v2.py',json.loads((H/'export-command.json').read_text())[2]);self.assertIn('host-memory-guard-v4',root['merge_guard'][2]);self.assertIn('host-memory-guard-v4',root['export_guard'][2])
  for name in ('prepare_root_bindings.py','export_candidate_v2.py'):
   text=ast.unparse(ast.parse((P/name).read_text()));self.assertIn("check_task_recipe(m['sft_identity'], m['recipe_sha256'])",text);self.assertIn("m['sft_checkpoint_manifest']['step'] == 1000",text);self.assertIn("m['source_cursor'] == 16000",text)
 def test_observation_sequence_uses_no_argument_first(self):
  seq=json.loads((H/'root-commands.json').read_text())['observation_sequence'];self.assertEqual(Path(seq[0][-1]).name,'observe_client.py');self.assertEqual(len(seq[0]),5);self.assertEqual(seq[2][-1],'--check-policy-only')
  self.assertNotIn('ROOT_FILL',json.dumps(json.loads((H/'root-commands.json').read_text())))
 def test_harness_unchanged_except_selection_binding(self):
  old=H.parent/'r2-expanded-native-selection-d500-v1/packet'
  for rel in ('native_controller/root_controller.py','native_controller/guarded_dev_client.py','native_controller/client_source_policy.py','native_controller/origin_snapshot.py','native_controller/observe_client.py','native_controller/build_closure.py','native_evaluator/native_identity.py','native_evaluator/native_transport.py','native_evaluator/native_evaluator.py','native_evaluator/rehearse_dev.py','native_evaluator/final_binding.py','native_evaluator/final_row_gate.py'):
   self.assertEqual((old/rel).read_bytes(),(P/rel).read_bytes(),rel)
 def test_no_model_framework_imported(self):self.assertFalse(set(sys.modules)&{'torch','transformers','peft','gguf'})
if __name__=='__main__':unittest.main(verbosity=2)
