import ast,copy,hashlib,json,os,sys,unittest
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path[:0]=[str(H),str(H/'native_evaluator')];os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from selection_contract import check_task_recipe,export_commands,SOURCE,TOK,TOKCFG
from selection_binding import validate_profile
from bind_native_profile import build

class Tests(unittest.TestCase):
 def recipe(self,kind='midtrain_control'):
  r=json.loads((H.parent/'r2-cloud-control-entry-v1/task-recipe.template.json').read_text());r['identity']['source']=SOURCE;r['identity']['parent'].update(kind=kind,revision='explicit-bound-revision',weights_sha256='1'*64);r['identity']['policy']['initialization']='new_lora_on_midtrain_control' if kind=='midtrain_control' else 'new_lora_on_cpt_merged_parent';r['model_path']='/synthetic/explicit-parent';r['inputs']=[{'path':r['model_path']+'/'+n,'sha256':s} for n,s in [('model.safetensors','1'*64),('tokenizer.json',TOK),('tokenizer_config.json',TOKCFG)]];return r
 def parent(self,kind='midtrain_control'):
  r=self.recipe(kind);i=r['identity'];return {'schema_version':'sepalith.r2-task-sft.parent-manifest.v1','kind':'merged_task_sft','tokenizer_original_bytes_restored':True,'sft_identity':i,'sft_checkpoint_manifest':{'identity':i,'full':True,'step':250},'source_cursor':4000,'recipe_sha256':'2'*64,'checkpoint_manifest_sha256':'3'*64,'merged_model_path':'/synthetic/merged-r2','merge_verification':{'accumulation_dtype':'float32','rounding':'one final BF16 cast','independent_lora_matrix_merges_exact':294}}
 def profile(self,kind='midtrain_control'):
  integrity={'status':'Q8_integrity_verified_pending_native_quality','parent_manifest_sha256':'4'*64,'q8':{'path':'/synthetic/export/model-Q8_0.gguf','bytes':2679710496,'sha256':'5'*64}}
  return build(self.parent(kind),integrity,'4'*64,'6'*64)
 def test_fresh_midtrain_parent_supported(self):check_task_recipe(self.recipe())
 def test_explicit_merged_cpt_parent_supported(self):check_task_recipe(self.recipe('cpt_merged'));validate_profile(self.profile('cpt_merged'))
 def test_cloud_parent_relocation_keeps_identity_and_hashes(self):
  from bind_merge_recipe import relocate
  r=self.recipe();bound=relocate(r,{'path':'/synthetic/cloud-recipe.json','sha256':'2'*64},'/synthetic/local-midtrain')
  self.assertEqual(bound['identity'],r['identity']);self.assertEqual([x['sha256'] for x in bound['inputs']],[x['sha256'] for x in r['inputs']]);self.assertEqual(bound['model_path'],'/synthetic/local-midtrain')
 def test_stale_source_rejected(self):
  r=self.recipe();r['identity']['source']='5149a406'+'0'*56
  with self.assertRaises(ValueError):check_task_recipe(r)
 def test_wrong_parent_input_hash_rejected(self):
  r=self.recipe();r['inputs'][0]['sha256']='8'*64
  with self.assertRaises(ValueError):check_task_recipe(r)
 def test_tokenizer_drift_rejected(self):
  r=self.recipe();r['inputs'][1]['sha256']='8'*64
  with self.assertRaises(ValueError):check_task_recipe(r)
 def test_exact_export_argv_no_theta0(self):
  commands=export_commands('/synthetic/merged-r2','/synthetic/exports-r2')
  self.assertEqual(commands[0][3],'/synthetic/merged-r2');self.assertEqual(commands[0][-2:],['--outtype','f16']);self.assertEqual(commands[1][1:5],['--output-tensor-type','q8_0','--token-embedding-type','q8_0']);self.assertEqual(commands[1][-2:],['Q8_0','2']);self.assertNotIn('theta0',json.dumps(commands))
 def test_native_controller_actual_argv_uses_bound_model(self):
  tree=ast.parse((H/'native_controller/root_controller.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_argv');ns={'BUNDLE':Path('/synthetic/cuda-b10453'),'MODEL':Path('/synthetic/export/model-Q8_0.gguf')};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-native-argv','exec'),ns)
  argv=ns['native_argv'](18403);self.assertEqual(argv[2],'/synthetic/export/model-Q8_0.gguf');self.assertEqual(argv[argv.index('-b')+1],'256');self.assertEqual(argv[argv.index('-ub')+1],'256');self.assertEqual(argv[argv.index('-c')+1],'4096');self.assertEqual(argv[argv.index('--parallel')+1],'1')
 def test_bound_profile_has_new_hash_revision_and_tokenizer_path(self):
  p=self.profile();validate_profile(p);self.assertEqual(p['model']['sha256'],'5'*64);self.assertEqual(p['selection']['tokenizer_dir'],'/synthetic/merged-r2');self.assertNotIn('theta0',p['model_profile']['modelRevision'])
 def test_unbound_profile_rejected(self):
  with self.assertRaises(ValueError):validate_profile(json.loads((H/'native-profile.template.json').read_text()))
 def test_cap_or_graphopt_drift_rejected(self):
  for field,value in [('output',512),('cuda_graph_opt',1),('batch',1024)]:
   p=self.profile();p[field]=value
   with self.assertRaises(ValueError):validate_profile(p)
 def test_checkpoint_step_must_be_selected_milestone(self):
  p=self.profile();p['selection']['checkpoint_step']=25
  with self.assertRaises(ValueError):validate_profile(p)
 def test_profile_hash_mismatch_rejected(self):
  p=self.profile();p['model_profile']['modelSha256']='7'*64
  with self.assertRaises(ValueError):validate_profile(p)
 def test_final_row_gates_and_transport_unchanged(self):
  old=H.parent/'final-native-evaluator-preparation-v3'
  for name in ['native_transport.py','native_evaluator.py','final_binding.py','final_row_gate.py']:
   self.assertEqual((old/name).read_bytes(),(H/'native_evaluator'/name).read_bytes())
 def test_accepted_fp32_merge_call_order(self):
  tree=ast.parse((H/'merge_task_sft_cpu.py').read_text());lines={}
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='model' and n.func.attr in ('float','merge_and_unload','to'):lines[n.func.attr]=n.lineno
  self.assertLess(lines['float'],lines['merge_and_unload']);self.assertLess(lines['merge_and_unload'],lines['to'])
 def test_no_framework_or_native_launch_imported(self):self.assertFalse(set(sys.modules)&{'torch','transformers','peft','gguf'})
if __name__=='__main__':unittest.main(verbosity=2)
