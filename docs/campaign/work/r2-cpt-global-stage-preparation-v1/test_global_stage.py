import copy,hashlib,json,os,random,sys,tempfile,unittest
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'source/experiments/training'))
from campaign_cpt import _check_identity_contract,_check_checkpoint_contract
from campaign_cpt_parent import check_merged_cpt_parent,TOKENIZER_SHA,TOKENIZER_CONFIG_SHA
from campaign_cpt_data import validate_draw_schedule,CptDataError,validate_materialized_row
from bind_after_merge import bind,sha

def write(p,v):p.write_text(json.dumps(v,sort_keys=True)+'\n');return {'path':str(p),'sha256':sha(p)}
class TransitionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  self.old=json.loads((H.parent/'lead/r2-cpt-broad-a/recipe.json').read_text());self.oldrec=write(self.p/'previous-recipe.json',self.old)
  self.cp={'schema_version':1,'full':True,'step':750,'identity':self.old['identity'],'files':{'synthetic':{'bytes':1,'sha256':'a'*64}}}
  cpdir=self.p/'checkpoint-750';cpdir.mkdir();self.cprec=write(cpdir/'campaign-manifest.json',self.cp)
  inv=[{'path':'model.safetensors','bytes':100,'sha256':'1'*64}]
  self.m={'schema_version':'sepalith.merged-cpt.parent-manifest.v1','kind':'merged_cpt','cpt_identity':self.old['identity'],'cpt_checkpoint_manifest':self.cp,'cpt_checkpoint':str(cpdir),'checkpoint_manifest_sha256':self.cprec['sha256'],'recipe_sha256':self.oldrec['sha256'],'recipe_path':self.oldrec['path'],'source_cursor':12000,'source_script_sha256':sha(H/'source/experiments/training/merge_cpt_cpu.py'),'tokenizer_original_bytes_restored':True,'merge_verification':{'device':'cpu','dtype':'bfloat16','accumulation_dtype':'float32','rounding':'one final BF16 cast','adapter_tensors_exact':588,'independent_lora_matrix_merges_exact':294,'cuda_started':False},'weight_inventory':inv,'weight_inventory_sha256':hashlib.sha256(json.dumps(inv,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'merged_weights_sha256':'1'*64,'base_model_revision':self.old['identity']['parent']['revision'],'tokenizer':{'tokenizer_json_sha256':TOKENIZER_SHA,'tokenizer_config_sha256':TOKENIZER_CONFIG_SHA},'merged_model_path':'/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SYNTHETIC_NOT_CREATED','config_sha256':'2'*64,'generation_config_sha256':'3'*64}
  self.m['previous_sampler']={'consumed_draws':12000,'schedule_sha256':self.old['draw_schedule']['sha256'],'split_id':self.old['split_id']}
  self.mpath=self.p/'parent-manifest.preparation.json';write(self.mpath,self.m)
  template=json.loads((H/'recipe-template.json').read_text())
  self.r=bind(template,self.mpath,sha(self.mpath),json.loads((H/'source-manifest.json').read_text())['id'],self.p/'output',self.p/'archive','2026-09-14T06:15:00Z',5400)
 def refresh(self):
  rec=write(self.mpath,self.m);self.r['merged_cpt_parent']['manifest']=rec;self.r['identity']['parent']['merged_cpt_manifest_sha256']=rec['sha256']
 def reject(self):
  with self.assertRaises(ValueError):_check_identity_contract(self.r,self.r['parameters'])
 def test_actual_runner_recipe_argv_and_guards(self):
  from make_runner_recipe import make
  path=self.p/'bound-recipe.json';write(path,self.r);v=make(path)
  prepare=v['steps'][0]['argv'];guard=v['steps'][1]['argv']
  self.assertEqual(prepare[3],'{source}/experiments/training/campaign_launch.py');self.assertEqual(prepare[4],str(path))
  self.assertEqual(guard[guard.index('--seconds')+1],'5460');self.assertEqual(guard[guard.index('--minimum-free-mib')+1],'8192')
  self.assertEqual(guard[guard.index('--release-cache-file')+1],self.r['model_path']+'/model.safetensors')
  self.assertEqual(v['snapshot'],self.r['identity']['source']);self.assertEqual(v['env']['CUDA_VISIBLE_DEVICES'],'0')
 def test_metadata_happy_path_no_model_files(self):
  result=check_merged_cpt_parent(self.r);self.assertEqual(result['previous_source_cursor'],12000);self.assertFalse(result['model_bytes_verified']);self.assertFalse((self.p/'merged').exists());self.assertIsNone(self.r['resume_from']);self.assertFalse(self.r['launch_authorized'])
 def test_live_verification_requires_actual_model_bytes(self):
  with self.assertRaises(ValueError):check_merged_cpt_parent(self.r,verify_model_files=True)
 def test_old_midtrain_contract_unchanged(self):_check_identity_contract(self.old,self.old['parameters'])
 def test_unknown_policy(self):self.r['identity']['policy']['initialization']='anything';self.reject()
 def test_unbound_template_rejected(self):
  r=json.loads((H/'recipe-template.json').read_text())
  with self.assertRaises((ValueError,TypeError)):_check_identity_contract(r,r['parameters'])
 def test_previous_optimizer_cannot_resume(self):self.r['resume_from']=self.m['cpt_checkpoint'];self.reject()
 def test_full_checkpoint_required(self):self.m['cpt_checkpoint_manifest']=dict(self.cp,full=False);self.refresh();self.reject()
 def test_previous_identity_mismatch(self):self.m['cpt_identity']=dict(self.old['identity'],source='f'*64);self.refresh();self.reject()
 def test_wrong_manifest_kind(self):self.m['kind']='merged_sft';self.refresh();self.reject()
 def test_tokenizer_reserialization_rejected(self):self.m['tokenizer']['tokenizer_config_sha256']='0'*64;self.refresh();self.reject()
 def test_missing_original_byte_witness(self):self.m['tokenizer_original_bytes_restored']=False;self.refresh();self.reject()
 def test_wrong_model_file_pin(self):
  self.r['inputs'][-6]['sha256']='0'*64
  self.reject()
 def test_wrong_parent_weight_identity(self):self.r['identity']['parent']['weights_sha256']='0'*64;self.reject()
 def test_wrong_rounding_policy(self):self.m['merge_verification']['rounding']='delta BF16 then add';self.refresh();self.reject()
 def test_wrong_previous_sampler(self):self.m['previous_sampler']['consumed_draws']=11999;self.refresh();self.reject()
 def test_wrong_cursor(self):self.m['source_cursor']=11999;self.refresh();self.reject()
 def test_wrong_merge_source_even_if_binding_agrees(self):self.m['source_script_sha256']='a'*64;self.r['merged_cpt_parent']['merge_script_sha256']='a'*64;self.refresh();self.reject()
 def test_metadata_hash_drift(self):self.mpath.write_text('{}');self.reject()
 def test_changed_lr_rejected(self):self.r['parameters']['learning_rate']=2e-4;self.r['identity']['schedule']=copy.deepcopy(self.r['parameters']);self.reject()
 def test_checkpoint_last_step_in_horizon(self):_check_checkpoint_contract(self.r,self.r['parameters']);self.assertEqual(self.r['parameters']['max_steps'],1187)
 def test_no_framework_imports(self):self.assertFalse(set(sys.modules)&{'torch','transformers','unsloth','peft'})

class MergeSourceTests(unittest.TestCase):
 def test_accepted_float_merge_cast_order(self):
  import ast
  t=ast.parse((H/'source/experiments/training/merge_cpt_cpu.py').read_text());lines={}
  for n in ast.walk(t):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='model' and n.func.attr in ('float','merge_and_unload','to'):
    lines[n.func.attr]=n.lineno
  self.assertLess(lines['float'],lines['merge_and_unload']);self.assertLess(lines['merge_and_unload'],lines['to'])
 def test_double_rounding_counterexample_without_framework(self):
  import struct
  def bf(x):
   bits=struct.unpack('<I',struct.pack('<f',x))[0];bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
   return struct.unpack('<f',struct.pack('<I',bits))[0]
  self.assertNotEqual(bf(1.0+bf(0.00391)),bf(1.0+0.00391))

class ScheduleTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'id':str(i)} for i in range(31)];ids=[r['id'] for r in self.rows];random.Random(3407).shuffle(ids)
  self.s={'method':'one_pass_plus_named_replay_v1','seed':3407,'split_id':'synthetic','max_steps':2,'effective_batch':16,'token_rows_sha256':'a'*64,'row_ids':ids+ids[:1],'replay_row_ids':ids[:1]}
 def check(self):return validate_draw_schedule(self.s,self.rows,token_rows_sha256='a'*64,max_steps=2)
 def test_exact_complete_pass_with_named_replay(self):self.assertEqual(self.check()['draws'],32)
 def test_silent_omission_rejected(self):self.s['row_ids'][-2]=self.s['row_ids'][0];self.assertRaises(CptDataError,self.check)
 def test_wrong_replay_declaration(self):self.s['replay_row_ids']=[];self.assertRaises(CptDataError,self.check)
 def test_seed_order_changed(self):self.s['row_ids'][0],self.s['row_ids'][1]=self.s['row_ids'][1],self.s['row_ids'][0];self.assertRaises(CptDataError,self.check)
 def test_duplicate_source_ids_rejected(self):self.rows[-1]=self.rows[0];self.assertRaises(CptDataError,self.check)
 def test_original_method_remains_accepted(self):self.s['method']='sequential';self.check()
 def test_actual_global_schedule_denominator(self):
  e=json.loads((H/'exposure.json').read_text());self.assertEqual((e['rows'],e['draws'],e['steps'],e['omitted_rows']),(18991,18992,1187,0));self.assertEqual(e['replay_totals']['loss_tokens'],127)
if __name__=='__main__':
 if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 unittest.main(verbosity=2)
