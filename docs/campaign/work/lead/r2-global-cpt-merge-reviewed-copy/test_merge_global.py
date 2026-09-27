import ast,copy,importlib.util,json,sys,unittest
from pathlib import Path
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W))
import merge_global_cpt_cpu as merge
OLD=W.parent/'r2-cpt-global-stage-preparation-v1'
sys.path.append(str(OLD))
spec=importlib.util.spec_from_file_location('upstream_transition_tests',OLD/'test_global_stage.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

class GlobalMetadataTests(unittest.TestCase):
 def setUp(self):
  self.fixture=prior.TransitionTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.r=copy.deepcopy(self.fixture.r);self.r['identity']['source']=merge.GLOBAL_SOURCE
  self.cp={'full':True,'step':250,'identity':copy.deepcopy(self.r['identity'])}
  self.state={'full':True,'step':250,'identity':copy.deepcopy(self.r['identity']),'sampler':{'consumed_draws':4000,'schedule_sha256':merge.GLOBAL_DRAWS,'split_id':self.r['split_id']}}
 def check(self):return merge.validate_global_metadata(self.r,self.cp,self.state)
 def reject(self):
  with self.assertRaises((ValueError,KeyError)):self.check()
 def test_valid_complete_lineage_without_model(self):self.assertEqual(self.check()['consumed_draws'],4000)
 def test_unknown_source(self):self.r['identity']['source']='f'*64;self.reject()
 def test_original_source_rejected(self):self.r['identity']['source']='5149a4065285c681c2230352e5847b861e2ad2e416c8c263131a557e48bd23ca';self.reject()
 def test_wrong_initialization(self):self.r['identity']['policy']['initialization']='new_lora_on_midtrain';self.reject()
 def test_wrong_training_rows(self):self.r['train_rows']['sha256']='a'*64;self.reject()
 def test_wrong_schedule(self):self.r['draw_schedule']['sha256']='a'*64;self.reject()
 def test_wrong_validation(self):self.r['validation_rows']['sha256']='a'*64;self.reject()
 def test_wrong_horizon(self):self.r['parameters']['max_steps']=1188;self.reject()
 def test_partial_checkpoint(self):self.cp['full']=False;self.reject()
 def test_wrong_checkpoint_identity(self):self.cp['identity']['source']='a'*64;self.reject()
 def test_wrong_state_identity(self):self.state['identity']['source']='a'*64;self.reject()
 def test_wrong_state_step(self):self.state['step']=249;self.reject()
 def test_partial_state(self):self.state['full']=False;self.reject()
 def test_wrong_cursor(self):self.state['sampler']['consumed_draws']=3999;self.reject()
 def test_wrong_sampler_schedule(self):self.state['sampler']['schedule_sha256']='a'*64;self.reject()
 def test_wrong_sampler_split(self):self.state['sampler']['split_id']='other';self.reject()
 def test_beyond_horizon(self):self.cp['step']=1188;self.reject()
 def test_upstream_metadata_tamper(self):self.fixture.mpath.write_text('{}');self.reject()
 def test_upstream_merge_hash_tamper(self):self.r['merged_cpt_parent']['merge_script_sha256']='a'*64;self.reject()
 def test_no_framework_imports(self):self.assertFalse(set(sys.modules)&{'torch','transformers','peft','unsloth'})

class UnchangedMergeTests(unittest.TestCase):
 def test_numerical_and_tokenizer_body_exact(self):
  old=(OLD/'source/experiments/training/merge_cpt_cpu.py').read_text()
  new=(W/'merge_global_cpt_cpu.py').read_text()
  a='    import torch\n';b='    manifest = {'
  self.assertEqual(old[old.index(a):old.index(b)],new[new.index(a):new.index(b)])
 def test_all_frozen_source_dependency_hashes_checked(self):self.assertEqual(len(merge.DEPENDENCY_PINS),14)
 def test_compile(self):compile((W/'merge_global_cpt_cpu.py').read_text(),str(W/'merge_global_cpt_cpu.py'),'exec')

if __name__=='__main__':unittest.main(verbosity=2)
