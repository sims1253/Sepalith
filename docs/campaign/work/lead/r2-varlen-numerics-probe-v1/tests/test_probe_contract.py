import copy,importlib.util,json,sys,unittest
from pathlib import Path
PKT=Path(__file__).parents[1];sys.path.insert(0,str(PKT/'source'))
from verify_probe_report import verify

def fixture():
 metric={'max_abs':0.0,'max_relative':0.0,'relative_l2':0.0,'reference_max_abs':1.0,'elements':4}
 per_tensor={f'x{i}':metric for i in range(21)}
 gradients={'min':0.0,'median':0.005,'mean':0.005,'max':0.02,'per_tensor':per_tensor}
 repeat={'min':0.0,'median':0.0,'mean':0.0,'max':0.0,'per_tensor':per_tensor}
 return {'schema':'sepalith.sft11.varlen-numerics-probe.v1','status':'measurement_complete_root_parity_decision_required',
 'model_manifest_sha256':'6ae1d9cef615d179d4e55b7da7f68cc67b92239ffce5daeb8852c547c6dfc15d','model_weights_sha256':'aac456d2481869d1cec9c6e5e693c8807068ecb8a981d78c1384dbb0d4e39701','rows_sha256':'96c5e875e473e53f941318bfd8ba1b2dff8196590e87eb1bfb9c35ae0dc77c4b','model_file_identity_before':{'dev':1,'ino':2,'size':3,'mtime_ns':4,'ctime_ns':5},'model_file_identity_after':{'dev':1,'ino':2,'size':3,'mtime_ns':4,'ctime_ns':5},'no_optimizer_created':True,'no_optimizer_step':True,'contract':{'loss_denominator':9681,'physical_groups':1,'rows':16,'tokens':9700},'saved_precision':{'fp32_tensors_restored':85},'cuda_peak':{'allocated_bytes':1,'reserved_bytes':2},'result':{'backend':'xformers','cross_document_isolation':metric,'standalone_loss':1.0,'standalone_repeat_loss':1.0,'packed_loss':1.0,'ordinary_padded_batch_loss':1.0,'layer0_attribution':{k:metric for k in ('input_rmsnorm','q_proj','k_proj','v_proj','gate_proj','up_proj','q_proj_fp32')},'held_qkv_attention':metric,'gradient_summaries':{'same_mode_repeat':repeat,'packed':copy.deepcopy(gradients),'ordinary_padded_batch':copy.deepcopy(gradients)}}}
class Tests(unittest.TestCase):
 def test_valid_pass(self):self.assertTrue(verify(fixture())['production_varlen_pass'])
 def test_material_gradient_fails(self):
  v=fixture();v['result']['gradient_summaries']['packed']['max']=.25;self.assertFalse(verify(v)['production_varlen_pass'])
 def test_mutation_and_optimizer_rejected(self):
  v=fixture();v['model_file_identity_after']['ctime_ns']=6
  with self.assertRaises(ValueError):verify(v)
  v=fixture();v['no_optimizer_created']=False
  with self.assertRaises(ValueError):verify(v)
 def test_source_compiles(self):
  for path in (PKT/'source').glob('*.py'):
   compile(path.read_text(),str(path),'exec')
if __name__=='__main__':unittest.main()
