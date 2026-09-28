import copy, importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parents[1]
p=HERE/'source'/'experiments'/'training'/'runtime_update_contract.py'
spec=importlib.util.spec_from_file_location('contract',p); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def h(ch):return ch*64
def binding():
 d={'schema':'sepalith.rl11.full-weight-update-binding.v1','optimizer_updates':True,'objective':'trl-0.24-grpo-bnpo-group-beta0','candidate_count':4,'policy_step':2,'optimizer_global_step':2,'rollout_cursor':2,'rollout_complete':True,'same_process_live_policy_logprobs':True,'full_weight_checkpoint_kind':'full_weights','sampler_boundary':'one_generation_buffer_per_optimizer_update'}
 for i,k in enumerate(('source_manifest_sha256','model_manifest_sha256','model_weights_sha256','tokenizer_json_sha256','reward_buffer_manifest_sha256','prompt_context_manifest_sha256','generation_policy_sha256','optimizer_dispatch_sha256'),2):d[k]=h(hex(i)[2:])
 return d
def group(equal=False):
 prompt=[0,22]; gs=[];rs=[]
 for i in range(4):
  ids=[24+i,1]; digest=c.ids_sha256(ids)
  gs.append({'group_index':5,'row_id':'r','candidate_index':i,'prompt_ids_sha256':c.ids_sha256(prompt),'generated_ids':ids,'generated_ids_sha256':digest,'cap_hit':False})
  rs.append({'group_index':5,'row_id':'r','candidate_index':i,'output_ids_sha256':digest,'reward':1.0 if equal else float(i),'parser_infrastructure_failure':False})
 return {'group_index':5,'row_id':'r','generations':gs,'rewards':rs}
class TestContract(unittest.TestCase):
 def test_valid_and_flat_group_is_valid_zero_advantage(self):
  x=c.prepare_update(binding=binding(),groups=[group(True)],prompt_ids_by_row={'r':[0,22]},expected_first_group=5)
  self.assertEqual((x['varying_reward_groups'],x['zero_advantage_groups']),(0,1))
 def test_missing_and_bool_step_rejected(self):
  for value in (None,True,-1):
   b=binding(); b['policy_step']=value
   with self.assertRaisesRegex(ValueError,'policy_step'):c.validate_update_binding(b)
 def test_prompt_bool_out_of_range_and_multiple_bos_rejected(self):
  for prompt in ([0,True,22],[0,130560],[0,22,0]):
   with self.assertRaises(ValueError):c.prepare_update(binding=binding(),groups=[group()],prompt_ids_by_row={'r':prompt},expected_first_group=5)
 def test_cap_hit_must_be_bool(self):
  x=group();x['generations'][0]['cap_hit']=None
  with self.assertRaisesRegex(ValueError,'cap_hit'):c.prepare_update(binding=binding(),groups=[x],prompt_ids_by_row={'r':[0,22]},expected_first_group=5)
 def test_hash_join_and_complete_group(self):
  x=group();x['rewards'][2]['output_ids_sha256']=h('f')
  with self.assertRaisesRegex(ValueError,'reward IDs'):c.prepare_update(binding=binding(),groups=[x],prompt_ids_by_row={'r':[0,22]},expected_first_group=5)
  x=group();x['generations'].pop()
  with self.assertRaisesRegex(ValueError,'incomplete'):c.prepare_update(binding=binding(),groups=[x],prompt_ids_by_row={'r':[0,22]},expected_first_group=5)
if __name__=='__main__':unittest.main()
