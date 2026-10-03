from __future__ import annotations
import copy, json, tempfile
from pathlib import Path
import unittest
from sepalith.training.rl import full_weight_rl_production as p

def h(ch): return ch*64

def binding():
 return {
  'schema':'sepalith.rl11.full-weight-production-binding.v1','status':'root_admitted',
  'optimizer_updates':True,'reward_implementation':'CampaignPRM03Reward','synthetic_reward':False,
  'train_split_only':True,'sealed_final_access':False,'checkpoint_kind':'full_weights',
  'expected_parameter_tensors':381,'expected_trainable_parameters':2516756480,'peft_tensors':0,
  'geometry':{'candidate_count':4,'prompt_groups_per_update':1,'steps_per_generation':1},
  'gradient_accumulation_steps':1,'initial_policy_step':2,'initial_rollout_group_cursor':2,
  'sampling':{'do_sample':True,'seed':3407},'required_reward_families':['edit','no_op'],
  'signal_evidence':{'status':'root_admitted','groups_by_family':{'edit':12,'no_op':4},'varying_groups_by_family':{'edit':4,'no_op':1},'parser_infrastructure_failures':0},
  **{name:h(hex(i)[2:]) for i,name in enumerate(p.SHA_FIELDS,2)},
 }

def live(equal=False):
 inputs=[];records=[];rewards=[]
 for i in range(4):
  inputs.append({'id':'row-a','prompt':{'ids':[0,22]}})
  ids=[24+i,1];digest=p.ids_sha256(ids)
  records.append({'prompt_tokens':[0,22],'generated_tokens':ids,'terminal_reason':'eos'})
  rewards.append({'id':'row-a','reward':1.0 if equal else float(i),'output_ids_sha256':digest})
 return inputs,{'records':records},rewards

class RewardBase:
 def _calculate_rewards(self,inputs,prompts,completions,ids): return ['trl_rewards']
 def campaign_sampler_state(self): return {'source_draw_cursor':3}

class Reward:
 pass
Reward.__name__='CampaignPRM03Reward'

class ContractTrainer(p.ProductionRuntimeContractMixin,RewardBase):
 def __init__(self,b,rewards):
  self.state=type('S',(),{'global_step':2})();self.args=type('A',(),{'output_dir':tempfile.mkdtemp()})()
  self._campaign_reward=Reward();self._campaign_reward.last_records=rewards
  self.configure_production_contract(b)

class TestProductionDriver(unittest.TestCase):
 def test_live_prm03_generation_reward_join_and_flat_group(self):
  b=binding();inputs,generation,rewards=live(True);t=ContractTrainer(b,rewards);t._campaign_last_generation=generation
  self.assertEqual(t._calculate_rewards(inputs,None,None,[x['generated_tokens'] for x in generation['records']]),['trl_rewards'])
  self.assertEqual(t._campaign_verified_update['zero_advantage_groups'],1)
  self.assertEqual(t._campaign_verified_update['varying_reward_groups'],0)
  t.state.global_step=3
  state=t.campaign_sampler_state()
  self.assertEqual((state['rollout_buffer_state'],state['rollout_group_cursor'],state['policy_step']),('empty',3,3))
 def test_reward_id_mismatch_rejected_before_loss(self):
  b=binding();inputs,generation,rewards=live();rewards[1]['output_ids_sha256']=h('f')
  t=ContractTrainer(b,rewards);t._campaign_last_generation=generation
  with self.assertRaisesRegex(ValueError,'reward IDs'):t._calculate_rewards(inputs,None,None,[x['generated_tokens'] for x in generation['records']])
 def test_typed_binding_and_train_only_gates(self):
  for field,value in [('initial_policy_step',True),('initial_rollout_group_cursor',None),('synthetic_reward',True),('train_split_only',False),('peft_tensors',1)]:
   b=binding();b[field]=value
   with self.assertRaises(ValueError,msg=field):p.validate_production_binding(b)
 def test_group_prompt_geometry_and_cap_types(self):
  inputs,generation,rewards=live();inputs[1]['prompt']={'ids':[0,23]}
  with self.assertRaisesRegex(ValueError,'prompt envelopes'):p.build_live_groups(inputs=inputs,generation=generation,reward_records=rewards,completion_ids_list=[x['generated_tokens'] for x in generation['records']],candidate_count=4,first_group=2)
  inputs,generation,rewards=live();generation['records'][0]['terminal_reason']='length';generation['records'][0]['generated_tokens']=[24,25];rewards[0]['output_ids_sha256']=p.ids_sha256([24,25])
  groups,prompts=p.build_live_groups(inputs=inputs,generation=generation,reward_records=rewards,completion_ids_list=[x['generated_tokens'] for x in generation['records']],candidate_count=4,first_group=2)
  out=p.prepare_update(binding={
   'schema':'sepalith.rl11.full-weight-update-binding.v1','optimizer_updates':True,'objective':'trl-0.24-grpo-bnpo-group-beta0','candidate_count':4,
   'policy_step':2,'optimizer_global_step':2,'rollout_cursor':2,'rollout_complete':True,'same_process_live_policy_logprobs':True,
   'full_weight_checkpoint_kind':'full_weights','sampler_boundary':'one_generation_buffer_per_optimizer_update',**{n:binding()[n] for n in p.SHA_FIELDS}},groups=groups,prompt_ids_by_row=prompts,expected_first_group=2)
  self.assertTrue(out['rows'][0]['cap_hit'])
 def test_resume_requires_rng_and_exact_empty_cursor(self):
  b=binding()
  with tempfile.TemporaryDirectory() as td:
   cp=Path(td);state={'step':3,'sampler':{'rollout_buffer_state':'empty','rollout_group_cursor':3,'optimizer_global_step':3,'policy_step':3,'generation_policy_sha256':b['generation_policy_sha256']}}
   (cp/'campaign-state.json').write_text(json.dumps(state))
   files={n:{'bytes':1} for n in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','campaign-state.json')}
   self.assertEqual(p.validate_resume_boundary(cp,b,{'checkpoint_kind':'full_weights','full':True,'files':files})['rollout_group_cursor'],3)
   bad=copy.deepcopy(files);bad.pop('rng_state.pth')
   with self.assertRaisesRegex(ValueError,'rng_state'):p.validate_resume_boundary(cp,b,{'checkpoint_kind':'full_weights','full':True,'files':bad})
   state['sampler']['rollout_buffer_state']='partial';(cp/'campaign-state.json').write_text(json.dumps(state))
   with self.assertRaisesRegex(ValueError,'inside a rollout'):p.validate_resume_boundary(cp,b,{'checkpoint_kind':'full_weights','full':True,'files':files})
 def test_mro_dense_optimizer_precedes_campaign_base(self):
  class Base: pass
  cls=p.production_trainer_class(Base)
  names=[x.__name__ for x in cls.mro()]
  self.assertLess(names.index('ProductionRuntimeContractMixin'),names.index('FullWeightOptimizerTrainerMixin'))
  self.assertLess(names.index('FullWeightOptimizerTrainerMixin'),names.index('Base'))

 def test_campaign_recipe_accepts_dense_parent_and_rejects_lora_parent(self):
  from sepalith.training.rl import campaign_rl_train as rl
  ids=['row-a']
  identity={
   'parent':{'kind':'full_weight_edit_sft_checkpoint','manifest_sha256':h('a'),'model_weights_sha256':h('b')},
   'tokenizer':{'vocab_size':130560,'bos_id':0,'eos_id':1,'pad_id':1,'native_eog_ids':[1,130073]},
   'renderer':{'renderer_id':rl.RENDERER_ID,'tokenization_policy':rl.TOKENIZATION_POLICY,'terminal':rl.TERMINAL,'no_edit':'[NO_EDIT]'},
   'data':{'split':'train','admission_status':'admitted','rows_sha256':h('1'),'context_sha256':h('2'),'sidecar_artifact_sha256':h('2'),'selected_ids_sha256':h('3'),'ordered_ids_sha256':h('4'),'row_identity_sha256':h('5'),'reward_buffer_manifest_sha256':h('6'),'reward_buffer_sidecar_sha256':h('7'),'row_count':1,'selected_ids':ids,'row_identities':[{'id':'row-a'}]},
   'source':{'manifest_sha256':h('8')},
   'policy':{'checkpoint_kind':'full_weights','expected_parameter_tensors':381,'expected_trainable_parameters':2516756480,'peft_tensors':0,'prompt_max_tokens':2048,'completion_max_tokens':192,'generation_groups_per_call':1,'per_device_train_batch_size':8,'gradient_accumulation_steps':4,'candidate_count':4,'rollout_rows_per_update':32,'loss_type':'bnpo','scale_rewards':'group','beta':0,'reward_policy_id':'protocol_restraint_syntax_coverage_v2','reward_source_sha256':h('9'),'reward_parse_harness_sha256':h('a'),'sampling':{'do_sample':True,'temperature':0.7,'top_p':0.95,'repetition_penalty':1.0}},
   'schedule':{'seed':3407,'sampler_id':rl.CampaignRepeatSampler.SAMPLER_ID,'generation_batch_size':32,'steps_per_generation':4,'num_iterations':1},
  }
  recipe={'schema_version':rl.TRAIN_SCHEMA_VERSION,'training_mode':'full_weights','identity':identity,'generation_groups_per_call':1,'generation_kwargs':identity['policy']['sampling'],'data':{'rows_path':'/x','rows_sha256':h('1'),'sidecar_path':'/y','sidecar_artifact_sha256':h('2'),'selected_ids_path':'/z','selected_ids_sha256':h('3'),'reward_buffer_manifest_path':'/r','reward_buffer_manifest_sha256':h('6')},'reward_parse':{'operation':'base::parse_only','generated_r_executed':False,'harness_sha256':h('a')}}
  checked,geometry=rl.validate_rl_recipe(recipe)
  self.assertEqual((checked['parent']['kind'],geometry.prompt_groups_per_update),('full_weight_edit_sft_checkpoint',8))
  recipe=copy.deepcopy(recipe);recipe['identity']['parent']['kind']='merged_sft'
  with self.assertRaisesRegex(ValueError,'full_weight_edit_sft_checkpoint'):rl.validate_rl_recipe(recipe)

 def test_train_wrapper_uses_only_verified_resume_path(self):
  class T:
   _campaign_production_binding={}
   _campaign_verified_resume_from='/verified/checkpoint-3'
   def train(self,**kwargs):self.kwargs=kwargs;return 'trained'
  t=T();self.assertEqual(p.train_production(t),'trained')
  self.assertEqual(t.kwargs,{'resume_from_checkpoint':'/verified/checkpoint-3'})
  class U: pass
  with self.assertRaisesRegex(ValueError,'no production binding'):p.train_production(U())

 def test_root_reproduction_changed_generated_prompt_rejected(self):
  b=binding();inputs,generation,rewards=live();generation['records'][0]['prompt_tokens']=[0,999]
  t=ContractTrainer(b,rewards);t._campaign_last_generation=generation
  with self.assertRaisesRegex(ValueError,'generation prompt differs'):
   t._calculate_rewards(inputs,None,None,[x['generated_tokens'] for x in generation['records']])
 def test_passed_trl_completion_ids_and_terminal_reason_are_exact(self):
  b=binding();inputs,generation,rewards=live();t=ContractTrainer(b,rewards);t._campaign_last_generation=generation
  passed=[list(x['generated_tokens']) for x in generation['records']];passed[2]=[31,1]
  with self.assertRaisesRegex(ValueError,'TRL completion IDs differ'):t._calculate_rewards(inputs,None,None,passed)
  inputs,generation,rewards=live();generation['records'][0]['terminal_reason']='unknown';t=ContractTrainer(b,rewards);t._campaign_last_generation=generation
  with self.assertRaisesRegex(ValueError,'terminal reason'):t._calculate_rewards(inputs,None,None,[x['generated_tokens'] for x in generation['records']])

 def test_real_fixed_id_mixin_records_join_to_contract(self):
  import contextlib,torch;from sepalith.training.rl import campaign_rl_train as rl
  class Model:
   is_gradient_checkpointing=False
   def eval(self):return self
   def generate(self,input_ids=None,attention_mask=None,**kwargs):
    tails=torch.tensor([[24+i,1] for i in range(input_ids.shape[0])],dtype=torch.long)
    return type('O',(),{'sequences':torch.cat((input_ids,tails),dim=1)})()
  class Generator(rl.FixedIDGRPOTrainerMixin):pass
  g=Generator();g.num_generations=4;g.accelerator=type('A',(),{'device':torch.device('cpu'),'unwrap_model':lambda self,m:m,'state':type('State',(),{'deepspeed_plugin':None})()})();g.model=g.model_wrapped=Model();g.processing_class=object();g.state=type('S',(),{'global_step':2,'num_input_tokens_seen':0})();g.args=type('Args',(),{'ds3_gather_for_generation':False})();g._step=0
  g.configure_campaign_runtime(generation_guard_factory=lambda model:contextlib.nullcontext(),prompt_max_tokens=8,completion_max_tokens=2,context_max_tokens=10,generation_kwargs={'do_sample':True,'temperature':0.7,'top_p':0.95,'repetition_penalty':1.0},generation_groups_per_call=1)
  prompt_ids,completion_ids,_,_=g._generate_single_turn([{'ids':[0,22]} for _ in range(4)])
  inputs=[{'id':'row-a','prompt':{'ids':x}} for x in prompt_ids];rewards=[]
  for ids in completion_ids:rewards.append({'id':'row-a','reward':1.0,'output_ids_sha256':p.ids_sha256(ids)})
  t=ContractTrainer(binding(),rewards);t._campaign_last_generation=g._campaign_last_generation
  self.assertEqual(t._calculate_rewards(inputs,None,None,completion_ids),['trl_rewards'])

if __name__=='__main__':unittest.main()
