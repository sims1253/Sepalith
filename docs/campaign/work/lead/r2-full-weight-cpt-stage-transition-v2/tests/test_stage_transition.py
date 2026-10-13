import copy,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]
TRAINING=PACKET/'source/experiments/training';sys.path.insert(0,str(TRAINING))
import torch
from full_weight_optimizer import FullWeightCompositeOptimizer,OptimizerConfig
import full_weight_cpt_trainer as trainer
from stage_transition_contract import inspect_transition,source_identity,stage_cursor_from_checkpoint,sha256
from streaming_trainer_adapter import StageCursorView

OPT={'arm':'aurora_mix','hidden_lr':3e-6,'side_lr':3e-7,'weight_decay':0.0,'momentum':0.95,'adam_betas':[0.9,0.999],'adam_eps':1e-8,'ns_steps':5,'aurora_K':2,'aurora_beta':0.5,'rms_scale':0.2,'state_dtype':'float32','bf16_update_policy':'stochastic_round','stochastic_round_chunk_elements':1048576}

class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR','/mnt/e'));self.d=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def fixture(self):
  base=Path(tempfile.mkdtemp(dir=self.d));srcdir=base/'source';srcdir.mkdir();unit=srcdir/'unit.py';unit.write_text('X=1\n');sm=srcdir/'source-manifest.json';sm.write_text(json.dumps({'schema':'sepalith.full-weight-cpt-representative-trainer-source.v3','files':[{'path':'unit.py','bytes':unit.stat().st_size,'sha256':sha256(unit)}]}))
  sr={'schema':'sepalith.sft11.full-weight-cpt-representative-bound.v3','source':{'manifest_path':str(sm.resolve()),'manifest_sha256':sha256(sm)},'parent':{'candidate_id':'p','files':{'model.safetensors':'1'*64,'tokenizer.json':'2'*64},'saved_precision':{'fp32_tensors':0,'fp32_elements':0}},'cohort':{'id':'old','rows':{'sha256':'3'*64},'max_sequence_tokens':16384},'runtime':{'optimizer':OPT,'max_steps':66,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_ratio':2/66,'checkpoint_every':24,'mandatory_stop_step':24}}
  srp=base/'source-recipe.json';srp.write_text(json.dumps(sr));cp=base/'checkpoint-66';cp.mkdir()
  names=('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','tokenizer.json','config.json','generation_config.json','tokenizer_config.json')
  for name in names:
   (cp/name).write_text(json.dumps({'global_step':66}) if name=='trainer_state.json' else name)
  files={n:{'bytes':(cp/n).stat().st_size,'sha256':sha256(cp/n)} for n in names};ident=source_identity(sr)
  state={'checkpoint_kind':'full_weights','full':True,'identity':ident,'step':66,'sampler':{'cursor':1056,'draw_schedule_sha256':'4'*64,'effective_batch':16,'global_step':66,'ignore_data_skip':False,'method':'sequential_frozen_draw_schedule'}};(cp/'campaign-state.json').write_text(json.dumps(state));files['campaign-state.json']={'bytes':(cp/'campaign-state.json').stat().st_size,'sha256':sha256(cp/'campaign-state.json')}
  manifest={'schema_version':1,'step':66,'full':True,'identity':ident,'files':files,'checkpoint_kind':'full_weights'};mp=cp/'campaign-manifest.json';mp.write_text(json.dumps(manifest))
  decision={'checkpoint':str(cp.resolve()),'checkpoint_manifest_sha256':sha256(mp),'weights_sha256':files['model.safetensors']['sha256']};dp=base/'decision.json';dp.write_text(json.dumps(decision))
  recipe={'cohort':{'id':'new','rows':{'sha256':'5'*64},'streaming_cache':{'manifest_sha256':'6'*64},'updates':100},'runtime':{'optimizer':copy.deepcopy(OPT),'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'max_steps':166},'transition':{'source_recipe':{'path':str(srp),'sha256':sha256(srp)},'source_manifest':{'path':str(sm.resolve()),'sha256':sha256(sm)},'source_checkpoint':{'path':str(cp),'manifest_sha256':sha256(mp),'candidate_id':'rep66'},'source_decision':{'path':str(dp),'sha256':sha256(dp)},'global_optimizer_step_offset':66},'parent':{'kind':'full_weight_stage_checkpoint','path':str(cp),'files':{'model.safetensors':files['model.safetensors']['sha256'],'tokenizer.json':files['tokenizer.json']['sha256']},'saved_precision':{'fp32_tensors':0,'fp32_elements':0}}}
  return recipe,srp,cp
 def test_source_checkpoint_recipe_and_new_corpus_are_cross_checked(self):
  recipe,_,_=self.fixture();e=inspect_transition(recipe);self.assertEqual(e['source_step'],66);self.assertEqual(e['destination_initial_cursor'],0)
  bad=copy.deepcopy(recipe);bad['runtime']['optimizer']['hidden_lr']=1e-5
  with self.assertRaisesRegex(ValueError,'optimizer configuration'):inspect_transition(bad)
  bad=copy.deepcopy(recipe);bad['cohort']['rows']['sha256']='3'*64
  with self.assertRaisesRegex(ValueError,'distinct destination'):inspect_transition(bad)
 def test_manifest_identity_and_decision_mismatch_fail_closed(self):
  recipe,srp,cp=self.fixture();source=json.loads(srp.read_text());source['runtime']['learning_rate']=9e-6;srp.write_text(json.dumps(source));recipe['transition']['source_recipe']['sha256']=sha256(srp)
  with self.assertRaisesRegex(ValueError,'identity differs'):inspect_transition(recipe)
  recipe,srp,cp=self.fixture();decision=Path(recipe['transition']['source_decision']['path']);x=json.loads(decision.read_text());x['weights_sha256']='0'*64;decision.write_text(json.dumps(x));recipe['transition']['source_decision']['sha256']=sha256(decision)
  with self.assertRaisesRegex(ValueError,'decision/checkpoint'):inspect_transition(recipe)
 def test_stage_sampler_starts_zero_and_destination_resume_uses_local_cursor(self):
  self.assertEqual(stage_cursor_from_checkpoint({'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':'a','stage_cursor':384,'cursor':384,'global_step':90,'ignore_data_skip':True}},66,'a'),384)
  with self.assertRaisesRegex(ValueError,'schedule'):stage_cursor_from_checkpoint({'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':'x','stage_cursor':0,'global_step':66,'ignore_data_skip':True}},66,'a')
  args=trainer.training_arguments_kwargs({'seed':1,'runtime':{'micro_batch':1,'gradient_accumulation':16,'max_steps':166,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'checkpoint_every':24}},self.d/'out');self.assertTrue(args['ignore_data_skip']);self.assertEqual(args['max_steps'],166);self.assertEqual(args['warmup_steps'],2)
  self.assertEqual(trainer.milestone_action(90,66,{'mandatory_stop_step':90,'max_steps':166,'checkpoint_every':24},66),{'save':True,'stop':True,'cursor':384})
  self.assertEqual(trainer.milestone_action(114,90,{'mandatory_stop_step':90,'max_steps':166,'checkpoint_every':24},66),{'save':True,'stop':False,'cursor':768})
  class Base:
   def __len__(self):return 10
   def __getitem__(self,i):return {'_draw_position':i}
   def close(self):pass
  view=StageCursorView(Base(),4);self.assertEqual(len(view),6);self.assertEqual([view[i]['_draw_position'] for i in range(3)],[4,5,6])
 def test_tiny_hybrid_optimizer_scheduler_rng_transition_matches_uninterrupted(self):
  cfg=OptimizerConfig(arm='aurora_mix',hidden_lr=3e-3,side_lr=3e-4,stochastic_round_chunk_elements=32)
  manifest={'ordered_rows_sha256':'f'*64}
  def make():
   hidden=torch.nn.Parameter(torch.tensor([[1,2],[3,4]],dtype=torch.bfloat16));side=torch.nn.Parameter(torch.tensor([1,2],dtype=torch.bfloat16));opt=FullWeightCompositeOptimizer([('h',hidden,'muon'),('s',side,'adamw')],manifest,cfg);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:min((n+1)/2,1.0));return [hidden,side],opt,sched
  grads=[(torch.tensor([[.2,-.1],[.3,.4]]),torch.tensor([.1,-.2])),(torch.tensor([[.1,.2],[-.2,.3]]),torch.tensor([-.3,.2])),(torch.tensor([[.4,.1],[.2,-.1]]),torch.tensor([.2,.1]))]
  def update(ps,opt,sched,g):
   for p,v in zip(ps,g):p.grad=v.to(torch.bfloat16)
   opt.step();sched.step();opt.zero_grad(set_to_none=True)
  torch.manual_seed(71);direct,opt_d,sch_d=make()
  for g in grads:update(direct,opt_d,sch_d,g)
  torch.manual_seed(71);first,opt_a,sch_a=make();update(first,opt_a,sch_a,grads[0]);saved=([p.detach().clone() for p in first],copy.deepcopy(opt_a.state_dict()),copy.deepcopy(sch_a.state_dict()),torch.get_rng_state().clone())
  resumed,opt_b,sch_b=make()
  with torch.no_grad():
   for p,v in zip(resumed,saved[0]):p.copy_(v)
  opt_b.load_state_dict(saved[1]);sch_b.load_state_dict(saved[2]);torch.set_rng_state(saved[3])
  for g in grads[1:]:update(resumed,opt_b,sch_b,g)
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(direct,resumed)));self.assertEqual(sch_d.state_dict(),sch_b.state_dict())
  for state in opt_b.state.values():
   for k in ('momentum_buffer','exp_avg','exp_avg_sq'):
    if k in state:self.assertEqual(state[k].dtype,torch.float32)
 def test_root_can_admit_an_exact_later_stage_checkpoint_without_dry_restart(self):
  recipe={'transition':{'global_optimizer_step_offset':66},'runtime':{'mandatory_stop_step':90,'max_steps':166},'cohort':{'draw_schedule':{'sha256':'d'*64}}};rp=self.d/'bound.json';rp.write_text(json.dumps(recipe));cp=self.d/'checkpoint-114';cp.mkdir();(cp/'campaign-manifest.json').write_text(json.dumps({'step':114}));(cp/'campaign-state.json').write_text(json.dumps({'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':'d'*64,'stage_cursor':768,'global_step':114,'ignore_data_skip':True}}));ap=self.d/'continue.json';ap.write_text(json.dumps({'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'admitted','decision':'continue','bound_recipe_sha256':sha256(rp),'checkpoint':str(cp.resolve()),'checkpoint_manifest_sha256':sha256(cp/'campaign-manifest.json'),'step':114}));self.assertEqual(trainer.validate_continuation(rp,recipe,cp,ap)['sha256'],sha256(ap))
  state=json.loads((cp/'campaign-state.json').read_text());state['sampler']['stage_cursor']=0;(cp/'campaign-state.json').write_text(json.dumps(state))
  with self.assertRaisesRegex(ValueError,'cursor/global step'):trainer.validate_continuation(rp,recipe,cp,ap)
 def test_template_and_commands_are_unlaunchable(self):
  t=json.loads((PACKET/'recipe.template.json').read_text());a=json.loads((PACKET/'root-admission.template.json').read_text());c=json.loads((PACKET/'root-commands.json').read_text());self.assertIsNone(t['transition']);self.assertFalse(t['launch_authorized']);self.assertFalse(a['launch_authorized']);self.assertFalse(c['execution_authorized']);self.assertIn('--stage-transition-admission',c['initial_transition_root_guard_required'])

if __name__=='__main__':unittest.main(verbosity=2)
