import copy,json,os,sys,tempfile,unittest
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACKET/'source/experiments/training'))
from stage_transition_contract import inspect_transition,source_identity,sha256
OPT={'arm':'aurora_mix','hidden_lr':3e-6,'side_lr':3e-7,'weight_decay':0.0,'momentum':0.95,'adam_betas':[0.9,0.999],'adam_eps':1e-8,'ns_steps':5,'aurora_K':2,'aurora_beta':0.5,'rms_scale':0.2,'state_dtype':'float32','bf16_update_policy':'stochastic_round','stochastic_round_chunk_elements':1048576}
class Recursive(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR','/mnt/e'));self.d=Path(self.t.name);self.n=0
 def tearDown(self):self.t.cleanup()
 def fixture(self):
  self.n+=1;base=self.d/str(self.n);base.mkdir();source=base/'closure';source.mkdir();unit=source/'unit.py';unit.write_text('X=2\n');sm=source/'source-manifest.json';sm.write_text(json.dumps({'files':[{'path':'unit.py','bytes':unit.stat().st_size,'sha256':sha256(unit)}]}))
  prior={'schema':'sepalith.sft11.full-weight-cpt-stage-transition-bound.v1','source':{'manifest_path':str(sm.resolve()),'manifest_sha256':sha256(sm)},'parent':{'candidate_id':'rep66','files':{'model.safetensors':'1'*64,'tokenizer.json':'2'*64},'saved_precision':{'fp32_tensors':0,'fp32_elements':0}},'cohort':{'id':'corpusA','rows':{'sha256':'3'*64},'streaming_cache':{'manifest_sha256':'4'*64},'draw_schedule':{'sha256':'5'*64},'updates':100,'max_sequence_tokens':16384},'transition':{'global_optimizer_step_offset':66,'source_checkpoint':{'manifest_sha256':'6'*64}},'runtime':{'optimizer':copy.deepcopy(OPT),'max_steps':166,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'checkpoint_every':24,'mandatory_stop_step':90,'evaluation_steps':[90,166],'selected_milestones':[90,166]}}
  sr=base/'prior.json';sr.write_text(json.dumps(prior));cp=base/'checkpoint-90';cp.mkdir()
  names=('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','tokenizer.json','config.json','generation_config.json','tokenizer_config.json')
  for n in names:(cp/n).write_text(json.dumps({'global_step':90}) if n=='trainer_state.json' else n)
  files={n:{'bytes':(cp/n).stat().st_size,'sha256':sha256(cp/n)} for n in names};ident=source_identity(prior)
  state={'checkpoint_kind':'full_weights','full':True,'identity':ident,'step':90,'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':'5'*64,'stage_cursor':384,'cursor':384,'global_step':90,'effective_batch':16,'ignore_data_skip':True}}
  (cp/'campaign-state.json').write_text(json.dumps(state));files['campaign-state.json']={'bytes':(cp/'campaign-state.json').stat().st_size,'sha256':sha256(cp/'campaign-state.json')}
  manifest={'schema_version':1,'step':90,'full':True,'identity':ident,'files':files,'checkpoint_kind':'full_weights'};mp=cp/'campaign-manifest.json';mp.write_text(json.dumps(manifest))
  decision={'checkpoint':str(cp.resolve()),'checkpoint_manifest_sha256':sha256(mp),'weights_sha256':files['model.safetensors']['sha256']};dp=base/'decision.json';dp.write_text(json.dumps(decision))
  dest={'cohort':{'id':'backfill','rows':{'sha256':'7'*64},'streaming_cache':{'manifest_sha256':'8'*64},'updates':10},'runtime':{'optimizer':copy.deepcopy(OPT),'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'max_steps':100},'transition':{'source_recipe':{'path':str(sr),'sha256':sha256(sr)},'source_manifest':{'path':str(sm.resolve()),'sha256':sha256(sm)},'source_checkpoint':{'path':str(cp),'manifest_sha256':sha256(mp),'candidate_id':'stageA90'},'source_decision':{'path':str(dp),'sha256':sha256(dp)},'global_optimizer_step_offset':90},'parent':{'kind':'full_weight_stage_checkpoint','path':str(cp),'files':{'model.safetensors':files['model.safetensors']['sha256'],'tokenizer.json':files['tokenizer.json']['sha256']},'saved_precision':{'fp32_tensors':0,'fp32_elements':0}}}
  return dest,sr,cp
 def test_prior_transition_checkpoint_can_start_fresh_backfill_cursor(self):
  d,_,_=self.fixture();x=inspect_transition(d);self.assertEqual((x['source_step'],x['source_recipe_schema'],x['destination_initial_cursor']),(90,'sepalith.sft11.full-weight-cpt-stage-transition-bound.v1',0));self.assertEqual(x['source_sampler_provenance']['stage_cursor'],384)
 def test_wrong_prior_offset_or_cursor_fails(self):
  for field,value,pattern in [('global_optimizer_step_offset',65,'offset'),('stage_cursor',368,'cursor')]:
   d,sr,cp=self.fixture()
   state=json.loads((cp/'campaign-state.json').read_text());state['sampler'][field]=value;(cp/'campaign-state.json').write_text(json.dumps(state));m=json.loads((cp/'campaign-manifest.json').read_text());m['files']['campaign-state.json']={'bytes':(cp/'campaign-state.json').stat().st_size,'sha256':sha256(cp/'campaign-state.json')};(cp/'campaign-manifest.json').write_text(json.dumps(m));d['transition']['source_checkpoint']['manifest_sha256']=sha256(cp/'campaign-manifest.json');dec=Path(d['transition']['source_decision']['path']);x=json.loads(dec.read_text());x['checkpoint_manifest_sha256']=d['transition']['source_checkpoint']['manifest_sha256'];dec.write_text(json.dumps(x));d['transition']['source_decision']['sha256']=sha256(dec)
   with self.assertRaisesRegex(ValueError,pattern):inspect_transition(d)
 def test_wrong_source_step_or_identity_fails(self):
  d,sr,cp=self.fixture();d['transition']['global_optimizer_step_offset']=91
  with self.assertRaisesRegex(ValueError,'global optimizer step'):inspect_transition(d)
  d,sr,cp=self.fixture();m=json.loads((cp/'campaign-manifest.json').read_text());m['identity']['data']['rows_sha256']='f'*64;(cp/'campaign-manifest.json').write_text(json.dumps(m));d['transition']['source_checkpoint']['manifest_sha256']=sha256(cp/'campaign-manifest.json');dec=Path(d['transition']['source_decision']['path']);x=json.loads(dec.read_text());x['checkpoint_manifest_sha256']=d['transition']['source_checkpoint']['manifest_sha256'];dec.write_text(json.dumps(x));d['transition']['source_decision']['sha256']=sha256(dec)
  with self.assertRaisesRegex(ValueError,'identity differs'):inspect_transition(d)
if __name__=='__main__':unittest.main(verbosity=2)
