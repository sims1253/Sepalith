import copy,hashlib,json,sys,tempfile,unittest
from unittest import mock
from pathlib import Path
ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');P=ROOT/'docs/campaign/work/lead/r2-full-weight-edit-sft-eval-gate-v1';T=P/'source/experiments/training';sys.path.insert(0,str(T));sys.path.insert(0,str(P/'source'))
import bind_full_weight_edit_sft as binder
import campaign_eval
import full_weight_edit_sft as trainer
import milestone_gate
import prepare_dev_gate

def sha(p):
 h=hashlib.sha256(Path(p).read_bytes());return h.hexdigest()
class EvalGateTests(unittest.TestCase):
 def admission(self,cap=1024):
  a=json.loads((P/'root-admission.template.json').read_text());a['status']='admitted';a['launch_authorized']=True;a['selected']['development_max_new_tokens']=cap;a['parent']['candidate_id']='cpt1902';a['parent']['path']='/mnt/e/root-parent';a['parent']['files'].update({'model.safetensors':'a'*64,'config.json':'b'*64,'generation_config.json':'c'*64,'tokenizer_config.json':'d'*64,'parent-manifest.preparation.json':'e'*64});return a
 def test_binder_requires_mandatory_stops_and_configurable_generation_budget(self):
  for cap in (1024,2048):
   with tempfile.TemporaryDirectory() as tmp:
    a=Path(tmp)/'a';o=Path(tmp)/'o';a.write_text(json.dumps(self.admission(cap)));bound=binder.bind(P/'recipe.template.json',a,o);self.assertEqual(bound['dev_gate']['mandatory_steps'],[290,580,870,1160]);self.assertEqual(bound['dev_gate']['generation_max_new_tokens'],cap)
  for mutate,message in ((lambda a:a['selected'].update(mandatory_dev_gate_steps=[580,1160]),'mandatory DEV'),(lambda a:a['selected'].update(development_max_new_tokens=8193),'generation budget')):
   with tempfile.TemporaryDirectory() as tmp:
    value=self.admission();mutate(value);a=Path(tmp)/'a';a.write_text(json.dumps(value))
    with self.assertRaisesRegex(ValueError,message):binder.bind(P/'recipe.template.json',a,Path(tmp)/'o')
 def test_boundary_control_always_saves_and_stops_at_gate(self):
  class Control:should_save=False;should_training_stop=False
  c=Control();self.assertFalse(trainer.enforce_mandatory_gate_boundary(289,290,c));self.assertFalse(c.should_training_stop)
  self.assertTrue(trainer.enforce_mandatory_gate_boundary(290,290,c));self.assertTrue(c.should_save and c.should_training_stop)
  with self.assertRaisesRegex(ValueError,'crossed'):trainer.enforce_mandatory_gate_boundary(291,290,Control())
 def test_missing_evidence_fails_closed_then_exact_root_evidence_allows_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);gates=root/'gates';archive=root/'archive';checkpoint=archive/'full/checkpoint-2';checkpoint.mkdir(parents=True);(checkpoint/'campaign-manifest.json').write_text('{}')
   recipe={'runtime':{'evaluation_steps':[2,4],'max_steps':4},'development':{'generation_budget':{'prepared_max_new_tokens_bound':8192}},'dev_gate':{'mandatory_steps':[2,4],'generation_max_new_tokens':1024,'evidence_directory':str(gates),'panel':{'path':'/dev75','sha256':milestone_gate.PANEL_SHA}},'outputs':{'archive':str(archive)}};recipe_path=root/'bound.json';recipe_path.write_text(json.dumps(recipe))
   with mock.patch.object(milestone_gate,'mandatory_steps',return_value=[2,4]):
    with self.assertRaisesRegex(ValueError,'missing root DEV gate'):milestone_gate.continuation_plan(recipe,recipe_path,checkpoint)
   result={'schema':'sepalith.sft11.full-weight-edit-dev-generation.v1','status':'complete','step':2,'bound_recipe':{'sha256':sha(recipe_path)},'checkpoint':{'campaign_manifest_sha256':sha(checkpoint/'campaign-manifest.json')},'panel':recipe['dev_gate']['panel'],'summary':{'denominators':{'cases':75},'counts':{'protocol_valid':70},'generation_max_new_tokens':1024}}
   result_path=root/'result.json';result_path.write_text(json.dumps(result));decision={'schema':'sepalith.sft11.full-weight-edit-dev-decision.v1','status':'admitted_for_continuation','continue_training':True,'step':2,'bound_recipe_sha256':sha(recipe_path),'checkpoint_manifest_sha256':sha(checkpoint/'campaign-manifest.json'),'generation_result_sha256':sha(result_path),'panel_sha256':milestone_gate.PANEL_SHA};decision_path=root/'decision.json';decision_path.write_text(json.dumps(decision))
   gate=gates/'dev-gate-step-2.json';prepare_dev_gate.prepare(recipe_path,checkpoint,result_path,decision_path,gate)
   with mock.patch.object(milestone_gate,'mandatory_steps',return_value=[2,4]):plan=milestone_gate.continuation_plan(recipe,recipe_path,checkpoint)
   self.assertEqual(plan,{'initial_step':2,'next_mandatory_stop':4,'prior_gates_verified':[2]})
   result['summary']['denominators']['cases']=74;result_path.write_text(json.dumps(result))
   with mock.patch.object(milestone_gate,'mandatory_steps',return_value=[2,4]):
    with self.assertRaisesRegex(ValueError,'gate result differs'):milestone_gate.continuation_plan(recipe,recipe_path,checkpoint)
 def test_generation_termination_accounts_for_both_eog_and_cap(self):
  self.assertEqual(campaign_eval.generation_termination([8,1],1024,(1,130073)),{'response_complete':True,'stop_token_id':1,'cap_hit':False})
  self.assertEqual(campaign_eval.generation_termination([8,130073],1024,(1,130073)),{'response_complete':True,'stop_token_id':130073,'cap_hit':False})
  self.assertEqual(campaign_eval.generation_termination([8]*1024,1024,(1,130073)),{'response_complete':False,'stop_token_id':None,'cap_hit':True})
 def test_one_pass_alternative_is_exactly_15006_unique_plus_two_terminal_replays(self):
  x=json.loads((P/'one-pass-15008-draw-schedule.alternative.json').read_text());self.assertEqual((x['draw_count'],x['max_steps']),(15008,938));self.assertEqual(x['coverage']['distinct_rows_drawn'],15006);self.assertEqual(x['coverage']['replay_draws_by_reason'],{'terminal_batch_padding_replay':2});self.assertGreater(x['coverage']['first_replay_draw'],x['coverage']['last_unique_draw']);self.assertTrue(x['checks']['no_target_truncation']);self.assertFalse(x['policy']['selected_for_training'])
  presentations=[d['presentation'] for d in x['draws']];self.assertEqual(presentations.count(1),15006);self.assertEqual(sum(v>1 for v in presentations),2)
 def test_checkpoint_identity_binds_gate_steps_and_panel(self):
  recipe={'parent':{'candidate_id':'p','files':{'model.safetensors':'a','tokenizer.json':'b'}},'cohort':{'max_sequence_tokens':16384,'id':'c','rows':{'sha256':'d'},'draw_schedule':{'sha256':'e'}},'source':{'manifest_sha256':'f'},'runtime':{'optimizer':{},'max_steps':4,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':1e-5,'scheduler':'cosine','warmup_ratio':.03,'checkpoint_every':2},'dev_gate':{'mandatory_steps':[2,4],'panel':{'sha256':milestone_gate.PANEL_SHA}}};a=trainer.identity(recipe);other=copy.deepcopy(recipe);other['dev_gate']['mandatory_steps']=[4];self.assertNotEqual(a,trainer.identity(other))
 def test_generation_driver_is_bound_to_full_checkpoint_and_corrected_dev_only(self):
  source=(T/'run_dev_generation.py').read_text();self.assertIn("expected_checkpoint_kind='full_weights'",source);self.assertIn("summary['denominators']['cases']==75",source);self.assertIn("final_set_access'] is False",source);self.assertIn("generation_max_new_tokens",source);self.assertNotIn('final.jsonl',source)
 def test_template_remains_launch_refusing_and_preflight_passes(self):
  recipe=json.loads((P/'recipe.template.json').read_text());self.assertIsNone(recipe['parent']);self.assertFalse(recipe['launch_authorized']);self.assertEqual(recipe['development']['generation_budget']['default_candidate'],1024)
  result=trainer.preflight_template(P/'recipe.template.json');self.assertEqual((result['unique_rows'],result['draws']),(15006,18560));self.assertFalse(result['cuda_started'])

if __name__=='__main__':unittest.main(verbosity=2)
