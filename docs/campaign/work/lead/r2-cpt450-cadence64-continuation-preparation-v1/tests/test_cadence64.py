import copy,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];T=P/'source/experiments/training';sys.path.insert(0,str(T))
def load(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load('prep',P/'prepare.py');Trainer=load('trainer',T/'full_weight_cpt_trainer.py');Contract=load('contract',T/'native_runtime_contract.py');Stage=load('stage',T/'stage_transition_contract.py')
class TestCadence64(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.source=json.loads(A.SOURCE_RECIPE.read_text());cls.recipe=A.build_recipe(cls.source,'/tmp/not-admitted','x')
 def migration(self):return {'allowed_changes':Contract.CADENCE64_ALLOWED_CHANGES,'source_identity_sha256':A.SOURCE_IDENTITY,'destination_identity_sha256':A.canonical(Trainer.identity(self.recipe))}
 def test_only_cadence_and_review_identity_fields_change(self):
  destination=Contract.validate_cadence64_identity(self.source,self.recipe,Trainer.identity,self.migration(),check_evidence=False);self.assertEqual(destination,'6ac45eec834ee675bfecd880b06972ec028b7314497c47ca0130d6d32c43ce48')
  old=Trainer.identity(self.source);new=Trainer.identity(self.recipe)
  self.assertEqual({k:v for k,v in old.items() if k!='schedule'},{k:v for k,v in new.items() if k!='schedule'})
  self.assertEqual((new['schedule']['checkpoint_every'],new['schedule']['mandatory_stop_step']),(64,706))
 def test_other_identity_mutation_rejected(self):
  for path,value in [(('runtime','learning_rate'),4e-6),(('cohort','rows','sha256'),'0'*64),(('runtime','warmup_steps'),3)]:
   bad=copy.deepcopy(self.recipe);node=bad
   for key in path[:-1]:node=node[key]
   node[path[-1]]=value;m=self.migration();m['destination_identity_sha256']=A.canonical(Trainer.identity(bad))
   with self.assertRaises(ValueError):Contract.validate_cadence64_identity(self.source,bad,Trainer.identity,m,check_evidence=False)
 def test_source_cursor6144_and_schedule(self):
  state={'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'draw_schedule_sha256':self.recipe['cohort']['draw_schedule']['sha256'],'stage_cursor':6144,'cursor':6144,'global_step':450,'ignore_data_skip':True}}
  self.assertEqual(Stage.stage_cursor_from_checkpoint(state,66,self.recipe['cohort']['draw_schedule']['sha256']),6144)
  bad=copy.deepcopy(state);bad['sampler']['cursor']=6160
  with self.assertRaises(ValueError):Stage.stage_cursor_from_checkpoint(bad,66,self.recipe['cohort']['draw_schedule']['sha256'])
 def test_exact_saves_and_stop_are_aligned(self):
  r=self.recipe['runtime'];self.assertEqual([s for s in range(451,707) if Trainer.milestone_action(s,450,r,66,706)['save']],[514,578,642,706]);self.assertTrue(Trainer.milestone_action(706,450,r,66,706)['stop'])
 def test_templates_fail_closed(self):
  migration=json.loads((P/'cadence64-migration-admission.template.json').read_text());self.assertFalse(migration['launch_authorized']);self.assertNotEqual(migration['status'],'admitted')
  recipe=json.loads((P/'runtime-recipe.template.json').read_text());self.assertTrue(recipe['status'].startswith('ROOT_MUST_BIND'))
  source=(P/'prepare.py').read_text();self.assertIn("Path(args.output).resolve().parent",source);self.assertNotIn("write(PACKET/'continuation-450",source)
if __name__=='__main__':unittest.main()
