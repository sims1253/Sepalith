import copy,hashlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];T=P/'source/experiments/training';sys.path.insert(0,str(T))
def load(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load('prep',P/'prepare.py');Trainer=load('trainer',T/'full_weight_cpt_trainer.py');Contract=load('contract',T/'native_runtime_contract.py');Stage=load('stage',T/'stage_transition_contract.py');Ordinary=load('ordinary_canary_resume',T/'ordinary_canary_resume.py');sys.modules['ordinary_canary_resume']=Ordinary
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

class TestResumeFrontdoor(unittest.TestCase):
 def fixture(self,root):
  source=json.loads(A.SOURCE_RECIPE.read_text());recipe=A.build_recipe(source,root/'migration.json','m'*64);recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe)+'\n')
  old_identity=Trainer.identity(source);new_identity=Trainer.identity(recipe);resume=root/'checkpoint-450';resume.mkdir()
  prior_admission=root/'packed330-admission.json';prior_admission.write_text('{}\n')
  prior={'schema':'sepalith.sft11.selected-packed330-resume-lineage.v2','transition_admission':str(prior_admission),'transition_admission_sha256':Ordinary.sha(prior_admission)}
  state={'full':True,'checkpoint_kind':'full_weights','step':450,'identity':old_identity,'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_step':450,'cursor':6144,'stage_cursor':6144,'global_optimizer_step_offset':66,'effective_batch':16,'draw_schedule_sha256':'5ea4a04f4082f93ad8e7211a139e19e7ffffd33f4ef2661d93e42134933a3cac','ignore_data_skip':True,'resume_lineage':prior}}
  manifest={'full':True,'checkpoint_kind':'full_weights','step':450,'identity':old_identity,'files':{n:{'bytes':1,'sha256':'1'*64} for n in Ordinary.FULL_FILES}}
  mp=resume/'campaign-manifest.json';sp=resume/'campaign-state.json';mp.write_text(json.dumps(manifest)+'\n');sp.write_text(json.dumps(state)+'\n')
  reviews=[]
  for name in ('checkpoint','matched'):
   q=root/f'{name}.json';q.write_text(json.dumps({'status':'accepted','checkpoint_step':450})+'\n');reviews.append({'path':str(q),'sha256':Ordinary.sha(q)})
  admission={'schema':Ordinary.CADENCE64_SCHEMA,'status':'admitted','decision':'continue_checkpoint450_with_cadence64','launch_authorized':True,'bound_recipe_sha256':Ordinary.sha(recipe_path),'checkpoint':str(resume),'checkpoint_manifest_sha256':Ordinary.sha(mp),'campaign_state_sha256':Ordinary.sha(sp),'checkpoint_step':450,'source_identity_sha256':Ordinary.CADENCE24_IDENTITY_SHA,'destination_identity_sha256':Ordinary.CADENCE64_IDENTITY_SHA,'source_runtime_recipe':str(A.SOURCE_RECIPE),'source_runtime_recipe_sha256':A.SOURCE_RECIPE_SHA,'checkpoint450_review':reviews[0],'matched450_review':reviews[1],'allowed_changes':Ordinary.CADENCE64_ALLOWED_CHANGES,'payload_action':'load_checkpoint450_full_state_without_rewrite_or_optimizer_reset'}
  ap=root/'cadence-admission.json';ap.write_text(json.dumps(admission)+'\n')
  continuation={'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'admitted','decision':'continue','launch_authorized':True,'bound_recipe_sha256':Ordinary.sha(recipe_path),'checkpoint':str(resume),'checkpoint_manifest_sha256':Ordinary.sha(mp),'step':450};cp=root/'continuation.json';cp.write_text(json.dumps(continuation)+'\n')
  stop={'schema':'sepalith.sft11.native-cpt-execution-stop.v1','status':'admitted','launch_authorized':True,'bound_recipe_sha256':Ordinary.sha(recipe_path),'resume_global_step':450,'stop_at_global_step':706};ep=root/'stop.json';ep.write_text(json.dumps(stop)+'\n')
  return recipe,recipe_path,resume,ap,cp,ep,manifest,state
 def test_resolve450_then_later_same_identity_without_checkpoint330(self):
  with tempfile.TemporaryDirectory() as td:
   recipe,rp,resume,ap,cp,ep,manifest,state=self.fixture(Path(td));old=Ordinary.SOURCE_CHECKPOINT;Ordinary.SOURCE_CHECKPOINT=Path(td)/'evicted330'
   try:
    observed,lineage=Ordinary.resolve_resume_identity(rp,Trainer.identity(recipe),resume,ap);self.assertEqual(Ordinary.canonical_sha(observed),Ordinary.CADENCE24_IDENTITY_SHA);self.assertIsNotNone(lineage)
    manifest['identity']=state['identity']=Trainer.identity(recipe);(resume/'campaign-manifest.json').write_text(json.dumps(manifest)+'\n');(resume/'campaign-state.json').write_text(json.dumps(state)+'\n')
    observed2,lineage2=Ordinary.resolve_resume_identity(rp,Trainer.identity(recipe),resume,None);self.assertEqual(observed2,Trainer.identity(recipe));self.assertEqual(lineage2,lineage)
   finally:Ordinary.SOURCE_CHECKPOINT=old
 def test_manifest_cursor_admission_and_lr_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);recipe,rp,resume,ap,cp,ep,manifest,state=self.fixture(root)
   cases=[]
   bad=copy.deepcopy(recipe);bad['runtime']['learning_rate']=4e-6;cases.append(('lr',bad,None))
   badstate=copy.deepcopy(state);badstate['sampler']['cursor']=6160;cases.append(('cursor',recipe,badstate))
   for name,candidate,newstate in cases:
    if newstate is not None:(resume/'campaign-state.json').write_text(json.dumps(newstate)+'\n')
    with self.assertRaises((ValueError,AssertionError)):Ordinary.resolve_resume_identity(rp,Trainer.identity(candidate),resume,ap)
    (resume/'campaign-state.json').write_text(json.dumps(state)+'\n')
   ad=json.loads(ap.read_text());ad['checkpoint_manifest_sha256']='0'*64;ap.write_text(json.dumps(ad)+'\n')
   with self.assertRaises(ValueError):Ordinary.resolve_resume_identity(rp,Trainer.identity(recipe),resume,ap)
 def test_actual_preflight_resume_with_only_payload_verifier_mocked(self):
  with tempfile.TemporaryDirectory() as td:
   recipe,rp,resume,ap,cp,ep,manifest,state=self.fixture(Path(td));recipe['_native_validation']={'bundle_id':'fixture'};old_load,old_cohort=Trainer.load_bound,Trainer._validated_cohort
   import campaign_checkpoint
   old_verify=campaign_checkpoint.verify_checkpoint;campaign_checkpoint.verify_checkpoint=lambda *a,**k: manifest
   Trainer.load_bound=lambda path:recipe;Trainer._validated_cohort=lambda value,cursor:(types.SimpleNamespace(close=lambda:None),{'status':'pass','initial_cursor':cursor})
   try:
    report=Trainer.preflight_resume(rp,resume,cp,ep,ap);self.assertEqual((report['resume_step'],report['initial_cursor']),(450,6144));self.assertEqual(report['ordinary_canary_resume_lineage']['schema'],'sepalith.sft11.selected-packed330-resume-lineage.v2')
   finally:Trainer.load_bound,Trainer._validated_cohort=old_load,old_cohort;campaign_checkpoint.verify_checkpoint=old_verify
if __name__=='__main__':unittest.main()
