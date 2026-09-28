import hashlib,json,tempfile,unittest
from pathlib import Path
from campaign_fixtures import LEAD,load,rebased_copy
from sepalith.training.sft import bind_full_weight_edit_sft as binder
from sepalith.training.sft import campaign_edit_data as edit
from sepalith.training.sft import full_weight_edit_sft as trainer
PACKET=LEAD/'r2-full-weight-edit-sft-preparation-v1';TRAIN=Path(trainer.__file__).parent
# The package keeps the eval-gate-v1 trainer and binder, which supersede this packet's
# versions. Their templates bind the same 15,006-row cohort and schedule.
GATE_PACKET=LEAD/'r2-full-weight-edit-sft-eval-gate-v1'
FIXTURES=tempfile.mkdtemp(prefix='sepalith-c01-');TEMPLATE=rebased_copy(GATE_PACKET/'recipe.template.json',FIXTURES)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

class PacketTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.recipe=load(TEMPLATE)
  cls.rows,cls.draws,cls.schedule,cls.totals=edit.inspect(cls.recipe['cohort']['rows'],cls.recipe['cohort']['draw_schedule'],16384,16)
 def admission(self):
  value=load(GATE_PACKET/'root-admission.template.json');value['template_sha256']=sha(TEMPLATE);value['status']='admitted';value['launch_authorized']=True
  value['parent']['candidate_id']='root-selected-cpt1902-merged';value['parent']['path']='/mnt/e/root-selected-placeholder'
  value['parent']['files'].update({'model.safetensors':'a'*64,'config.json':'b'*64,'generation_config.json':'c'*64,'tokenizer_config.json':'d'*64,'parent-manifest.preparation.json':'e'*64})
  return value
 def test_actual_15006_full_coverage_and_named_replays(self):
  self.assertEqual((len(self.rows),len(self.draws),self.schedule['max_steps']),(15006,18560,1160))
  self.assertEqual(set(self.draws),set(range(15006)));self.assertEqual(self.schedule['excluded'],[])
  coverage=self.schedule['coverage'];self.assertEqual(coverage['replay_draws_by_reason'],{'noop_ratio_length_alignment_replay':3546,'edit_batch_completion_alignment_replay':8})
  self.assertLess(coverage['last_unique_draw_by_pool']['noop'],coverage['first_replay_draw_by_pool']['noop']);self.assertLess(coverage['last_unique_draw_by_pool']['edit'],coverage['first_replay_draw_by_pool']['edit'])
  self.assertEqual(self.schedule['token_denominators']['unique']['supervised_target_tokens_including_eos'],1577478)
  self.assertEqual(self.schedule['token_denominators']['scheduled']['supervised_target_tokens_including_eos'],1620249)
 def test_exact_length_profile_and_complete_targets(self):
  profile=json.loads((PACKET/'length-profile.json').read_text())
  lengths=[len(x['input_ids']) for x in self.rows];targets=[len(x['input_ids'])-x['target_start'] for x in self.rows]
  self.assertEqual((profile['rows'],sum(lengths),max(lengths)),(15006,13755681,3064));self.assertEqual((sum(targets),max(targets)),(1577478,933))
  self.assertTrue(profile['observed_fit_without_truncation']);self.assertEqual(profile['sequence_tokens']['over_threshold_counts']['4096'],0)
  for row in self.rows:self.assertEqual(row['input_ids'][row['target_start']:],row['target_body_tokens']+row['target_terminal_tokens']+[1])
 def test_target_only_collator_masks_prompt_and_supervises_body_terminal_eos(self):
  chosen=[self.rows[0],max(self.rows,key=lambda x:len(x['input_ids']))];batch=edit.target_only_collator(chosen,16384)
  self.assertEqual(edit.verify_batch(batch,chosen),sum(len(x['input_ids'])-x['target_start'] for x in chosen))
  for i,row in enumerate(chosen):
   self.assertTrue(batch['labels'][i,:row['target_start']].eq(-100).all());self.assertEqual(batch['labels'][i,row['target_start']:len(row['input_ids'])].tolist(),row['target_body_tokens']+row['target_terminal_tokens']+[1])
 def test_adapter_accepts_over_4096_complete_target_and_rejects_truncation(self):
  body=[8]*1100;terminal=[9]*100;prompt=[7]*3797;ids=[0]+prompt+body+terminal+[1]
  row={'id':'long','split':'train','renderer_id':'zeta2-prm03-v1','tokenizer_json_sha256':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','input_ids':ids,'target_start':3798,'prompt_token_count':3797,'target_body_tokens':body,'target_terminal_tokens':terminal,'target_body_token_count':1100,'target_terminal_token_count':100,'target_token_count':1200,'family':'finish_block','package_id':'p'}
  checked=edit.validate_row(row,16384);self.assertEqual(len(checked['input_ids']),4999);self.assertEqual(len(checked['target_body_tokens'])+len(checked['target_terminal_tokens']),1200)
  bad=dict(row);bad['input_ids']=ids[:-1]
  with self.assertRaisesRegex(ValueError,'BOS/EOS|complete target'):edit.validate_row(bad,16384)
 def test_template_preflight_binds_reviewed_data_and_pending_roxy_registry(self):
  result=trainer.preflight_template(TEMPLATE);self.assertEqual((result['unique_rows'],result['draws'],result['updates']),(15006,18560,1160));self.assertEqual(result['additional_roxy_rows_pending'],10017);self.assertFalse(result['cuda_started'])
  queue=json.loads((PACKET/'data-queue.json').read_text());pending=queue['pending_cohorts'][0]
  self.assertEqual((pending['rows'],pending['admitted_rows'],pending['status']),(10017,0,'review_complete_training_admission_pending'));self.assertEqual(queue['active_cohort']['rows_sha256'],self.recipe['cohort']['rows']['sha256'])
 def test_binder_requires_root_parent_and_bounded_reviewed_optimizer(self):
  with tempfile.TemporaryDirectory() as tmp:
   admission=self.admission();a=Path(tmp)/'a.json';out=Path(tmp)/'bound.json';a.write_text(json.dumps(admission));bound=binder.bind(TEMPLATE,a,out)
   self.assertEqual(bound['parent']['kind'],'merged_cpt_parent');self.assertEqual(bound['runtime']['effective_batch'],16);self.assertEqual(bound['runtime']['max_steps'],1160)
  for mutation,message in ((lambda x:x['parent'].update(kind='adapter_checkpoint'),'selected parent'),(lambda x:(x['selected'].update(learning_rate=1e-4),x['selected']['optimizer'].update(hidden_lr=1e-4)),'editing LR'),(lambda x:x['selected']['optimizer'].update(state_dtype='bfloat16'),'optimizer precision')):
   with tempfile.TemporaryDirectory() as tmp:
    admission=self.admission();mutation(admission);a=Path(tmp)/'a.json';a.write_text(json.dumps(admission))
    with self.assertRaisesRegex(ValueError,message):binder.bind(TEMPLATE,a,Path(tmp)/'bound.json')
 def test_full_weight_resume_and_tokenizer_hooks_are_present(self):
  source=(TRAIN/'full_weight_edit_sft.py').read_text();self.assertIn('expected_checkpoint_kind="full_weights"',source);self.assertIn('restore_trainer_eog_alignment',source);self.assertIn('assert_serialized_tokenizer',source);self.assertIn('EXPECTED_PARAMETER_TENSORS = 381',source);self.assertIn('EXPECTED_PARAMETERS = 2_516_756_480',source);self.assertNotIn('get_peft_model',source)
 def test_evaluation_request_keeps_metric_roles_and_final_sealed(self):
  # eval-gate-v1 takes the generation budget from the bound DEV gate instead of a fixed 192.
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/'a.json';a.write_text(json.dumps(self.admission()));bound=binder.bind(TEMPLATE,a,Path(tmp)/'bound.json')
  request=trainer.build_evaluation_request(bound,290,Path('/checkpoint-290'));self.assertEqual(request['teacher_forced']['role'],'diagnostic_only_not_edit_accuracy');self.assertEqual((request['generation']['role'],request['generation']['max_new_tokens']),('required_edit_noop_quality',bound['dev_gate']['generation_max_new_tokens']));self.assertFalse(request['final_set_access']);self.assertFalse(request['same_model_in_callback_executed'])
 def test_training_arguments_cpu_constructor_and_sequential_resume_contract(self):
  from transformers import TrainingArguments
  recipe={'seed':3407,'runtime':{'micro_batch':1,'gradient_accumulation':16,'max_steps':1160,'learning_rate':1e-5,'scheduler':'cosine','warmup_ratio':.03,'checkpoint_every':290}}
  with tempfile.TemporaryDirectory() as tmp:
   kwargs=trainer.training_arguments_kwargs(recipe,Path(tmp));kwargs['use_cpu']=True;args=TrainingArguments(**kwargs)
   self.assertEqual((args.max_steps,args.save_steps,args.warmup_steps),(1160,290,34));self.assertFalse(args.ignore_data_skip);self.assertEqual(kwargs['train_sampling_strategy'],'sequential')
 def test_atomic_publication_moves_sealed_checkpoint_on_same_filesystem(self):
  from sepalith.training.checkpoint import campaign_checkpoint  # noqa: F401
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'runtime/checkpoint-1';source.mkdir(parents=True);(source/'payload').write_bytes(b'abc')
   manifest={'files':{'payload':{'bytes':3,'sha256':sha(source/'payload')}}};(source/'campaign-manifest.json').write_text(json.dumps(manifest))
   destination=root/'archive/full/checkpoint-1';result=trainer.publish_sealed_checkpoint(source,destination,manifest)
   self.assertEqual(result,destination);self.assertFalse(source.exists());self.assertEqual((destination/'payload').read_bytes(),b'abc')

if __name__=='__main__':unittest.main(verbosity=2)
