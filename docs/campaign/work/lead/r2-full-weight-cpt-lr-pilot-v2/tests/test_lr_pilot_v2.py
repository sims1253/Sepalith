import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
PACKET=Path(__file__).resolve().parents[1];V1=PACKET.parent/'r2-full-weight-cpt-lr-pilot-v1';TRAIN=PACKET/'source/experiments/training';sys.path.insert(0,str(TRAIN));sys.path.insert(0,str(PACKET/'source'))
import bind_full_weight_cpt as binder
import full_weight_cpt_trainer as trainer
import campaign_cpt_data as cpt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]

class LrPilotTests(unittest.TestCase):
 def admitted(self,label):
  p=PACKET/f'root-admission.{label}.template.json';x=json.loads(p.read_text());x['status']='admitted';x['launch_authorized']=True;return x
 def test_panel_is_384_distinct_full_8k_all_token_rows(self):
  values=rows(V1/'panel/train384-ctx8192.jsonl');self.assertEqual(len(values),384)
  self.assertEqual(len({x['package'] for x in values}),384);self.assertEqual(len({x['document_id'] for x in values}),384)
  for x in values:
   self.assertEqual(len(x['input_ids']),8192);self.assertEqual(x['labels'][0],-100);self.assertEqual(x['labels'][1:-1],x['input_ids'][1:-1]);self.assertEqual(x['labels'][-1],-100);self.assertEqual(x['attention_mask'],[1]*8192)
   cpt.validate_materialized_row(x,max_sequence_tokens=8192)
 def test_draws_match_exact_row_order_and_holdout_is_disjoint(self):
  values=rows(V1/'panel/train384-ctx8192.jsonl');draw=json.loads((V1/'panel/draws384.json').read_text());valid=rows('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl')
  self.assertEqual(draw['row_ids'],[x['row_id'] for x in values]);self.assertEqual(draw['token_rows_sha256'],sha(V1/'panel/train384-ctx8192.jsonl'))
  self.assertFalse({x['package'] for x in values}&{x['package'] for x in valid});self.assertFalse({x['document_id'] for x in values}&{x['document_id'] for x in valid})
 def test_frozen_dataset_accepts_exact_24_update_schedule(self):
  recipe=json.loads((PACKET/'recipe.lr3e-5.template.json').read_text());dataset=trainer.FrozenTokenRowDataset(recipe)
  self.assertEqual(len(dataset),384);self.assertEqual(dataset[0]['_draw_position'],0);self.assertEqual(dataset[383]['_draw_position'],383)
 def test_binder_accepts_only_two_lower_lr_pairs_and_terminal_save(self):
  for label,expected in [('lr3e-5',3e-5),('lr1e-5',1e-5)]:
   with tempfile.TemporaryDirectory() as value:
    a=Path(value)/'admission.json';o=Path(value)/'bound.json';a.write_text(json.dumps(self.admitted(label)));x=binder.bind(PACKET/f'recipe.{label}.template.json',a,o)
    self.assertEqual(x['runtime']['learning_rate'],expected);self.assertEqual(x['runtime']['evaluation_steps'],[24]);self.assertEqual(x['runtime']['checkpoint_every'],24)
  with tempfile.TemporaryDirectory() as value:
   x=self.admitted('lr3e-5');x['selected']['optimizer']['side_lr']=2e-5;a=Path(value)/'a.json';a.write_text(json.dumps(x))
   with self.assertRaisesRegex(ValueError,'pair differs'):binder.bind(PACKET/'recipe.lr3e-5.template.json',a,Path(value)/'b.json')
 def test_binder_rejects_extra_checkpoint_or_wrong_warmup(self):
  for field,value,message in [('checkpoint_every',12,'terminal checkpoint'),('warmup_ratio',0.0,'warmup')]:
   with tempfile.TemporaryDirectory() as tmp:
    x=self.admitted('lr3e-5');x['selected'][field]=value
    if field=='checkpoint_every':x['selected']['evaluation_steps']=[12,24];x['selected']['selected_milestones']=[12,24]
    a=Path(tmp)/'a.json';a.write_text(json.dumps(x))
    with self.assertRaisesRegex(ValueError,message):binder.bind(PACKET/'recipe.lr3e-5.template.json',a,Path(tmp)/'b.json')
 def test_baseline_precedes_trainer_construction_and_train(self):
  source=(TRAIN/'full_weight_cpt_trainer.py').read_text();baseline=source.index('baseline = evaluator(');build=source.index('trainer = FullWeightTrainer(');train=source.index('result = trainer.train(')
  self.assertLess(baseline,build);self.assertLess(build,train);self.assertIn('"before_parent_baseline"',source);self.assertIn('"after_parent_baseline"',source)
 def test_training_arguments_are_24_updates_two_warmup_on_cpu(self):
  from transformers import TrainingArguments
  with tempfile.TemporaryDirectory() as tmp:
   recipe={'seed':3407,'runtime':{'micro_batch':1,'gradient_accumulation':16,'max_steps':24,'learning_rate':3e-5,'scheduler':'constant_with_warmup','warmup_ratio':1/12,'checkpoint_every':24}}
   kwargs=trainer.training_arguments_kwargs(recipe,Path(tmp));kwargs['use_cpu']=True;args=TrainingArguments(**kwargs)
   self.assertEqual(args.max_steps,24);self.assertEqual(args.warmup_steps,2);self.assertEqual(args.save_steps,24)
 def test_terminal_checkpoint_only_and_resume_token_contract_remain(self):
  source=(TRAIN/'full_weight_cpt_trainer.py').read_text();self.assertIn('expected_checkpoint_kind="full_weights"',source);self.assertIn('post_load_repair = restore_trainer_eog_alignment',source);self.assertIn('assert_serialized_tokenizer(source, reference',source)
  self.assertNotIn('save_only_model": True',source)

 def test_per_row_diagnostics_preserve_aggregate_contract(self):
  import campaign_cpt_eval as ev,torch
  valid='/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl'
  evaluator=ev.package_holdout_evaluator({'validation_rows':{'path':valid},'parameters':{'max_sequence_tokens':2048},'materialized_rows':True,'validation_batch_size':1})
  class Model:
   training=True
   parameter=torch.nn.Parameter(torch.zeros(1))
   def parameters(self):return iter([self.parameter])
   def eval(self):self.training=False;return self
   def train(self,value=True):self.training=value;return self
   def __call__(self,**kwargs):return type('Output',(),{'loss':torch.tensor(2.0)})()
  result=evaluator(Model(),None,Path('/checkpoint'),0)
  self.assertEqual(len(result['row_metrics']),499);self.assertEqual(result['denominators']['validation_rows'],499)
  self.assertEqual(result['metrics']['mean_causal_nll'],2.0)
  self.assertEqual(sum(x['loss_tokens'] for x in result['row_metrics']),result['denominators']['validation_loss_tokens'])
  self.assertAlmostEqual(sum(x['loss_sum'] for x in result['row_metrics'])/sum(x['loss_tokens'] for x in result['row_metrics']),result['metrics']['mean_causal_nll'])
  self.assertTrue(all(x['row_id'] and x['package_id'] for x in result['row_metrics']))
 def test_high_lr_pairs_are_rejected(self):
  with tempfile.TemporaryDirectory() as value:
   x=self.admitted('lr3e-5');x['selected']['learning_rate']=1e-4;x['selected']['optimizer']['hidden_lr']=1e-4;x['selected']['optimizer']['side_lr']=1e-5
   a=Path(value)/'a';a.write_text(json.dumps(x))
   with self.assertRaisesRegex(ValueError,'pair differs'):binder.bind(PACKET/'recipe.lr3e-5.template.json',a,Path(value)/'b')

if __name__=='__main__':unittest.main(verbosity=2)
