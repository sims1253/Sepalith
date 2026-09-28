import json,sys,unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]
TRAINING=PACKET/'source/experiments/training'
sys.path.insert(0,str(TRAINING))
import full_weight_cpt_trainer as trainer
from stage_transition_contract import source_identity


class ProductionWiringTests(unittest.TestCase):
 def test_internal_trainer_step_is_one_while_scientific_membership_is_sixteen(self):
  recipe={'seed':7,'runtime':{'micro_batch':1,'gradient_accumulation':16,'effective_batch':16,
    'max_steps':90,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,
    'checkpoint_every':4}}
  args=trainer.training_arguments_kwargs(recipe,Path('/mnt/e/test-only'))
  self.assertEqual(args['per_device_train_batch_size'],1);self.assertEqual(args['gradient_accumulation_steps'],1)
  source=(TRAINING/'full_weight_cpt_trainer.py').read_text()
  self.assertIn('PackedOptimizerWindowDataset(dataset',source)
  self.assertIn('VarlenUpdateTrainerMixin, FullWeightOptimizerTrainerMixin, Trainer',source)

 def test_identity_binds_backend_and_global_token_reduction(self):
  recipe={'parent':{'candidate_id':'x','files':{'model.safetensors':'a','tokenizer.json':'t'},'saved_precision':{},},
   'transition':{'source_checkpoint':{'manifest_sha256':'b'},'global_optimizer_step_offset':66},
   'cohort':{'max_sequence_tokens':16384,'id':'c','rows':{'sha256':'d'},'streaming_cache':{'manifest_sha256':'e'},'updates':4},
   'source':{'manifest_sha256':'f'},'attention_backend':{'selected':'xformers','extension_sha256':'1'*64},
   'runtime':{'optimizer':{},'max_steps':70,'effective_batch':16,'micro_batch':1,'gradient_accumulation':16,'learning_rate':3e-6,'scheduler':'constant_with_warmup','warmup_steps':2,'checkpoint_every':4,'mandatory_stop_step':68}}
  policy=trainer.identity(recipe)['policy']
  self.assertEqual(policy['attention_backend'],'xformers');self.assertEqual(policy['loss_reduction'],'global_supervised_token_mean')
  recipe['schema']='sepalith.sft11.full-weight-cpt-stage-transition-bound.v1'
  self.assertEqual(source_identity(recipe),trainer.identity(recipe))

 def test_templates_fail_closed(self):
  recipe=json.loads((PACKET/'recipe.template.json').read_text())
  admission=json.loads((PACKET/'root-admission.template.json').read_text())
  commands=json.loads((PACKET/'root-commands.json').read_text())
  self.assertFalse(recipe['launch_authorized']);self.assertIsNone(recipe['attention_backend'])
  self.assertFalse(admission['launch_authorized']);self.assertIsNone(admission['attention_backend']['kernel_probe'])
  self.assertFalse(commands['execution_authorized']);self.assertIn('PYTHONPATH',commands['required_environment'])


if __name__=='__main__':unittest.main(verbosity=2)
