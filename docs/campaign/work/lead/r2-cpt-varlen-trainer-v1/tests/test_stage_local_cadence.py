import sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];SRC=HERE/'source/experiments/training';sys.path.insert(0,str(SRC))
import full_weight_cpt_trainer as trainer

class TestStageLocalCadence(unittest.TestCase):
 def recipe(self):
  return {'transition':{'global_optimizer_step_offset':66},'cohort':{'updates':11443},'runtime':{'max_steps':11509,'checkpoint_every':128,'mandatory_stop_step':194,'evaluation_steps':[194,11509],'selected_milestones':[194,11509]}}
 def test_representative66_to_stage128_global194_is_valid(self):
  self.assertEqual(trainer.validate_stage_schedule(self.recipe()),{'offset':66,'mandatory_stage_step':128,'terminal_stage_step':11443})
  self.assertEqual(trainer.milestone_action(194,193,self.recipe()['runtime'],66),{'save':True,'stop':True,'cursor':2048})
 def test_global_modulo_is_irrelevant_but_stage_modulo_is_required(self):
  self.assertNotEqual(194%128,0)
  bad=self.recipe();bad['runtime']['mandatory_stop_step']=195;bad['runtime']['evaluation_steps']=[195,11509];bad['runtime']['selected_milestones']=[195,11509]
  with self.assertRaisesRegex(ValueError,'stage-local'):trainer.validate_stage_schedule(bad)
 def test_milestones_must_remain_in_destination_and_terminal_evaluated(self):
  for field,value,pattern in [('evaluation_steps',[66,194,11509],'leave destination'),('selected_milestones',[194],'terminal milestone')]:
   bad=self.recipe();bad['runtime'][field]=value
   with self.assertRaisesRegex(ValueError,pattern):trainer.validate_stage_schedule(bad)

if __name__=='__main__':unittest.main()
