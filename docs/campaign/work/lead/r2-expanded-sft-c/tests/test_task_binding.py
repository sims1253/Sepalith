"""Task-only data/resume metadata checks; no serialized model/state is read."""
import copy, json, tempfile, unittest
from pathlib import Path
from test_campaign_task_sft import _recipe, _rows
from campaign_task_sft import task_target_only_policy, validate_task_admission, validate_task_rows

class TaskBindingTests(unittest.TestCase):
    def test_changed_data_record_rejected_under_unchanged_identity(self):
        for key in ('token_rows','draw_schedule','development_panel'):
            candidate=_recipe();candidate[key]['sha256']='9'*64
            with self.assertRaisesRegex(ValueError,'differs from identity.data'):
                task_target_only_policy(candidate)

    def test_real_temp_metadata_250_500_and_changed_cursor_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as temp:
            root=Path(temp);recipe=_recipe()
            recipe['draw_schedule']['path']=str(root/'draws.json')
            schedule={'split_id':recipe['identity']['data']['split_id']}
            def put(name,value):(root/name).write_text(json.dumps(value)+'\n')
            put('draws.json',schedule)
            self.assertEqual(validate_task_admission(recipe,_rows())['rows'],2)
            recipe['resume_from']=str(root)
            for step in (250,500):
                good={'step':step,'full':True,'identity':copy.deepcopy(recipe['identity']),
                      'sampler':{'split_id':schedule['split_id'],
                                 'schedule_sha256':recipe['draw_schedule']['sha256'],
                                 'consumed_draws':step*16}}
                put('trainer_state.json',{'global_step':step})
                put('campaign-manifest.json',{'step':step,'full':True,'identity':recipe['identity']})
                put('campaign-state.json',good)
                self.assertEqual(validate_task_admission(recipe,_rows())['rows'],2)
                for key,value in [('consumed_draws',step*16-1),('schedule_sha256','9'*64),('split_id','wrong-split')]:
                    bad=copy.deepcopy(good);bad['sampler'][key]=value;put('campaign-state.json',bad)
                    with self.assertRaisesRegex(ValueError,'sampler split/schedule/cursor'):
                        validate_task_admission(recipe,_rows())
                put('campaign-state.json',good);put('trainer_state.json',{'global_step':step+1})
                with self.assertRaisesRegex(ValueError,'step/full identity'):
                    validate_task_admission(recipe,_rows())
                put('trainer_state.json',{'global_step':step})
                recipe['resume_milestone']=750
                with self.assertRaisesRegex(ValueError,'named resume milestone'):
                    validate_task_admission(recipe,_rows())
                del recipe['resume_milestone']
            recipe['resume_from']=None;put('draws.json',{'split_id':'wrong-split'})
            with self.assertRaisesRegex(ValueError,'draw split'):
                validate_task_admission(recipe,_rows())

    def test_noop_complete_terminal_and_eos_is_supervised(self):
        recipe=_recipe();row=copy.deepcopy(_rows()[0]);row['target_body_tokens']=[]
        row['target_body_token_count']=0
        row['input_ids']=row['input_ids'][:row['target_start']]+row['target_terminal_tokens']+[1]
        summary=validate_task_rows(recipe,[row])
        self.assertEqual(summary['target_tokens_including_eos'],len(row['target_terminal_tokens'])+1)

if __name__=='__main__':unittest.main(verbosity=2)
