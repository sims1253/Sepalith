import copy,importlib.util,json,math,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
S=importlib.util.spec_from_file_location('review354',P/'prepare.py');M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def fixture():
 result={'status':'root_admitted_execution_stopped','global_step':354,'global_optimizer_step_offset':66,'initial_cursor':4224,'observed_draws':384,'last_draw_position':4607,'train_loss':1.0}
 rows=[]
 for step in range(331,355):
  rows += [{'event':'pre_optimizer','next_step':step,'finite_gradient_tensors':381,'nonzero_gradient_tensors':381},{'event':'log','step':step,'logs':{'loss':1.0,'grad_norm':2.0}}]
 return result,rows

class TestReview(unittest.TestCase):
 def test_exact_24_update_accounting(self):
  result,rows=fixture();self.assertEqual(M.validate_accounting(result,rows),{'updates':24,'draws':384,'cursor':4608})
 def test_missing_update_rejected(self):
  result,rows=fixture();rows=[x for x in rows if not(x.get('event')=='pre_optimizer' and x.get('next_step')==340)]
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
 def test_gradient_or_cursor_corruption_rejected(self):
  result,rows=fixture();rows[0]['nonzero_gradient_tensors']=380
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
  result,rows=fixture();result['last_draw_position']=4606
  with self.assertRaises(AssertionError):M.validate_accounting(result,rows)
 def test_long_evaluator_does_not_repeat_anchor(self):
  source=(P/'evaluate_long_only.py').read_text();self.assertIn("for label in ('8k', '16k')",source);self.assertNotIn("for label in ('anchor2k'",source);compile(source,str(P/'evaluate_long_only.py'),'exec')
 def test_runtime_handles_fail_closed_placeholders(self):
  value=json.loads((P/'runtime-handles.template.json').read_text());self.assertEqual(value['status'],'ROOT_FILLS_terminal_success');self.assertTrue(all(type(value[k]) is str for k in ('controller_pid','guard_pid','model_child_pid')))

if __name__=='__main__':unittest.main()
