import copy,importlib.util,json,sys,unittest
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1/final-01/candidate-tokenrows.jsonl')
def load(name,path):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
P=load('token_review_test_protocol',PLAN/'docs/campaign/work/lead/r2-full-weight-edit-sft-preparation-v1/source/packages/sepalith/src/sepalith/campaign_protocol.py')
S=load('token_review_test_strict',PLAN/'docs/campaign/work/lead/r2-roxy8597-root-fixes-v1/audit_authoritative_stream.py')
class ReviewControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with ROWS.open() as stream:cls.row=json.loads(next(stream))
 def rejected(self,mutate):
  row=copy.deepcopy(self.row);mutate(row)
  with self.assertRaises(Exception):P.validate_training_row(row)
  self.assertTrue(S.validate_token_row(row,full_text=True))
 def test_baseline(self):self.assertEqual(P.validate_training_row(self.row)['id'],self.row['id']);self.assertEqual(S.validate_token_row(self.row,full_text=True),[])
 def test_target_start_mutation(self):self.rejected(lambda x:x.__setitem__('target_start',x['target_start']+1))
 def test_internal_token_mutation(self):self.rejected(lambda x:x['input_ids'].__setitem__(x['target_start'],x['input_ids'][x['target_start']]+1))
 def test_missing_eos(self):self.rejected(lambda x:x['input_ids'].pop())
 def test_terminal_mutation(self):self.rejected(lambda x:x['target_terminal_tokens'].__setitem__(0,2))
 def test_eos_budget_is_separate(self):
  context,prompt,reserve=16384,15359,1024
  self.assertEqual(1+prompt+reserve,context)
  self.assertEqual(1+prompt+reserve+1,context+1)
if __name__=='__main__':unittest.main()
