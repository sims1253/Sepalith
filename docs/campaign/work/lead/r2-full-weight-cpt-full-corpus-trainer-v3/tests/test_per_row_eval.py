import sys,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/"source/experiments/training"))
class T(unittest.TestCase):
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
  self.assertTrue(all(x['row_id'] and x['document_id'] and x['package_id'] for x in result['row_metrics']))
  raw=[json.loads(line) for line in Path(valid).read_text().splitlines() if line]
  self.assertEqual([x['row_id'] for x in result['row_metrics']],[x['row_id'] for x in raw])
  self.assertEqual([x['document_id'] for x in result['row_metrics']],[x['document_id'] for x in raw])
  self.assertEqual(len({x['row_id'] for x in result['row_metrics']}),499)
 def test_row_identity_is_chunk_specific_when_document_repeats(self):
  import campaign_cpt_eval as ev
  left={'row_id':'chunk-a','document_id':'shared-document','package':'package-a'}
  right={'row_id':'chunk-b','document_id':'shared-document','package':'package-a'}
  self.assertNotEqual(ev._row_id(left),ev._row_id(right))
  self.assertEqual(ev._case_id(left),ev._case_id(right))
  with self.assertRaisesRegex(ValueError,'unique row identity'):
   ev._row_id({'document_id':'shared-document','package':'package-a'})

if __name__=='__main__':unittest.main(verbosity=2)
