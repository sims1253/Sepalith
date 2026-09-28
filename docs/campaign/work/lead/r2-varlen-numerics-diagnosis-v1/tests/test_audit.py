import json,os,sys,tempfile,unittest
from pathlib import Path
import torch
PKT=Path(__file__).parents[1];sys.path.insert(0,str(PKT/'source'))
from analyze_varlen_reports import analyze
from numerics_probe_helpers import split_concat_projection,threshold_decision
V3=Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-sm120-varlen-v1/xformers-actual-model-v3.json')
V4=Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-sm120-varlen-v1/xformers-actual-model-v4.json')
class Tests(unittest.TestCase):
 def test_exact_reports(self):
  d=analyze(V3,V4);self.assertEqual(d['decision'],'production varlen equivalence not established')
  self.assertGreater(d['observed']['gradient_relative_l2']['max'],.24)
  self.assertEqual(d['observed']['cross_document_isolation_max_abs'],0)
 def test_tampered_report_rejected(self):
  with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as t:
   p=Path(t)/'v4.json';p.write_bytes(V4.read_bytes()+b' ')
   with self.assertRaises(ValueError):analyze(V3,p)
 def test_projection_helper(self):
  layer=torch.nn.Linear(4,3,bias=False); rows=[torch.randn(2,4),torch.randn(3,4)]
  self.assertLess(split_concat_projection(layer,rows,torch)['relative_l2'],1e-6)
 def test_threshold_requires_controls_and_rejects_observed_scale(self):
  candidate={'isolation_max_abs':0.0,'repeat_relative_l2':0.0,'gradient_median_relative_l2':.0519,
   'gradient_max_relative_l2':.2422,'optimizer_update_relative_l2':.03,'optimizer_update_cosine':.998}
  ctrl={'gradient_median_relative_l2':.002,'gradient_max_relative_l2':.01}
  self.assertFalse(threshold_decision(candidate,ctrl)['pass'])
  with self.assertRaises(ValueError):threshold_decision(candidate,None)
if __name__=='__main__':unittest.main()
