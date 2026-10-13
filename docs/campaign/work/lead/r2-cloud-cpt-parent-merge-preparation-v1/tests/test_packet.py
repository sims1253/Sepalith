import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
MERGE=load('merge',ROOT/'source/merge_cloud_cpt_parent.py')
BINDER=ROOT/'source/prepare_binding.py'; REG=ROOT/'candidates.json'
class Tests(unittest.TestCase):
 def test_streaming_merge_matches_fp32_reference_and_one_cast(self):
  g=torch.Generator().manual_seed(3407)
  base=torch.randn(7,5,generator=g).to(torch.bfloat16); left=torch.randn(7,3,generator=g); right=torch.randn(3,5,generator=g)
  got=MERGE.merge_matrix_once(base,left,right,2.0,torch)
  expected=(base.float()+left.float().matmul(right.float())*2.0).to(torch.bfloat16)
  self.assertTrue(torch.equal(got,expected)); self.assertEqual(got.dtype,torch.bfloat16)
 def test_shape_mismatch_rejected(self):
  with self.assertRaises(AssertionError): MERGE.merge_matrix_once(torch.zeros(2,2),torch.zeros(3,1),torch.zeros(1,2),2,torch)
 def test_1585_review_is_exact_and_unselected(self):
  with tempfile.TemporaryDirectory() as d:
   o=Path(d)/'review.json'
   subprocess.run([sys.executable,str(BINDER),'--registry',str(REG),'--candidate','1585','--review-output',str(o)],check=True,capture_output=True,text=True)
   x=json.loads(o.read_text()); self.assertEqual(x['selected_checkpoint_step'],1585); self.assertEqual(x['candidate']['cursor'],25360)
   self.assertTrue(x['candidate_selection_pending']); self.assertEqual(x['tokenizer']['native_eog_ids'],[1,130073])
   self.assertNotEqual(x['tokenizer']['native_tokenizer_json_sha256'],x['tokenizer']['cloud_checkpoint_tokenizer_json_sha256'])
 def test_pending_1902_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=subprocess.run([sys.executable,str(BINDER),'--registry',str(REG),'--candidate','1902','--review-output',str(Path(d)/'x')],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0); self.assertIn('pending_verified_readback',p.stderr)
 def test_admission_must_match_review(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); r=d/'review'; adm=d/'adm'; out=d/'binding'
   subprocess.run([sys.executable,str(BINDER),'--registry',str(REG),'--candidate','1585','--review-output',str(r)],check=True,capture_output=True)
   x=json.loads(r.read_text()); adm.write_text(json.dumps({'schema':'sepalith.cloud-cpt-parent-merge.root-admission.v1','status':'admitted','action':'merge_cloud_cpt_parent_candidate','selected_checkpoint_step':1585,'binding_review_sha256':'0'*64,'authorized_output':x['output'],'candidate_remains_unselected_after_merge':True,'cuda_authorized':False,'final_promotion_authorized':False}))
   p=subprocess.run([sys.executable,str(BINDER),'--registry',str(REG),'--candidate','1585','--review-output',str(d/'review2'),'--root-admission',str(adm),'--binding-output',str(out)],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0); self.assertFalse(out.exists())
 def test_source_never_loads_checkpoint_tokenizer(self):
  s=(ROOT/'source/merge_cloud_cpt_parent.py').read_text()
  self.assertNotIn("AutoTokenizer.from_pretrained(str(checkpoint)",s)
  self.assertIn("shutil.copyfile(base/name,temporary/name)",s)
if __name__=='__main__': unittest.main()
