import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'prepare_binding_v2.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def candidate(step=1902):return {'schema':'sepalith.cloud-cpt-parent-merge.candidates.v1','candidates':{str(step):{'status':'verified_readback','step':step,'full':True,'cursor':30432,'checkpoint_path':'/x','campaign_manifest_sha256':'a'*64,'campaign_state_sha256':'b'*64,'adapter_config_sha256':'c'*64,'adapter_weights_sha256':'d'*64,'checkpoint_tokenizer_json_sha256':'d5ede0bcd21e0676a58b176937262fb80a06a32b5cb4ed8bfed8b7d11e45b0e1','source_identity':'7e4e80059416b3783c783a532d95b40af6067076e7a0a23d64e3b571f478c9df','draw_schedule_sha256':'914cf353199f45697c028da3a6f06b732fd57bd265869890b75bf43151851b58','train_rows_sha256':'836b61c5b7d5545063aeaa3054c62915b8a4d24574908cade4c2d3eed1537a87','verified_readback_receipt':{'path':'/r','sha256':'e'*64}}}}
class T(unittest.TestCase):
 def test_alternate_registry_and_merge_source_are_exactly_bound(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td); reg=td/'alternate-1902.json'; merge=td/'reviewed-merge.py'; out=td/'review.json';reg.write_text(json.dumps(candidate()));merge.write_text('# exact reviewed merge\n')
   subprocess.run([sys.executable,str(B),'--registry',str(reg),'--merge-source',str(merge),'--candidate','1902','--review-output',str(out)],check=True,capture_output=True)
   x=json.loads(out.read_text());self.assertEqual(x['candidate_registry_path'],str(reg.resolve()));self.assertEqual(x['candidate_registry_sha256'],sha(reg));self.assertEqual(x['merge_source_path'],str(merge.resolve()));self.assertEqual(x['merge_source_sha256'],sha(merge))
 def test_same_candidate_different_registry_bytes_changes_review_identity(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);merge=td/'m';merge.write_text('x');ids=[]
   for i in (1,2):
    reg=td/f'r{i}'; x=candidate();x['annotation']=i;reg.write_text(json.dumps(x));out=td/f'o{i}';subprocess.run([sys.executable,str(B),'--registry',str(reg),'--merge-source',str(merge),'--candidate','1902','--review-output',str(out)],check=True,capture_output=True);ids.append(json.loads(out.read_text())['binding_review_sha256'])
   self.assertNotEqual(ids[0],ids[1])
 def test_missing_merge_source_fails_before_review(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);reg=td/'r';reg.write_text(json.dumps(candidate()));out=td/'o';p=subprocess.run([sys.executable,str(B),'--registry',str(reg),'--merge-source',str(td/'absent'),'--candidate','1902','--review-output',str(out)],capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
