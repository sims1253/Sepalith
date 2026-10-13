from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from reward_buffer_binding import BindingError,RewardBufferIndex,sha_file

class BindingTest(unittest.TestCase):
 def packet(self,root:Path,supported=True):
  text='x <- 1\n';sha=hashlib.sha256(text.encode()).hexdigest();blob=root/f'baselines/{sha[:2]}/{sha}.R';blob.parent.mkdir(parents=True);blob.write_text(text)
  row={'position':0,'row_id':'train-1','supported':supported,'repair_reason':None if supported else 'repair',
       'buffer_mode':'complete_document','baseline_sha256':sha,'baseline_bytes':len(text.encode()),
       'baseline_blob':str(blob.relative_to(root)),'baseline_parse_ok':True,'gold_applied_sha256':sha,
       'gold_applied_parse_ok':True,'parser_identity':{'operation':'base::parse'}}
  side=root/'reward-buffer-sidecar.jsonl';side.write_text(json.dumps(row)+'\n')
  manifest={'status':'prepared_root_review_required','artifacts':{'reward-buffer-sidecar.jsonl':{'bytes':side.stat().st_size,'sha256':sha_file(side)}}}
  mp=root/'materialization.json';mp.write_text(json.dumps(manifest));return mp
 def test_positive_binding_and_blob_hash(self):
  with tempfile.TemporaryDirectory() as d:
   mp=self.packet(Path(d));idx=RewardBufferIndex.load(mp,sha_file(mp),['train-1']);env=idx.envelope_for('train-1')
   self.assertEqual(idx.baseline_text(env),'x <- 1\n')
 def test_order_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   mp=self.packet(Path(d))
   with self.assertRaises(BindingError):RewardBufferIndex.load(mp,sha_file(mp),['other'])
 def test_repair_refused_by_default(self):
  with tempfile.TemporaryDirectory() as d:
   mp=self.packet(Path(d),False)
   with self.assertRaises(BindingError):RewardBufferIndex.load(mp,sha_file(mp),['train-1'])
 def test_manifest_hash_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   mp=self.packet(Path(d))
   with self.assertRaises(BindingError):RewardBufferIndex.load(mp,'0'*64,['train-1'])
if __name__=='__main__':unittest.main()
