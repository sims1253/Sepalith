import json,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import audit_metadata,shared_replay_compare
class T(unittest.TestCase):
 def test_audit_refuses_large_reads(self):
  with tempfile.TemporaryDirectory() as x:
   p=Path(x)/'large';p.write_bytes(b'0'*(16*1024*1024+1))
   with self.assertRaisesRegex(ValueError,'refusing large'):audit_metadata.read_json(p)
 def test_compare_exact_and_rejects_difference(self):
  with tempfile.TemporaryDirectory() as x:
   p=Path(x); d={'resume_from':'same','global_step':2,'model_state_sha256':'a','optimizer_state_sha256':'b','scheduler_state_sha256':'c','cpu_rng_sha256':'d','cuda_rng_sha256':'e','draw_cursor':{'x':1},'optimizer_dispatch_sha256':'f'}
   (p/'a').write_text(json.dumps(d));(p/'b').write_text(json.dumps(d));self.assertEqual(shared_replay_compare.compare(p/'a',p/'b')['status'],'exact_match')
   d['cuda_rng_sha256']='z';(p/'b').write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,'differ exactly'):shared_replay_compare.compare(p/'a',p/'b')
 def test_replay_source_enforces_shared_checkpoint_and_determinism(self):
  s=(P/'source/shared_checkpoint_replay.py').read_text()
  for text in ('resume = Path(resume_override)','torch.use_deterministic_algorithms(True)','set_seed(seed, deterministic=True)','CUBLAS_WORKSPACE_CONFIG','expected_lane_positions = [1]'):
   self.assertIn(text,s)
  self.assertNotIn('verify_checkpoint(checkpoint, identity(recipe)',s[s.index('class SealFullCheckpoint'):s.index('class StopAfterFirstSavedUpdate')])
 def test_restoration_capture_is_before_forward_and_compares_exact_state(self):
  s=(P/'verify_restoration.py').read_text()
  self.assertIn('def on_step_begin',s)
  self.assertIn("'optimizer_update_executed':False",s)
  self.assertIn('raise RestorationCaptured()',s)
  self.assertNotIn('.training_step(',s)
  for field in ('model_state_sha256','optimizer_state_sha256','scheduler_state_sha256','cpu_rng_sha256','cuda_rng_sha256','optimizer_dispatch_sha256'):
   self.assertIn(field,s)
if __name__=='__main__':unittest.main(verbosity=2)
