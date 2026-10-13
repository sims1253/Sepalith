import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('g',P/'recover_geometry.py');G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
ORIGINAL=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl');SEMANTIC=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2/selected-01/selected-contexts.jsonl')
def first(p):
 with p.open() as stream:return json.loads(next(stream))
class TestGeometry(unittest.TestCase):
 def test_actual_full_snapshot_schema_and_utf16_range(self):
  x=first(ORIGINAL);out=G.recover(x,'context',x['row_id']);self.assertEqual(out['status'],'recovered');self.assertTrue(out['full_buffer_hash_verified']);self.assertEqual(out['source_cursor_geometry']['cursor'],x['context']['replacement_range']['start']);self.assertEqual(out['source_cursor_geometry']['replacement_range']['end']['character'],43)
 def test_actual_selected_context_schema_does_not_claim_full_buffer(self):
  x=first(SEMANTIC);out=G.recover(x,'selected_context',x['row_id']);self.assertEqual(out['status'],'recovered_context_geometry_buffer_unavailable');self.assertFalse(out['full_buffer_hash_verified']);self.assertEqual(out['source_cursor_geometry']['cursor'],{'line':352,'character':0})
 def test_row_cursor_and_hash_tamper_rejected(self):
  x=first(ORIGINAL)
  with self.assertRaisesRegex(G.RecoveryError,'row_id_join'):G.recover(x,'context','other')
  y=copy.deepcopy(x);y['context']['replacement_range']['start']['character']=-1
  with self.assertRaisesRegex(G.RecoveryError,'start'):G.recover(y,'context',y['row_id'])
  y=copy.deepcopy(x);y['context']['replacement_range']['content_sha256']='0'*64
  with self.assertRaisesRegex(G.RecoveryError,'claimed_full_snapshot_hash_mismatch'):G.recover(y,'context',y['row_id'])
 def test_target_selection_attestation_rejected(self):
  x=first(SEMANTIC);x['selection_target_or_gold_used']=True
  with self.assertRaisesRegex(G.RecoveryError,'target_used'):G.recover(x,'selected_context',x['row_id'])
if __name__=='__main__':unittest.main()
