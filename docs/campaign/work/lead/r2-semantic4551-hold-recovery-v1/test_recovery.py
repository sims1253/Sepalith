import hashlib,json,unittest
from pathlib import Path
E=Path('/mnt/e/sepalith/campaign-20260915/data-work');HERE=Path(__file__).parent;RID='407f72b3ac171a7e9f5e8e5f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class RecoveryTest(unittest.TestCase):
 def test_target_free_fixed_profile_and_complete_target(self):
  inp=json.loads((E/'Semantic4551-hold-recovery-v1/prediction-input.jsonl').read_text());render=json.loads((E/'Semantic4551-hold-recovery-v1/render-2048.jsonl').read_text());row=json.loads((E/'Semantic4551-hold-recovery-v1/final-01/candidate-tokenrows.jsonl').read_text());manifest=json.loads((E/'Semantic4551-hold-recovery-v1/final-01/manifest.json').read_text())
  self.assertEqual(inp['row_id'],RID);self.assertFalse(any(any(word in k.lower() for word in ('target','gold','completion','reward')) for k in inp));self.assertEqual(render['row_id'],RID);self.assertEqual(render['status'],'supported');self.assertFalse(render['selection_target_or_gold_used']);self.assertEqual((render['context_size'],render['generation_reserve'],render['prompt_tokens']),(16384,2048,433));self.assertEqual(render['prompt_sha256'],'16b9be7d5ff5b0ad32c749b5193e4fac3d62eef4fb337d06a99d25ae38711508');self.assertEqual(row['target_token_count'],1675);self.assertLessEqual(row['target_token_count'],2048);self.assertEqual(manifest['dedup_denominator'],20190);self.assertTrue(manifest['checks']['source_reapplication']);self.assertFalse(manifest['training_admission'])
 def test_exact_minimum_and_standard_reserve(self):
  prompt=433;target=1675;self.assertLessEqual(1+prompt+target,16384);self.assertGreater(target,1024);self.assertGreater(target,1674);self.assertLessEqual(target,2048)
 def test_hold_accounting(self):
  x=json.loads((HERE/'hold-categories.json').read_text());self.assertEqual(x['denominator'],115);self.assertEqual(sum(v['count'] for v in x['categories'].values()),115);self.assertEqual(x['context_length_holds_after_32k_fallback'],0);self.assertEqual(x['separate_fixed_reserve_recovery']['row_id'],RID)
 def test_frozen_predecessor_receipt(self):
  receipt=HERE.parents[2]/'receipts/DAT-10-semantic4551-provider-materialization.json';self.assertEqual(sha(receipt),'0c624d0f65a25f49d52edd1f3b20856e1c4d1663f61dce600d093328b850d120')
if __name__=='__main__':unittest.main()
