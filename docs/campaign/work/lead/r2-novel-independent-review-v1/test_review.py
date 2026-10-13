import json,unittest
from pathlib import Path
W=Path(__file__).resolve().parent
def jl(p):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
class ReviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(c):
  c.s=json.loads((W/'summary.json').read_text());c.a=jl(W/'admissible-ids.jsonl');c.l=jl(W/'audit-ledger.jsonl');c.r=json.loads((W/'source-line-replay.json').read_text())
 def test_denominator_partition_with_overlap(self):
  d=self.s['denominators'];self.assertEqual(d['review_rows'],5245);self.assertEqual(d['admissible_rows'],3503)
  self.assertEqual(d['admissible_rows']+d['permanent_duplicate_rows']+d['repair_queue_rows']-d['duplicate_and_repair_overlap'],5245)
 def test_admissible_ids_are_unique_finish_rows_without_reasons(self):
  self.assertEqual(len(self.a),3503);self.assertEqual(len({x['row_id'] for x in self.a}),3503);self.assertEqual({x['family'] for x in self.a},{'finish_block'})
  by={x['row_id']:x for x in self.l};self.assertTrue(all(not by[x['row_id']]['reasons'] for x in self.a))
 def test_na_rm_increment_is_entirely_non_novel(self):
  new=self.l[5109:];self.assertEqual(len(new),136);self.assertEqual({x['family'] for x in new},{'na_rm_propagation'})
  self.assertTrue(all('accepted_row_id_duplicate' in x['reasons'] and 'accepted_prompt_target_duplicate' in x['reasons'] for x in new))
 def test_raw_sources_and_frozen_prefix(self):
  self.assertEqual((self.r['requested_rows'],self.r['matched_rows'],self.r['mismatched_or_missing_rows']),(5245,5245,0));self.assertTrue(self.s['checks']['v4_exact_v3_prefix'])
 def test_no_heldout_or_geometry_failure(self):
  self.assertEqual(self.s['reason_rows'].get('cpt_validation_reserved',0),0);self.assertFalse(any('geometry' in reason and reason!='roxygen_geometry_or_target_contradiction' for reason in self.s['reason_rows']))
if __name__=='__main__':unittest.main(verbosity=2)
