import json, math, pathlib, unittest
HERE=pathlib.Path(__file__).parent
E=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Noop-pool-coverage-review-v1')
class ReviewTests(unittest.TestCase):
 def setUp(self): self.r=json.loads((HERE/'review.json').read_text())
 def test_train_inventory_partition_is_exact(self):
  x=self.r['inventory_reconciliation']
  self.assertTrue(x['reconciles_inventory_train_cpt_denominator'])
  self.assertEqual(10677, x['current_inventory_overlap']+x['old_gate_materialized_rows']+x['early_alternate_kind_support_queue_rows']+x['later_source_walk_converted_rows']+x['later_alternate_kind_support_queue_rows']+len(x['unresolved_metadata_only_rows']))
 def test_supported_pool_is_disjoint_and_validated(self):
  x=json.loads((E/'supported-noop-summary.json').read_text())
  self.assertEqual(4227,x['unique']['row_ids']); self.assertEqual(0,x['validation_failure_count'])
  self.assertEqual({'current15006_ids':0,'old_gate_ids':0,'support_queue_ids':0},x['overlap'])
  self.assertEqual(x['counts']['rows']-x['duplicates_within']['context_sha256'],x['unique']['contexts'])
 def test_projection_math_and_replay_accounting(self):
  for x in self.r['projections'].values():
   self.assertEqual(x['unique_rows'],x['unique_noops']+x['unique_edits'])
   self.assertEqual(x['scheduled_draws'],16*x['minimum_4_noop_12_edit_batches_for_all_edits'])
   self.assertEqual(x['scheduled_noop_draws'],4*x['minimum_4_noop_12_edit_batches_for_all_edits'])
   self.assertEqual(x['scheduled_noop_draws'],x['unique_noops']+x['noop_replay_draws_after_unique_coverage'])
 def test_no_heldout_admission(self):
  self.assertEqual('independent_review_complete_no_training_admission',self.r['status'])
  self.assertTrue(self.r['scope']['train_only']);self.assertFalse(self.r['scope']['dev_opened']);self.assertFalse(self.r['scope']['final_opened'])
if __name__=='__main__': unittest.main()
