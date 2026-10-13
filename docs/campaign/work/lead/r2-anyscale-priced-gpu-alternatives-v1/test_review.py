import json,unittest
from pathlib import Path
P=Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):self.r=json.loads((P/'cost-review.json').read_text());self.a=json.loads((P/'account-probe.json').read_text());self.o=json.loads((P/'official-evidence.json').read_text())
 def test_shape_and_memory_are_exact(self):
  self.assertEqual((self.r['route']['instance_type'],self.r['route']['gpu_count'],self.r['route']['gpu']),('p4d.24xlarge',8,'A100-40G'));self.assertEqual(self.o['aws']['p4d_24xlarge']['aggregate_gpu_memory_gb'],320)
 def test_category_is_not_misrepresented_as_node_quote(self):
  self.assertFalse(self.r['pricing']['public_anyscale_category']['exact_p4d_node_quote']);self.assertIsNone(self.r['pricing']['exact_account_all_in_p4d_rate']);self.assertIsNone(self.r['pricing']['exact_fundable_hours'])
 def test_fleet_zero_is_not_entitlement_claim(self):
  self.assertEqual(self.a['filtered_gpu_status']['record_count'],0);self.assertIsNone(self.r['availability']['account_enabled']);self.assertIsNone(self.r['availability']['current_capacity'])
 def test_no_mutation_or_spend(self):
  self.assertEqual(self.a['mutations'],0);self.assertFalse(self.r['safety']['paid_allocation']);self.assertEqual(self.r['budget_decision']['maximum_charge_usd'],0);self.assertFalse(self.r['budget_decision']['paid_launch_authorized'])
if __name__=='__main__':unittest.main()
