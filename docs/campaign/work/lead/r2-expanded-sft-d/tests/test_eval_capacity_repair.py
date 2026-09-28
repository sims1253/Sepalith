"""Regression checks for independent generation and reference capacities."""
import sys, unittest
from pathlib import Path
TRAIN=Path(__file__).resolve().parents[1]/'source/experiments/training';sys.path.insert(0,str(TRAIN))
from campaign_eval import validate_development_capacity
class CapacityTests(unittest.TestCase):
 def test_complete_reference_longer_than_generation_cap_is_retained(self):
  row={'input_ids':list(range(800)),'target_start':300}
  self.assertEqual(validate_development_capacity(row,4096,192,'long'),500)
 def test_exact_context_boundary(self): self.assertEqual(validate_development_capacity({'input_ids':[0]*4096,'target_start':3904},4096,192,'edge'),192)
 def test_reference_context_overflow_rejected(self):
  with self.assertRaises(ValueError): validate_development_capacity({'input_ids':[0]*4097,'target_start':100},4096,192,'ref')
 def test_generation_context_overflow_rejected(self):
  with self.assertRaises(ValueError): validate_development_capacity({'input_ids':[0]*4000,'target_start':4000},4096,192,'gen')
if __name__=='__main__':unittest.main(verbosity=2)
