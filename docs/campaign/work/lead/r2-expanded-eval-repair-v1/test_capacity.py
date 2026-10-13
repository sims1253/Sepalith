import sys
from pathlib import Path
import unittest
W=Path(__file__).resolve().parent
S=W.parent/'r2-expanded-sft-c/source/experiments/training'
sys.path[:0]=[str(W),str(S)]
from campaign_eval import validate_development_capacity
class CapacityTests(unittest.TestCase):
    def test_complete_reference_longer_than_generation_cap_is_retained(self):
        row={'input_ids':list(range(800)), 'target_start':300}
        self.assertEqual(validate_development_capacity(row,4096,192,'long'),500)
        self.assertEqual(len(row['input_ids']),800)
    def test_exact_context_boundary(self):
        self.assertEqual(validate_development_capacity({'input_ids':[0]*4096,'target_start':3904},4096,192,'edge'),192)
    def test_reference_context_overflow_rejected(self):
        with self.assertRaises(ValueError):validate_development_capacity({'input_ids':[0]*4097,'target_start':100},4096,192,'ref-overflow')
    def test_generation_context_overflow_rejected(self):
        with self.assertRaises(ValueError):validate_development_capacity({'input_ids':[0]*4000,'target_start':4000},4096,192,'gen-overflow')
