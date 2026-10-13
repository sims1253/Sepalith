import math
import unittest
from scoring import windows, summarize


class ScoringTests(unittest.TestCase):
    def test_every_token_once_including_first_and_tail(self):
        for length in [0, 1, 2, 7, 8, 9, 15, 64, 2051]:
            ids = list(range(100, 100 + length)); targets = []
            for seq, first, count in windows(ids, 0, 8, 3):
                self.assertLessEqual(len(seq), 8)
                self.assertGreaterEqual(first, 1)
                self.assertEqual(count, len(seq) - first)
                targets.extend(seq[first:])
            self.assertEqual(targets, ids)

    def test_byte_denominator_and_no_bos_count(self):
        result = summarize(6 * math.log(2), 3, 3, len('aé'.encode()))
        self.assertAlmostEqual(result['bits_per_byte'], 2)
        self.assertEqual(result['tokens'], 3)

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError): summarize(1, 3, 4, 8)

    def test_empty_and_nonfinite_rejected(self):
        for args in [(0,0,0,0),(math.inf,1,1,1),(math.nan,1,1,1)]:
            with self.assertRaises(ValueError): summarize(*args)

if __name__ == '__main__': unittest.main()
