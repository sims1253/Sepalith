import json
from pathlib import Path
import tempfile
import unittest

from compare_quants import compare, load


class ComparisonTests(unittest.TestCase):
    def test_paired_changes_preserve_denominators(self):
        a = {'one': {'exact': 1, 'valid_pass': 1}, 'two': {'exact': 0, 'valid_pass': 1}}
        b = {'two': {'exact': 1, 'valid_pass': 1}, 'one': {'exact': 0, 'valid_pass': 0}}
        result = compare(a, b)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['exact']['delta_pp'], 0)
        self.assertEqual(result['exact']['lost'], 1)
        self.assertEqual(result['exact']['gained'], 1)
        self.assertEqual(result['valid_pass']['delta_pp'], -50)

    def test_invalid_pairs_do_not_produce_quality_claim(self):
        valid = {'a': {'exact': 1, 'valid_pass': 1}}
        for invalid in ({}, {'b': valid['a']}, {'a': {'exact': 0.5, 'valid_pass': 1}},
                        {'a': {'exact': 1}}, {'a': {**valid['a'], 'fail_kind': 'error'}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare(valid, invalid)

    def test_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            row = json.dumps({'id': 'a', 'exact': 1, 'valid_pass': 1}) + '\n'
            path.write_text(row * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load(path)
