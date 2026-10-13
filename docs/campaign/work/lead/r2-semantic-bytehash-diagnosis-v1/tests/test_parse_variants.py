import collections
import hashlib
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'source/analyze_semantics.py'
spec = importlib.util.spec_from_file_location('variant_analyzer', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ParseVariants(unittest.TestCase):
    def test_same_path_distinct_reconstructions_remain_separate(self):
        pins, groups = {}, collections.defaultdict(list)
        for rid, payload in [('a', b'f <- function(x) x\n'), ('b', b'f <- function(y) y\n'), ('c', b'f <- function(x) x\n')]:
            item = {'row_id': rid, 'parse_bytes_sha256': hashlib.sha256(payload).hexdigest()}
            module.register_source_variant(pins, groups, '/same/source.R', payload, item, 1, 'f')
        self.assertEqual(len(pins), 2)
        self.assertEqual(sorted(sorted(r['row_id'] for r in value['rows']) for value in pins.values()), [['a', 'c'], ['b']])
        for key, value in pins.items():
            self.assertEqual(hashlib.sha256(value['parse_bytes']).hexdigest(), key[1])

    def test_mismatched_bytes_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'bytes/hash'):
            module.register_source_variant({}, collections.defaultdict(list), '/same/source.R', b'wrong', {'row_id':'a','parse_bytes_sha256':'0'*64}, 1, 'f')
