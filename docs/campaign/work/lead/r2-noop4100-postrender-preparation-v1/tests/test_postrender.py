import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('v',P/'source/verify_render16.py');V=importlib.util.module_from_spec(s);s.loader.exec_module(V)
class T(unittest.TestCase):
 def test_id_hash_order(self):
  self.assertNotEqual(V.idsha(['a','b']),V.idsha(['b','a']))
 def test_rows_reject_duplicate_and_malformed(self):
  with tempfile.TemporaryDirectory(dir='/mnt/e/sepalith/campaign-20260915/tmp') as d:
   p=Path(d)/'x';p.write_text('{"row_id":"a"}\n{"row_id":"a"}\n')
   with self.assertRaisesRegex(V.Error,'duplicate'):V.rows(p)
   p.write_text('{bad}\n')
   with self.assertRaisesRegex(V.Error,'jsonl'):V.rows(p)
 def test_fixed_denominators_and_source_pins(self):
  self.assertEqual(V.EXPECTED,tuple(range(10,41)));self.assertEqual(V.PLAN_SHA,'093fd437f6cc4a364b7273e7936459c836a18a31bc06fd84ec39ec88faacd725');self.assertEqual(V.INPUT_MANIFEST_SHA,'38736227196ba22a5c55d1826411a20a1a50a8f3db277bb9c36ead2b696a9e65')
if __name__=='__main__':unittest.main()
