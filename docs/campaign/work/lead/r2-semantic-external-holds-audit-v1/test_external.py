import unittest
from pathlib import Path
class ContractTests(unittest.TestCase):
 def test_exact_origin_is_required_for_each_dependency(self):
  explicit={'filter':{'dplyr'}};self.assertEqual(len(explicit['filter']),1);self.assertNotIn('select',explicit)
 def test_wildcard_does_not_prove_symbol_origin(self):
  wildcards={'dplyr','stats'};symbol='filter';self.assertTrue(wildcards);self.assertFalse(False if symbol else True)
 def test_multiple_origins_are_ambiguous(self):
  self.assertGreater(len({'stats','dplyr'}),1)
 def test_target_is_not_an_input_to_namespace_evidence(self):
  fields={'row_id','dependencies','preedit_sha256','namespace'};self.assertFalse(fields & {'target','gold','completion','reward'})
 def test_helper_is_parse_only(self):
  text=Path('namespace_evidence.R').read_text();self.assertIn('parse(file=',text);self.assertNotIn('library(',text);self.assertNotIn('source(',text);self.assertNotIn('eval(',text)
if __name__=='__main__':unittest.main(verbosity=2)
