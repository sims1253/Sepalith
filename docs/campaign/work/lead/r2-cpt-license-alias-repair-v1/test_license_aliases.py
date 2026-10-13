import re
import unittest
from license_aliases import normalize_recognized_alias

def original_gate(value):
    return bool(re.search(r"(^|[ |+,(])(?:A?GPL|LGPL|MIT|BSD|Apache|Artistic|MPL|CC0|Unlimited)(?:[- (]|$)", value))

class AliasTests(unittest.TestCase):
    def test_standard_bsd_spellings_recover(self):
        for value in ["BSD_2_clause + file LICENSE", "BSD_3_clause + file LICENCE"]:
            self.assertFalse(original_gate(value))
            self.assertTrue(original_gate(normalize_recognized_alias(value)))
    def test_long_names_map_existing_families(self):
        for value in ["GNU General Public License", "Mozilla Public License 2.0", "Mozilla Public License Version 2.0"]:
            self.assertTrue(original_gate(normalize_recognized_alias(value)))
    def test_custom_and_restricted_names_not_promoted(self):
        for value in ["BSD_3_clause_custom", "MyBSD_2_clause", "EUPL", "CC BY-NC 4.0", "file LICENSE", "BSL-1.0"]:
            self.assertFalse(original_gate(normalize_recognized_alias(value)))
    def test_existing_recognized_inputs_unchanged(self):
        for value in ["GPL (>= 2)", "MIT + file LICENSE", "Apache License (== 2.0)", "MPL-2.0", "BSD-3-clause", "LGPL-2.1"]:
            self.assertEqual(normalize_recognized_alias(value), value)
            self.assertTrue(original_gate(value))
if __name__ == "__main__": unittest.main()
