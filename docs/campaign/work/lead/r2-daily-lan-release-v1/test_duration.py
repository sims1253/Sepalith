import unittest
from unittest.mock import patch
import daily_lan as d
class Duration(unittest.TestCase):
 def test_invalid_duration_rejects_before_any_artifact_or_resource_access(self):
  with patch.object(d,"check_manifest",side_effect=AssertionError("unexpected artifact access")):
   for seconds in (0,59,28801,-1,True,1.5):
    with self.assertRaisesRegex(ValueError,"session_seconds"):
     d.run_session("/unused","/unused",seconds)
 def test_source_manifest(self):
  d.check_manifest()
