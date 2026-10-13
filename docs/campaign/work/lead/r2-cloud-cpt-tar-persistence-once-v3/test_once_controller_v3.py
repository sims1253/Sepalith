import ast,hashlib,importlib.util,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class ControllerTest(unittest.TestCase):
 def module(self):
  p=HERE/'root_tar_persistence_once_v3.py';spec=importlib.util.spec_from_file_location('controller_v3',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_nested_remote_compiles_and_exact_parent_identity_is_pinned(self):
  m=self.module();compile(m.REMOTE,'REMOTE','exec');self.assertEqual(m.PARENT_START,7757);self.assertEqual(m.WORKING_DIR,'s3_dc963855898b4fa1797b256f71b8711bd61b633c')
  self.assertIn("parts!=['python','-B',expected_entry,'binding.root.json']",m.REMOTE);self.assertIn("sha(submission_binding)!=binding_sha",m.REMOTE);self.assertNotIn("root+'/payload'",m.REMOTE);self.assertNotIn("root+'/binding.root.json'",m.REMOTE)
  self.assertNotIn('secret',m.safe_error('https://host/path?token=secret hf_abcdef'));self.assertIn('REDACTED',m.safe_error('https://host/path?token=secret hf_abcdef'))
 def test_authorization_disabled_and_start_tick_scoped(self):
  v=json.loads((HERE/'root-authorization-once.template.json').read_text());self.assertIs(v['authorized'],False);self.assertEqual(v['parent_start_tick'],7757);self.assertEqual(v['action'],'upload_checkpoint317_tar_once_v3')
 def test_opaque_transport_is_exact_frozen_v1_source(self):
  digest=hashlib.sha256((HERE/'tar_persistence_once_v3.py').read_bytes()).hexdigest();self.assertEqual(digest,'fc47085b0280f683929035f33be384c1cd85276ec113e20395526ebccba973b9')
if __name__=='__main__':unittest.main()
