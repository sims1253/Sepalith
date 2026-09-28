import importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('repair',HERE/'root_stop_and_upload_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def write(self,x):
  d=tempfile.TemporaryDirectory();p=Path(d.name)/'a.json';p.write_text(json.dumps(x));return d,p
 def test_false_template_rejected(self):
  x=json.loads((HERE/'root-authorization.template.json').read_text());d,p=self.write(x)
  with self.assertRaises(ValueError):m.auth(p)
  d.cleanup()
 def test_exact_authorization_only(self):
  x=json.loads((HERE/'root-authorization.template.json').read_text());x['authorized']=True;d,p=self.write(x);self.assertTrue(m.auth(p)['authorized']);x['sidecar_start_tick']+=1;p.write_text(json.dumps(x))
  with self.assertRaises(ValueError):m.auth(p)
  d.cleanup()
 def test_remote_scope_and_secret_safety(self):
  self.assertIn("os.kill(pid,signal.SIGTERM)",m.REMOTE);self.assertNotIn('SIGKILL',m.REMOTE);self.assertIn("timeout=900",m.REMOTE);self.assertIn("HF_TOKEN':token",m.REMOTE)
  self.assertNotIn("'token':",m.REMOTE);self.assertIn('safe_message(e)',m.REMOTE);self.assertIn('safe_stack(e)',m.REMOTE)
 def test_exact_source_and_process_pins(self):
  for v in ('5498','18486','0b3d4df5e4a0ffd5ea679ad58fe709a0b2126230cdffa4b7f3c77719c6400511','a055094b5ab39c35e10b32acb4938ff56537946a6e55ed8785612360b6aaf146','c31671d5e3895a394ee569b0d0537ebe81d636daa2dbbaad4a11fa6559d76dc5'):
   self.assertIn(v,m.REMOTE)
if __name__=='__main__':unittest.main(verbosity=2)
