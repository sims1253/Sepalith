import ast,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('v2',HERE/'root_stop_and_upload_once_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_actual_nested_sources_compile(self):
  compile(m.CHILD,'child','exec');compile(m.REMOTE,'remote','exec');self.assertNotIn('\\nfrom pathlib',m.CHILD)
 def test_actual_child_sanitizer_redacts_url_and_credentials(self):
  tree=ast.parse(m.CHILD);nodes=[n for n in tree.body if (isinstance(n,ast.Import) and all(a.name!='artifact_upload' for a in n.names)) or isinstance(n,ast.ImportFrom) or isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='SECRET' for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='safe']
  ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'safe','exec'),ns)
  text=ns['safe']('hf_ABC bearer qwerty https://host/path?X-Amz-Signature=SECRET&token=BAD')
  self.assertNotIn('hf_ABC',text);self.assertNotIn('qwerty',text);self.assertNotIn('SECRET',text);self.assertNotIn('BAD',text);self.assertIn('REDACTED_QUERY',text)
 def test_probe_precedes_signal_and_argv_python_preserved(self):
  self.assertIn('python=parts[0]',m.REMOTE);self.assertNotIn("realpath(base+'/exe')",m.REMOTE)
  self.assertLess(m.REMOTE.index('probe=subprocess.run'),m.REMOTE.index('os.kill(pid,signal.SIGTERM)'))
  self.assertIn("import artifact_upload,huggingface_hub",m.REMOTE)
 def test_false_default_and_exact_authorization(self):
  value=json.loads((HERE/'root-authorization.template.json').read_text())
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a';p.write_text(json.dumps(value))
   with self.assertRaises(ValueError):m.auth(p)
   value['authorized']=True;p.write_text(json.dumps(value));self.assertTrue(m.auth(p)['authorized'])
 def test_only_exact_sidecar_signal_and_one_upload(self):
  self.assertNotIn('SIGKILL',m.REMOTE);self.assertEqual(m.REMOTE.count('os.kill('),1);self.assertEqual(m.CHILD.count('upload_checkpoint('),1);self.assertIn('timeout=900',m.REMOTE)
if __name__=='__main__':unittest.main(verbosity=2)
