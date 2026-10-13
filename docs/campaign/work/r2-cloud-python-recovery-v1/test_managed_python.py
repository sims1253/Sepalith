import json,os,subprocess,tempfile,unittest
from pathlib import Path
from managed_python import discover_result,validate_runtime,layout_diagnostics,PROBE
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.install=self.root/'cpython-3.10.19';(self.install/'bin').mkdir(parents=True);self.exe=self.install/'bin/python3.10';self.exe.write_text('fixture');self.exe.chmod(0o700);(self.install/'Python.h').write_text('fixture')
  self.record={'version':[3,10,19],'implementation':'cpython','executable':str(self.exe),'prefix':str(self.install),'header':str(self.install/'Python.h')}
 def tearDown(self):self.t.cleanup()
 def test_duplicate_aliases(self):
  (self.root/'cpython-3.10').symlink_to(self.install,target_is_directory=True)
  self.assertEqual(layout_diagnostics(self.root)['legacy_glob_count'],2);self.assertEqual(layout_diagnostics(self.root)['legacy_unique_resolved_count'],1)
  self.assertEqual(discover_result(str(self.root/'cpython-3.10/bin/python3.10')+'\n',self.root),self.exe)
  self.assertTrue(validate_runtime(self.record,self.exe,self.root))
 def test_missing(self):
  with self.assertRaises((ValueError,FileNotFoundError)):discover_result(str(self.root/'missing'),self.root)
 def test_multiple_output(self):
  with self.assertRaises(ValueError):discover_result(str(self.exe)+'\n'+str(self.exe),self.root)
 def test_wrong_version(self):
  self.record['version']=[3,10,18]
  with self.assertRaises(ValueError):validate_runtime(self.record,self.exe,self.root)
 def test_outside_symlink(self):
  self.exe.unlink();self.exe.symlink_to('/usr/bin/python3')
  with self.assertRaises(ValueError):discover_result(str(self.exe),self.root)
 def test_outside_header(self):
  self.record['header']='/usr/include'
  with self.assertRaises(ValueError):validate_runtime(self.record,self.exe,self.root)
 def test_missing_header(self):
  (self.install/'Python.h').unlink()
  with self.assertRaises((ValueError,FileNotFoundError)):validate_runtime(self.record,self.exe,self.root)
 def test_wrong_implementation(self):
  self.record['implementation']='pypy'
  with self.assertRaises(ValueError):validate_runtime(self.record,self.exe,self.root)
 def test_actual_pinned_runtime(self):
  root=Path(__file__).parent.resolve()/'repro/python';e=next(root.glob('cpython-3.10.19-*/bin/python3.10')).resolve()
  result=subprocess.run([str(e),'-I','-c',PROBE],text=True,capture_output=True,check=True,timeout=10)
  self.assertTrue(validate_runtime(json.loads(result.stdout),e,root))
if __name__=='__main__':unittest.main(verbosity=2)
