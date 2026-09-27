import unittest, tempfile, subprocess, sys, os, json, signal, time
from pathlib import Path
from unittest.mock import patch
import daily_lan as d
import notebook_setup as n
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
class Lifecycle(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.p=Path(self.temp.name);self.lock=self.p/'lock';self.lock.touch();self.children=[]
 def tearDown(self):
  for p in self.children:
   if p.poll() is None:p.kill()
   p.wait(timeout=2)
  self.temp.cleanup()
 def child(self):
  p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);self.children.append(p);return p
 def test_exclusive_stable_lock_and_symlink(self):
  inode=self.lock.stat().st_ino
  with d.exclusive_lock(self.lock):
   with self.assertRaisesRegex(RuntimeError,'busy'):
    with d.exclusive_lock(self.lock):pass
  with d.exclusive_lock(self.lock):pass
  self.assertEqual(self.lock.stat().st_ino,inode)
  (self.p/'alias').symlink_to(self.lock)
  with self.assertRaises(OSError):
   with d.exclusive_lock(self.p/'alias'):pass
 def test_busy_and_unknown_ledger_rejects(self):
  d.ledger_free('| RTX 5090 | None | terminal |')
  for text in ['', '| RTX 5090 | lead | RL |','| RTX 5090 | unknown | x |','| RTX 5090 | None | x |\n| RTX 5090 | None | x |']:
   with self.assertRaises(RuntimeError):d.ledger_free(text)
 def test_pid_reuse_guard_never_signals_mismatched_process(self):
  child=self.child();ident=d.identity(child.pid);wrong={**ident,'startTick':str(int(ident['startTick'])+1)}
  self.assertFalse(d.signal_owned(wrong,signal.SIGTERM));self.assertIsNone(child.poll())
  self.assertTrue(d.signal_owned(ident,signal.SIGTERM));child.wait(timeout=2)
  self.assertFalse(d.signal_owned(ident,signal.SIGTERM))
 def test_cleanup_only_owned_and_start_failure_detected(self):
  bystander=self.child()
  with d.exclusive_lock(self.lock) as fd:
   owner=d.Owned(self.p,fd);child=owner.launch('synthetic',[sys.executable,'-c','import time;time.sleep(20)']);owner.check()
   rows=owner.cleanup();self.assertFalse(rows[0]['survived']);self.assertIsNone(bystander.poll());self.assertIsNotNone(child.poll())
   with self.assertRaises(RuntimeError):owner.check()
 def test_completed_children_retained_in_terminal(self):
  with d.exclusive_lock(self.lock) as fd:
   owner=d.Owned(self.p,fd);child=owner.launch('synthetic',[sys.executable,'-c','import time;time.sleep(.05)']);child.wait(timeout=2);owner.complete(child);owner.check()
   self.assertEqual(owner.cleanup()[0]['identity']['pid'],child.pid)
 def test_parent_death_releases_owned_child_and_lock(self):
  code='import daily_lan as d,sys,time;from pathlib import Path\nwith d.exclusive_lock(Path(sys.argv[1])) as fd:\n o=d.Owned(Path(sys.argv[2]),fd);o.launch("orphan-test",[sys.executable,"-c","import time;time.sleep(20)"]);time.sleep(20)'
  parent=subprocess.Popen([sys.executable,'-c',code,str(self.lock),str(self.p)],cwd=Path(__file__).parent);self.children.append(parent)
  end=time.monotonic()+2
  while not (self.p/'orphan-test-identity.json').exists() and time.monotonic()<end:time.sleep(.01)
  ident=json.loads((self.p/'orphan-test-identity.json').read_text());parent.kill();parent.wait(timeout=2)
  while d.identity(ident['pid']) is not None and time.monotonic()<end:time.sleep(.01)
  self.assertIsNone(d.identity(ident['pid']))
  with d.exclusive_lock(self.lock):pass
 def test_selected_native_and_ssh_arguments(self):
  a=d.native_argv()
  for flag,value in [('-c','4096'),('-b','256'),('-ub','256'),('--parallel','1'),('--host','127.0.0.1'),('-ngl','99')]:self.assertEqual(a[a.index(flag)+1],value)
  a=d.ssh_argv('-N','-R','127.0.0.1:18403:127.0.0.1:18423',d.HOST)
  for value in ['BatchMode=yes','StrictHostKeyChecking=yes','ExitOnForwardFailure=yes','127.0.0.1:18403:127.0.0.1:18423','m0hawk@192.168.178.40']:self.assertIn(value,a)
 def test_binding_identity_and_atomic_receiver(self):
  m=json.loads((d.HERE/'primary-manifest.json').read_text());v={'schema':1,'endpoint':'http://127.0.0.1:18403','instanceId':'f226169e-a888-4cd2-b2ad-412d3212062f','manifest':m,'backend':'cuda'}
  self.assertEqual(n.validated_binding(v),v)
  for bad in [{**v,'endpoint':'http://192.168.178.1:18403'},{**v,'instanceId':'bad'},{**v,'manifest':{}},{**v,'backend':'cpu'}]:
   with self.assertRaises(ValueError):n.validated_binding(bad)
  with patch.object(n,'ROOT',self.p):
   (self.p/'user-data/User').mkdir(parents=True);(self.p/'user-data/User/settings.json').write_text('{}')
   n.receive(json.dumps(v).encode());self.assertEqual(json.loads((self.p/'current-binding.json').read_text()),v)
   with self.assertRaises(ValueError):n.receive(b'x'*(1024*1024+1))
   self.assertEqual(json.loads((self.p/'current-binding.json').read_text()),v)
 def test_root_deployment_initial_ledger_and_reusable_admission(self):
  here=self.p/'package';here.mkdir();(here/'source-manifest.json').write_text('{}')
  ledger=self.p/'ledger';ledger.write_text('| RTX 5090 | None | released A |')
  state=self.p/'state';state.mkdir();admission=self.p/'admission.json'
  value={'schema':1,'status':'admitted','purpose':'daily-lan-deployment','sourceManifestSha256':d.sha(here/'source-manifest.json'),'cudaLockPath':str(d.LOCK),'initialLeaseLedgerSha256':d.sha(ledger)}
  admission.write_text(json.dumps(value))
  with patch.object(d,'HERE',here),patch.object(d,'LEDGER',ledger):
   d.check_admission(admission,state)
   ledger.write_text('| RTX 5090 | None | released B |');d.check_admission(admission,state)
   ledger.write_text('| RTX 5090 | lead | active final job |')
   with self.assertRaisesRegex(RuntimeError,'busy'):d.check_admission(admission,state)
   ledger.write_text('| RTX 5090 | None | released B |')
   value['status']='pending';admission.write_text(json.dumps(value))
   with self.assertRaisesRegex(RuntimeError,'admission_required'):d.check_admission(admission,state)
 def test_busy_lock_blocks_before_admission_or_artifact_reads(self):
  original=d.exclusive_lock
  with original(self.lock):
   with patch.object(d,'check_manifest'),patch.object(d,'exclusive_lock',lambda:original(self.lock)),patch.object(d,'check_admission',side_effect=AssertionError('admission reached while busy')):
    with self.assertRaisesRegex(RuntimeError,'busy'):d.run_session(self.p/'state','unused')
 def test_no_normal_profile_or_instrumented_settings(self):
  s=n.settings();self.assertFalse(s['sepalith.debugMode']);self.assertFalse(s['sepalith.autoStart']);self.assertFalse(s['sepalith.scopeContext']);self.assertEqual(s['sepalith.debounceMs'],1500);self.assertEqual(s['sepalith.remoteBindingPath'],str(n.ROOT/'current-binding.json'))
if __name__=='__main__':unittest.main(verbosity=2)
