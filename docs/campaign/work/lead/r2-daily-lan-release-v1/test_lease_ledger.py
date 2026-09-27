import unittest,tempfile,os,json,socket,stat
from pathlib import Path
from unittest.mock import patch
import daily_lan as d
import lease_ledger as l
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
UUID='f226169e-a888-4cd2-b2ad-412d3212062f'
class Ledger(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.p=Path(self.temp.name);self.ledger=self.p/'ledger';self.lock=self.p/'lock';self.lock.touch()
  self.prefix=b'# Preserve heading\r\n| Resource | Owner | Task | At | Release |\r\n';self.row=b'| RTX 5090 | None | previous terminal | previous UTC | root admission |\r\n';self.suffix=b'| Notebook | root | unrelated | t | r |\r\n\r\nKeep trailing content\r\n'
  self.ledger.write_bytes(self.prefix+self.row+self.suffix);self.ledger.chmod(0o640);self.lease=l.LedgerLease(self.ledger,UUID,d.identity(os.getpid()))
 def tearDown(self):self.temp.cleanup()
 def test_acquire_monitor_release_preserves_other_bytes_and_mode(self):
  with d.exclusive_lock(self.lock):
   row=self.lease.acquire();self.assertEqual(row['status'],'acquired');self.lease.monitor()
   self.assertIn(UUID.encode(),self.ledger.read_bytes());self.assertIn(b'| daily-lan |',self.ledger.read_bytes())
   self.assertTrue(self.ledger.read_bytes().startswith(self.prefix));self.assertTrue(self.ledger.read_bytes().endswith(self.suffix));self.assertEqual(stat.S_IMODE(self.ledger.stat().st_mode),0o640)
   release=self.lease.release(True);self.assertEqual(release['status'],'released');d.ledger_free(self.ledger.read_text())
   self.assertTrue(self.ledger.read_bytes().startswith(self.prefix));self.assertTrue(self.ledger.read_bytes().endswith(self.suffix));self.assertEqual(list(self.p.glob('.ledger-*')),[])
 def test_competing_row_rejects_without_mutation(self):
  self.ledger.write_bytes(self.prefix+self.row.replace(b'| None |',b'| another-owner |')+self.suffix);before=self.ledger.read_bytes()
  with d.exclusive_lock(self.lock):
   with self.assertRaisesRegex(RuntimeError,'busy'):self.lease.acquire()
   self.assertEqual(self.lease.release(True)['status'],'not_acquired')
  self.assertEqual(self.ledger.read_bytes(),before)
 def test_changed_row_fails_monitor_and_is_never_clobbered(self):
  with d.exclusive_lock(self.lock):
   self.lease.acquire();changed=self.ledger.read_bytes().replace(b'| daily-lan |',b'| replacement-owner |');self.ledger.write_bytes(changed)
   with self.assertRaisesRegex(RuntimeError,'row_changed'):self.lease.monitor()
   r=self.lease.release(True);self.assertEqual(r['status'],'retained_ledger_mismatch');self.assertFalse(r['ledgerChanged']);self.assertEqual(self.ledger.read_bytes(),changed)
 def test_other_row_edit_preserved_at_release(self):
  with d.exclusive_lock(self.lock):
   self.lease.acquire();self.ledger.write_bytes(self.ledger.read_bytes().replace(b'unrelated',b'new-other-resource'));self.lease.monitor();self.assertEqual(self.lease.release(True)['status'],'released');self.assertIn(b'new-other-resource',self.ledger.read_bytes())
 def test_unreleased_children_retain_owned_row(self):
  with d.exclusive_lock(self.lock):
   self.lease.acquire();before=self.ledger.read_bytes();self.assertEqual(self.lease.release(False)['status'],'retained_owned_children_unreleased');self.assertEqual(self.ledger.read_bytes(),before)
 def test_pre_replace_conflict_keeps_other_writer_bytes(self):
  original=l.snapshot;calls=0;replacement=self.prefix+self.row+self.suffix+b'new external content\r\n'
  def conflict(path):
   nonlocal calls
   calls+=1
   if calls==2:self.ledger.write_bytes(replacement)
   return original(path)
  with patch.object(l,'snapshot',conflict):
   with self.assertRaisesRegex(RuntimeError,'changed_before_replace'):l.replace_exact(self.ledger,self.row,self.row.replace(b'None',b'daily-lan'))
  self.assertEqual(self.ledger.read_bytes(),replacement);self.assertEqual(list(self.p.glob('.ledger-*')),[])
 def test_missing_or_duplicate_row_rejected(self):
  for data in [self.prefix+self.suffix,self.prefix+self.row+self.row+self.suffix]:
   self.ledger.write_bytes(data)
   with self.assertRaises(RuntimeError):self.lease.acquire()
   self.assertEqual(self.ledger.read_bytes(),data)
 def test_owned_guard_uses_row_during_readiness(self):
  with d.exclusive_lock(self.lock) as fd:
   self.lease.acquire();owner=d.Owned(self.p,fd);owner.guard=self.lease.monitor;owner.check();self.ledger.write_bytes(self.prefix+self.row+self.suffix)
   with self.assertRaisesRegex(RuntimeError,'row_changed'):owner.check()
class Ports(unittest.TestCase):
 def test_existing_listener_rejected(self):
  with socket.socket() as listener:
   listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
   with self.assertRaises(OSError):d.free_port(port)
 def test_closed_listener_time_wait_is_reusable(self):
  with socket.socket() as listener:
   listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',0));listener.listen();port=listener.getsockname()[1]
   client=socket.create_connection(('127.0.0.1',port));server,_=listener.accept();server.close();self.assertEqual(client.recv(1),b'');client.close()
  d.free_port(port)
if __name__=='__main__':unittest.main(verbosity=2)
