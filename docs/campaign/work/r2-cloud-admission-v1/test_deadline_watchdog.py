import sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import deadline_watchdog as w
NAME='sepalith-r2-control-'+'a'*32
class Fake:
 def __init__(self,states):self.now=100.;self.states=list(states);self.calls=[];self.events=[]
 def clock(self):return self.now
 def sleep(self,x):self.now+=x
 def call(self,op,name):
  self.calls.append(op)
  if op=='terminate':return {'ok':True}
  s=self.states.pop(0) if len(self.states)>1 else self.states[0]
  return {'ok':False} if s=='ERROR' else {'ok':True,'name':NAME,'id':'prodjob_test123','state':s}
 def run(self,deadline=140):return w.monitor(NAME,deadline,self.events.append,clock=self.clock,sleep=self.sleep,call=self.call)
class Tests(unittest.TestCase):
 def test_success_does_not_terminate(self):
  f=Fake(['RUNNING','SUCCEEDED']);self.assertEqual(f.run()['state'],'SUCCEEDED');self.assertNotIn('terminate',f.calls)
 def test_absolute_deadline_terminates_and_confirms(self):
  f=Fake(['RUNNING']*4+['TERMINATED']);r=f.run();self.assertTrue(r['termination_requested']);self.assertEqual(r['reason'],'absolute_deadline');self.assertEqual(f.calls.count('terminate'),1)
 def test_status_errors_cannot_wait_forever(self):
  f=Fake(['ERROR']);r=f.run(1000);self.assertEqual(r['reason'],'status_api_failed');self.assertLess(f.now,400)
 def test_no_terminal_proof_reports_unconfirmed(self):
  f=Fake(['RUNNING']);r=f.run();self.assertEqual(r['status'],'termination_unconfirmed')
 def test_terminal_failed_is_not_training_success(self):
  f=Fake(['FAILED']);self.assertEqual(f.run()['state'],'FAILED')
 def test_old_or_excessive_deadline_rejected(self):
  now=1789327000
  for deadline in ['2026-09-12T00:00:00Z','2026-09-14T12:15:00Z','2026-09-13T21:00:00']:
   with self.assertRaises(ValueError):w.validate(NAME,deadline,now)
 def test_unique_name_required(self):
  with self.assertRaises(ValueError):w.validate('sepalith-old-job','2026-09-13T21:00:00Z',1789327000)
 def test_valid_six_hour_bounded_deadline(self):
  self.assertGreater(w.validate(NAME,'2026-09-13T21:00:00Z',1789327000),1789327000)
if __name__=='__main__':unittest.main(verbosity=2)
