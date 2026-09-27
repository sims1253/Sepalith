"""Synthetic provider replies and time only; no API calls or real sleeping."""
import importlib.util,sys,unittest
from pathlib import Path
H=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(H))
import deadline_watchdog_delayed_submission as w
NAME='sepalith-r2-control-'+'a'*32
spec=importlib.util.spec_from_file_location('old_tests',H.parent/'r2-cloud-admission-v1/test_deadline_watchdog.py');old=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(H.parent/'r2-cloud-admission-v1'));spec.loader.exec_module(old)
# Reuse all unaffected original tests against the candidate. The one old test
# that requires quitting before any ID exists is exactly the behavior changed.
class InheritedTests(old.Tests):
 def setUp(self):old.w=w
 def test_status_errors_cannot_wait_forever(self):
  f=old.Fake(['RUNNING']+['ERROR']*100);r=f.run(1000)
  self.assertEqual(r['reason'],'status_api_failed');self.assertLess(f.now,400)
class DelayedTests(unittest.TestCase):
 def test_delayed_submission_240s_still_observed_and_terminated_at_deadline(self):
  now=[100.];calls=[];events=[];terminated=[False]
  def call(op,name):
   calls.append((now[0],op,name))
   if op=='terminate':terminated[0]=True;return {'ok':True}
   if now[0]<340:return {'ok':False,'exit_code':1}
   return {'ok':True,'name':NAME,'id':'prodjob_delayed123','state':'TERMINATED' if terminated[0] else 'RUNNING'}
  r=w.monitor(NAME,500,events.append,clock=lambda:now[0],sleep=lambda x:now.__setitem__(0,now[0]+x),call=call)
  self.assertEqual(r['status'],'terminal_observed');self.assertEqual(r['reason'],'absolute_deadline');self.assertEqual(r['job_id'],'prodjob_delayed123')
  terms=[x for x in calls if x[1]=='terminate'];self.assertEqual(terms,[(500.,'terminate','prodjob_delayed123')]);self.assertTrue(any(e['event']=='pending_submission' and e['utc_epoch']>=280 for e in events))
 def test_no_job_until_deadline_still_requests_unique_name_termination(self):
  now=[100.];calls=[];events=[]
  def call(op,name):calls.append((now[0],op,name));return {'ok':False,'exit_code':1}
  r=w.monitor(NAME,400,events.append,clock=lambda:now[0],sleep=lambda x:now.__setitem__(0,now[0]+x),call=call)
  self.assertEqual(r['status'],'termination_unconfirmed');self.assertEqual(r['reason'],'absolute_deadline');self.assertEqual(calls[-1],(400.,'terminate',NAME));self.assertIsNone(r['job_id'])
 def test_no_id_name_termination_success_has_bounded_confirmation(self):
  now=[100.];calls=[]
  def call(op,name):calls.append((now[0],op,name));return {'ok':True} if op=='terminate' else {'ok':False}
  r=w.monitor(NAME,400,lambda x:None,clock=lambda:now[0],sleep=lambda x:now.__setitem__(0,now[0]+x),call=call)
  self.assertEqual(r['status'],'termination_unconfirmed');self.assertTrue(r['termination_requested']);self.assertEqual(now[0],520)
if __name__=='__main__':unittest.main(verbosity=2)
