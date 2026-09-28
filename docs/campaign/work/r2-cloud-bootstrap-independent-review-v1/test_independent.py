"""Independent synthetic failure/argv/deadline tests; no SDK or model imports."""
import contextlib, importlib.util, io, json, os, pathlib, sys, tempfile, types, unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
H=pathlib.Path(__file__).resolve().parent;W=H.parent;F=H/'replay-inputs';sys.path.insert(0,str(F))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import cloud_entry as c, deadline_watchdog as v2, root_arm_watchdog as arm
from test_cloud_entry import binding

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v1=load('recovery_v1_watchdog',W/'r2-cloud-bootstrap-recovery-v1/deadline_watchdog.py')
resolver=load('source_only_resolver',F/'installed-job-resolver.py')
NAME='sepalith-r2-control-'+'a'*32

def scenario(guard,arrives):
 now=[0.];calls=[];events=[];terminated=[False]
 def call(op,selector):
  calls.append({'time':now[0],'operation':op,'selector':selector})
  if op=='terminate':terminated[0]=True;return {'ok':True}
  if arrives is None or now[0]<arrives:return {'ok':False,'exit_code':1}
  return {'ok':True,'name':NAME,'id':'prodjob_delayedindependent','state':'TERMINATED' if terminated[0] else 'RUNNING'}
 result=guard.monitor(NAME,600,events.append,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+s),call=call)
 return result,calls,events,now[0]

class Independent(unittest.TestCase):
 def test_v1_red_and_v2_green_delayed_300_second_submission(self):
  old,oldcalls,_,_=scenario(v1,300)
  new,calls,events,_=scenario(v2,300)
  self.assertEqual(old['reason'],'status_api_failed')
  self.assertEqual(next(r['time'] for r in oldcalls if r['operation']=='terminate'),140.)
  self.assertEqual(new['reason'],'absolute_deadline');self.assertEqual(new['status'],'terminal_observed')
  self.assertEqual([r for r in calls if r['operation']=='terminate'],[{'time':600.,'operation':'terminate','selector':'prodjob_delayedindependent'}])
  self.assertTrue(any(r['event']=='pending_submission' and r['utc_epoch']>180 for r in events))
  (H/'delayed-submission-evidence.json').write_text(json.dumps({'v1':old,'v1_termination_at':140,'v2':new,'v2_termination_at':600,'job_arrives_at':300},indent=2)+'\n')
 def test_v2_no_job_keeps_guard_to_deadline_and_bounded_confirmation(self):
  result,calls,events,end=scenario(v2,None)
  self.assertEqual(result['reason'],'absolute_deadline');self.assertEqual(result['status'],'termination_unconfirmed');self.assertEqual(end,720)
  self.assertEqual(next(r for r in calls if r['operation']=='terminate'),{'time':600.,'operation':'terminate','selector':NAME})
 def test_real_pre_venv_failure_uses_valid_binding_and_has_terminal_evidence(self):
  with tempfile.TemporaryDirectory(dir=H) as temp:
   root=pathlib.Path(temp);p=root/'binding.json';b=binding();p.write_text(json.dumps(b));run=root/'run';run.mkdir();out=io.StringIO()
   # Actual validate_binding runs with a synthetic clock; no model or subprocess is entered.
   with patch.object(sys,'argv',['cloud_entry.py',str(p)]),patch.object(c.time,'time',return_value=c.utc('2026-09-13T20:30:00Z')),patch.object(c.tempfile,'mkdtemp',return_value=str(run)),patch.object(c.shutil,'which',return_value=None),patch.object(c.subprocess,'Popen') as launch,contextlib.redirect_stdout(out):
    with self.assertRaises(SystemExit) as caught:c.main()
   self.assertEqual(caught.exception.code,1);launch.assert_not_called();rows=[json.loads(s) for s in out.getvalue().splitlines()]
   self.assertTrue(any(r.get('event')=='entry_failure' and r.get('reason')=='image compiler or GNU timeout missing' for r in rows))
   self.assertEqual(rows[-1]['event'],'entry_terminal');self.assertFalse(rows[-1]['upload_success']);self.assertFalse(rows[-1]['training_success'])
 def test_actual_cli_argv_resolves_with_installed_methods_for_both_selectors(self):
  obj=resolver.InstalledPrivateJobSDK();calls=[];model=types.SimpleNamespace(id='prodjob_synthetic',name=NAME)
  obj.client=types.SimpleNamespace(get_job=lambda **kw:(calls.append(kw) or model),get_job_runs=lambda *_:[],terminate_job=lambda *args,**kw:None)
  obj.logger=types.SimpleNamespace(info=lambda *_:None);obj._job_model_to_status=lambda **kw:{'id':model.id,'name':model.name,'state':'RUNNING'}
  def invoke(argv,**kw):
   op=argv[2];selector=argv[3];value=argv[4];params={'job_id' if selector=='--id' else 'name':value}
   if '--cloud' in argv:params['cloud']=argv[argv.index('--cloud')+1]
   result=getattr(obj,op)(**params)
   return types.SimpleNamespace(returncode=0,stdout=json.dumps(result))
  with patch.object(v2.subprocess,'run',side_effect=invoke):
   for op in ['status','terminate']:
    for selector in ['prodjob_synthetic',NAME]:self.assertTrue(v2.cli(op,selector)['ok'])
  self.assertEqual(len(calls),4)
  self.assertTrue(all(x['cloud'] is None for x in calls if x['job_id']))
  self.assertTrue(all(x['cloud']=='Anyscale Cloud' for x in calls if x['name']))
 def test_root_recovery_binding_preserves_original_absolute_deadline(self):
  b=binding();b['watchdog_armed_at_utc']='2026-09-13T20:55:00Z';b['absolute_deadline_utc']='2026-09-13T22:25:52Z'
  now=c.utc('2026-09-13T20:57:00Z');hard=c.validate_binding(b,now)
  self.assertEqual(hard,c.utc('2026-09-13T22:25:52Z'))
  self.assertEqual(c.remaining_phase(hard,4200,660,now),min(now+4200,hard-660))
  with self.assertRaises(ValueError):c.validate_binding(b,hard)
 def test_root_arming_enforces_7200_before_exec(self):
  b=binding();b['absolute_deadline_utc']='2026-09-13T22:00:01Z'
  with tempfile.TemporaryDirectory(dir=H) as tmp:
   p=pathlib.Path(tmp)/'binding.json';p.write_text(json.dumps(b))
   with patch.object(sys,'argv',['root_arm_watchdog.py',str(p),'--output',str(pathlib.Path(tmp)/'guard')]),patch.object(arm.os,'execv') as execute,patch.object(c.time,'time',return_value=c.utc('2026-09-13T20:30:00Z')):
    with self.assertRaisesRegex(ValueError,'watchdog limit expanded'):arm.main()
   execute.assert_not_called()
 def test_untrusted_error_strings_and_upload_tails_are_never_echoed(self):
  from entry_observer import safe_reason,setup_tail
  marker='synthetic_sensitive_marker'
  self.assertEqual(safe_reason(ValueError(marker)),'ValueError')
  with tempfile.TemporaryDirectory(dir=H) as tmp:
   p=pathlib.Path(tmp)/'log';p.write_text('https://synthetic.invalid/'+marker+'\nAuthorization: Bearer '+marker+'\n')
   self.assertNotIn(marker,json.dumps(setup_tail(p,'packages-bootstrap')));self.assertEqual(setup_tail(p,'artifact-upload'),[])
if __name__=='__main__':unittest.main(verbosity=2)
