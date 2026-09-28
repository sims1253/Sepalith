import importlib.util,json,types,unittest
from pathlib import Path
from unittest.mock import patch
import deadline_watchdog as w
class WatchdogRecoveryTests(unittest.TestCase):
 def test_id_selector_never_has_cloud_for_status_or_terminate(self):
  for op in ('status','terminate'):
   with patch.object(w.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='{}')) as run:
    self.assertTrue(w.cli(op,'prodjob_synthetic')['ok'])
   argv=run.call_args.args[0];self.assertEqual(argv[:5],[w.CLI,'job',op,'--id','prodjob_synthetic']);self.assertNotIn('--cloud',argv)
 def test_name_selector_retains_explicit_cloud(self):
  with patch.object(w.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='{}')) as run:w.cli('terminate','sepalith-r2-control-'+'a'*32)
  self.assertEqual(run.call_args.args[0][-2:],['--cloud','Anyscale Cloud']);self.assertIn('--name',run.call_args.args[0])
 def test_actual_installed_resolver_rejects_old_combination_before_client(self):
  spec=importlib.util.spec_from_file_location('installed_resolver',Path(__file__).with_name('installed-job-resolver.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  obj=module.InstalledPrivateJobSDK();calls=[]
  model=types.SimpleNamespace(id='prodjob_synthetic',name='synthetic')
  obj.client=types.SimpleNamespace(get_job=lambda **kw:(calls.append(('get',kw)) or model),terminate_job=lambda job_id:calls.append(('terminate',job_id)),get_job_runs=lambda job_id:[])
  obj.logger=types.SimpleNamespace(info=lambda *_:None);obj._job_model_to_status=lambda **kw:'synthetic-status'
  for method in (obj.status,obj.terminate):
   with self.assertRaisesRegex(ValueError,'only be used with'):method(job_id=model.id,cloud='Anyscale Cloud')
  self.assertEqual(calls,[])
  self.assertEqual(obj.status(job_id=model.id),'synthetic-status');self.assertEqual(obj.terminate(job_id=model.id),model.id)
  self.assertEqual(calls[-1],('terminate',model.id));self.assertTrue(all(row[1]['cloud'] is None for row in calls if row[0]=='get'))
  self.assertEqual(obj.terminate(name='synthetic',cloud='Anyscale Cloud'),model.id)

class RootArmingRecoveryTests(unittest.TestCase):
 def test_root_arming_executes_exact_local_corrected_watchdog(self):
  import root_arm_watchdog as arm,cloud_entry as c,tempfile,sys
  from test_cloud_entry import binding
  self.assertEqual(arm.WATCHDOG.resolve(),Path(w.__file__).resolve());self.assertEqual(c.sha(arm.WATCHDOG),arm.EXPECTED)
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);p=root/'binding.json';p.write_text(json.dumps(binding()))
   with patch.object(sys,'argv',['root_arm_watchdog.py',str(p),'--output',str(root/'guard')]),patch.object(arm.os,'execv') as execute:
    arm.main()
   argv=execute.call_args.args[1];self.assertEqual(Path(argv[2]).resolve(),arm.WATCHDOG.resolve());self.assertIn('sepalith-r2-control-'+'a'*32,argv)
   b=binding();b['absolute_deadline_utc']='2026-09-13T22:00:01Z';p.write_text(json.dumps(b))
   with patch.object(sys,'argv',['root_arm_watchdog.py',str(p),'--output',str(root/'guard')]),patch.object(arm.os,'execv') as execute:
    with self.assertRaisesRegex(ValueError,'watchdog limit expanded'):arm.main()
   execute.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
