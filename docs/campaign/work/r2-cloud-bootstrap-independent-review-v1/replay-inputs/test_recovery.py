"""CPU-only regression through the real cloud entrypoint failure path."""
import contextlib,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import cloud_entry as c
from test_cloud_entry import binding
class RecoveryTests(unittest.TestCase):
 def test_pre_venv_missing_compiler_is_visible(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);run=root/'run';run.mkdir();p=root/'binding.json';p.write_text(json.dumps(binding()));stdout=io.StringIO()
   with patch.object(sys,'argv',['cloud_entry.py',str(p)]),patch.object(c.tempfile,'mkdtemp',return_value=str(run)),patch.object(c,'validate_binding',return_value=time.time()+6000),patch.object(c.shutil,'which',return_value=None),patch.object(c.subprocess,'Popen') as popen,contextlib.redirect_stdout(stdout):
    with self.assertRaises(SystemExit):c.main()
   popen.assert_not_called()
   events=[json.loads(line) for line in stdout.getvalue().splitlines()]
   self.assertTrue(any(row.get('event')=='entry_failure' and row.get('phase')=='image-preflight' and row.get('reason')=='image compiler or GNU timeout missing' for row in events),stdout.getvalue())
   self.assertTrue(any(row.get('event')=='entry_terminal' and row.get('upload_success') is False for row in events),stdout.getvalue())

class ObservationTests(unittest.TestCase):
 def test_bootstrap_subprocess_failure_has_bounded_safe_tail(self):
  from entry_observer import setup_tail
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   log=Path(tmp)/'uv-bootstrap.log';stdout=io.StringIO()
   child="print(\"/private/path/python: No module named pip\"); print('Authorization: Bearer synthetic-private-marker'); raise SystemExit(7)"
   with contextlib.redirect_stdout(stdout):
    with self.assertRaises(ValueError):c.guarded_run([sys.executable,'-c',child],env={'PATH':os.environ['PATH']},cwd=tmp,deadline=time.time()+5,log=log,phase='uv-bootstrap',mem_probe=lambda:9*1024**3)
   events=[json.loads(s) for s in stdout.getvalue().splitlines()];last=events[-1]
   self.assertEqual(last['event'],'phase_terminal');self.assertEqual(last['exit_code'],7)
   self.assertIn('No module named pip',last['setup_log_tail']);self.assertNotIn('synthetic-private-marker',stdout.getvalue());self.assertNotIn('/private/path',stdout.getvalue())
   self.assertEqual(setup_tail(log,'sentinel'),[]);self.assertEqual(setup_tail(log,'artifact-upload'),[])
 def test_training_heartbeat_observes_real_append_without_optimizer(self):
  import hashlib
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);telemetry=root/'telemetry.jsonl';marker=root/'optimizer.bin';marker.write_bytes(b'synthetic unchanged optimizer sentinel');before=hashlib.sha256(marker.read_bytes()).hexdigest();stdout=io.StringIO()
   rows=[{'event':'optimizer_step','step':1,'seconds':.25,'remaining_seconds':20,'resources':{'process_id':123,'process_peak_rss_bytes':42,'unknown':'synthetic-private-marker'}},{'event':'trainer_metrics','step':1,'metrics':{'loss':1.25,'learning_rate':.0001,'private':'synthetic-private-marker'}}]
   child='import pathlib,time; p=pathlib.Path('+repr(str(telemetry))+');p.write_text('+repr(''.join(json.dumps(r)+'\n' for r in rows))+');time.sleep(.8)'
   with contextlib.redirect_stdout(stdout):
    c.guarded_run([sys.executable,'-c',child],env={'PATH':os.environ['PATH']},cwd=tmp,deadline=time.time()+5,log=root/'training.log',phase='training',training_telemetry=telemetry,heartbeat_interval=.05,mem_probe=lambda:9*1024**3)
   events=[json.loads(s) for s in stdout.getvalue().splitlines()];beats=[r for r in events if r['event']=='training_heartbeat']
   self.assertGreaterEqual(len(beats),2);self.assertTrue(any(r.get('last_completed_step')==1 and r.get('metrics',{}).get('loss')==1.25 for r in beats))
   self.assertTrue(all(r['linux_mem_available_bytes']==9*1024**3 for r in beats));self.assertNotIn('synthetic-private-marker',stdout.getvalue())
   self.assertEqual(before,hashlib.sha256(marker.read_bytes()).hexdigest());self.assertEqual(events[-1]['status'],'succeeded')
 def test_bounded_tail_rejects_incomplete_nonfinite_and_private_rows(self):
  from entry_observer import setup_tail,telemetry_snapshot
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);log=root/'setup.log';log.write_text('synthetic-private-marker\n'*10000)
   self.assertEqual(len(setup_tail(log,'packages-bootstrap')),12);self.assertNotIn('synthetic-private-marker',str(setup_tail(log,'packages-bootstrap')))
   t=root/'telemetry.jsonl';t.write_text(json.dumps({'event':'trainer_metrics','step':3,'metrics':{'loss':float('nan'),'private':'synthetic-private-marker'}})+'\n'+json.dumps({'event':'optimizer_step','step':4}))
   snap=telemetry_snapshot(t);self.assertNotIn('last_completed_step',snap);self.assertNotIn('metrics',snap);self.assertNotIn('synthetic-private-marker',str(snap))
 def test_upload_failure_does_not_echo_private_log(self):
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);stdout=io.StringIO()
   with contextlib.redirect_stdout(stdout):
    with self.assertRaises(ValueError):c.guarded_run([sys.executable,'-c',"print('synthetic-private-marker');raise SystemExit(8)"],env={'PATH':os.environ['PATH']},cwd=tmp,deadline=time.time()+5,log=root/'upload.log',phase='artifact-upload',mem_probe=lambda:9*1024**3)
   self.assertNotIn('synthetic-private-marker',stdout.getvalue());self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])['setup_log_tail'],[])

class ShellTests(unittest.TestCase):
 def test_missing_base_python_reports_failure_before_exec(self):
  import shutil,subprocess
  with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
   root=Path(tmp);bin_dir=root/'bin';bin_dir.mkdir();(bin_dir/'dirname').symlink_to(shutil.which('dirname'))
   script=root/'root-bound-entry.sh';script.write_bytes((c.HERE/'root-bound-entry.sh').read_bytes())
   run=subprocess.run(['/bin/bash',str(script)],env={'PATH':str(bin_dir)},capture_output=True,text=True,timeout=3)
   self.assertEqual(run.returncode,1);events=[json.loads(row) for row in run.stdout.splitlines()]
   self.assertEqual(events[-2]['reason'],'python3 unavailable');self.assertEqual(events[-1]['event'],'entry_terminal');self.assertFalse(events[-1]['upload_success'])
if __name__=='__main__':unittest.main(verbosity=2)
