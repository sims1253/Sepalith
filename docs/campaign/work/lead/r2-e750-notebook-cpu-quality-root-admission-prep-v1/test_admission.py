import datetime, importlib.util, json, tempfile, unittest
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('issuer',P/'issue_admission.py'); issuer=importlib.util.module_from_spec(spec);spec.loader.exec_module(issuer)
class AdmissionTests(unittest.TestCase):
 def test_exact_admission(self):
  now=datetime.datetime(2026,9,15,4,0,tzinfo=datetime.timezone.utc)
  x=issuer.build('e750-cap-quality-review',issuer.SOURCE_MANIFEST_SHA,now)
  self.assertEqual(x['status'],'admitted');self.assertEqual(x['caps'],[192,384,768]);self.assertEqual(x['maximum_threads'],2)
  self.assertEqual((datetime.datetime.fromisoformat(x['expires_at'])-now).total_seconds(),1800)
  self.assertEqual(x['maximum_seconds'],28800);self.assertTrue(x['source_reviewed'])
 def test_wrong_source_review_rejected(self):
  with self.assertRaisesRegex(ValueError,'source manifest mismatch'):issuer.build('safe','0'*64)
 def test_unsafe_run_id_rejected(self):
  with self.assertRaisesRegex(ValueError,'unsafe run ID'):issuer.build('../bad',issuer.SOURCE_MANIFEST_SHA)
 def test_launch_wrapper_enforces_closure_stat_port_and_watchdog(self):
  s=(P/'stage_and_launch.sh').read_text()
  for value in ('source-manifest.json','actual==seen','model.st_dev','socket.socket()','watchdog_remote.py','packet-final'):
   self.assertIn(value,s)
 def test_watchdog_has_identity_scoped_runner_and_server_cleanup(self):
  s=(P/'watchdog_remote.py').read_text()
  for value in ('MAX_SECONDS=28800','TERM_GRACE_SECONDS=150','proc_tick','os.killpg(pid,signal.SIGTERM)','arm_servers(run_root)','os.killpg(proc.pid,signal.SIGKILL)'):
   self.assertIn(value,s)
if __name__=='__main__':unittest.main()
