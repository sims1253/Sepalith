import json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import live_observer as o

class Client:
 def get_job(self,**_):return SimpleNamespace(id=o.JOB_ID,name=o.JOB_NAME,state=SimpleNamespace(cluster_id='ses_abc',current_state='RUNNING'),status_updated_at='now')
 def get_job_runs(self,*_,**__):return [SimpleNamespace(id='job_abc',cluster_id='ses_abc',status='RUNNING')]
 def logs_for_job_run(self,*_,**__):return 'prefix '+json.dumps({'schema':o.EVENT_SCHEMA,'at':'now','phase':'training','status':'START','argv':['secret']})+'\nraw secret\n'

class Tests(unittest.TestCase):
 def test_provider_allowlists_phase_event(self):
  x=o.provider(Client());self.assertEqual(x['cluster_id'],'ses_abc');self.assertNotIn('argv',json.dumps(x));self.assertEqual(x['phase_event_count'],1)
 def test_remote_program_allowlists_optimizer_fields(self):
  self.assertIn("('loss','learning_rate','grad_norm'",o.REMOTE);self.assertIn("'first_optimizer_record'",o.REMOTE);self.assertNotIn("identity=",o.REMOTE)
 def test_remote_command_binds_exact_run(self):
  x=o.remote_command();self.assertIn(o.RUN_ID,x);self.assertNotIn('/bin/sh',x)
  source=Path('live_observer.py').read_text();self.assertIn("ubuntu@'+host",source);self.assertIn("'ray@0.0.0.0'",source);self.assertIn("'5020'",source)
 def test_ssh_failure_is_categorized_without_raw_message(self):
  self.assertEqual(o.ssh_failure('ssh: connect to host 203.0.113.1 port 22: Connection timed out'),'connection_timeout')
 def test_actual_output_has_no_credential_material(self):
  path=Path('live-observation.json')
  if path.exists():
   text=path.read_text();self.assertNotIn('PRIVATE KEY',text);self.assertNotIn('Bearer ',text);self.assertNotRegex(text,r'hf_[A-Za-z0-9]+')
 def test_observe_does_not_infer_step_from_running(self):
  with patch.object(o,'private_artifacts',return_value={'private':True}),patch.object(o,'ssh_read',return_value={'telemetry':{'optimizer_steps':0}}):
   x=o.observe(client=Client());self.assertFalse(x['first_optimizer_update_observed']);self.assertIn('does not prove',x['interpretation'])
 def test_observe_requires_actual_optimizer_record(self):
  with patch.object(o,'private_artifacts',return_value={'private':True}),patch.object(o,'ssh_read',return_value={'telemetry':{'optimizer_steps':1}}):
   self.assertTrue(o.observe(client=Client())['first_optimizer_update_observed'])

if __name__=='__main__':unittest.main(verbosity=2)
