import contextlib,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'payload'))
import bootstrap_diagnostic as diagnostic
import diagnostic_contract as contract

def binding(now=1_800_000_000):
 iso=lambda value:__import__('datetime').datetime.fromtimestamp(value,__import__('datetime').timezone.utc).isoformat()
 return {'schema':'sepalith.cloud-cpt.bootstrap-diagnostic.v2','admitted':True,'run_id':'a'*32,'image_uri':contract.IMAGE,'instance_type':'g5.2xlarge','artifact_repo':contract.REPO,'artifact_prefix':'r2-cpt-bootstrap-diagnostic/'+'a'*32,'provider_timeout_seconds':600,'watchdog_timeout_seconds':900,'max_retries':0,'training_authorized':False,'input_staging_authorized':False,'payload_manifest_sha256':'b'*64,'root_admission_sha256':'c'*64,'root_admission_relative_path':'root-diagnostic-admission.json','watchdog_armed_receipt_sha256':'d'*64,'watchdog_armed_receipt_relative_path':'watchdog-armed.json','watchdog_armed_at_utc':iso(now-1),'absolute_deadline_utc':iso(now+899)}

class DiagnosticTests(unittest.TestCase):
 def test_exact_bounded_no_training_contract(self):self.assertGreater(contract.validate(binding(),now=1_800_000_000),1_800_000_000)
 def test_training_or_long_timeout_rejected(self):
  b=binding();b['training_authorized']=True
  with self.assertRaisesRegex(ValueError,'scope expanded'):contract.validate(b,now=1_800_000_000)
  b=binding();b['provider_timeout_seconds']=601
  with self.assertRaisesRegex(ValueError,'lifecycle bound'):contract.validate(b,now=1_800_000_000)
 def test_bootstrap_failure_emits_bounded_sanitized_tail(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);diagnostic.PHASE_CAPS={'injected':5};output=io.StringIO()
   command=[sys.executable,'-c',"import sys;print('hf_'+'x'*64);print('z'*6000);sys.exit(7)"]
   with contextlib.redirect_stdout(output):
    with self.assertRaisesRegex(ValueError,'injected failed'):diagnostic.phase('injected',command,time.time()+5,run,diagnostic.env(run))
   text=output.getvalue();self.assertIn('EXCEPTION',text);self.assertIn('exit_code',text);self.assertNotIn('hf_'+'x'*64,text);self.assertLess(len(json.loads(text.splitlines()[-1])['log_tail'].encode()),4200)
 def test_credential_removed_from_bootstrap_env(self):
  injected={'HF_TOKEN':'hidden','OTHER_SECRET':'hidden','AWS_ACCESS_KEY_ID':'hidden','API_KEY':'hidden','BASIC_AUTH':'hidden','SAFE_FLAG':'retained'}
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,injected,clear=False):
   clean=diagnostic.env(Path(d))
  for key in injected:
   self.assertEqual(clean.get(key),injected[key] if key=='SAFE_FLAG' else None)
 def test_admission_and_storage_failure_causes_are_bounded_and_sanitized(self):
  causes=('diagnostic admission does not bind payload','local storage below 24GiB')
  for cause in causes:
   with self.subTest(cause=cause):self.assertEqual(diagnostic.safe_message(ValueError(cause)),cause)
  hidden='hf_'+'s'*40
  message=diagnostic.safe_message(ValueError('local storage failure bearer '+hidden))
  self.assertIn('local storage failure',message);self.assertNotIn(hidden,message);self.assertLessEqual(len(message.encode()),1024)
 def test_small_remote_terminal_commit_is_sanitized_record_only(self):
  captured={}
  class Response:
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self):return json.dumps({'commitOid':'e'*40}).encode()
  def open_(request,timeout):captured['request']=request;return Response()
  with patch.dict(os.environ,{'HF_TOKEN':'hf_'+'q'*40}),patch.object(diagnostic.urllib.request,'urlopen',open_):
   revision=diagnostic.persist(binding(),{'status':'FAIL','error_type':'ValueError'})
  self.assertEqual(revision,'e'*40);self.assertNotIn(('hf_'+'q'*40).encode(),captured['request'].data);self.assertIn(b'diagnostic-terminal.json',captured['request'].data)
 def test_commands_are_bootstrap_only(self):
  names=[row[0] for row in diagnostic.PHASE_COMMANDS];self.assertEqual(names,['pip_uv','uv_version','python_install','python_discovery','python_probe','venv','requirements'])
  self.assertNotIn('campaign_cpt',json.dumps(diagnostic.PHASE_COMMANDS));self.assertNotIn('stage_inputs',json.dumps(diagnostic.PHASE_COMMANDS))

if __name__=='__main__':unittest.main()
