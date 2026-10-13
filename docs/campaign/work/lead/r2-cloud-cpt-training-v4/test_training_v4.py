"""CPU-only lifecycle and observability tests for full CPT training v3."""
import contextlib,datetime,importlib.util,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'payload'))
import cloud_entry as entry
V1=H.parent/'r2-cloud-cpt-admission-v1/payload'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def binding(now=None):
 now=time.time() if now is None else now;iso=lambda x:datetime.datetime.fromtimestamp(x,datetime.timezone.utc).isoformat()
 return {'schema':'sepalith.cloud-cpt.binding.v1','admitted':True,'run_id':'a'*32,'image_uri':'docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9','instance_type':'g5.2xlarge','input_repo':'scholzmx/sepalith-lora','artifact_repo':'scholzmx/sepalith-lora','artifact_prefix':'r2-cpt/'+'a'*32,'input_manifest_sha256':'a9afe5d58034b56e36abf83a0c1d169ec578b79cc215daa32603ccd210e7045b','recipe_sha256':'254de1150b0fcc8e8566178905542bdadc1255610a2dc3cb8a9168dc7017d538','input_revision':'1e4df9c9db3cee1a7df5bbcc7caa62920d2ab2db','root_recipe_admission_sha256':'c'*64,'root_recipe_admission_relative_path':'root-recipe-admission.json','watchdog_armed_receipt_sha256':'d'*64,'watchdog_armed_receipt_relative_path':'watchdog-armed.json','payload_manifest_sha256':'e'*64,'watchdog_armed_at_utc':iso(now-1),'absolute_deadline_utc':iso(now+28800),'provider_timeout_seconds':28800,'watchdog_timeout_seconds':29100,'max_incremental_charge_usd':28.0,'phase_seconds':{'bootstrap':1800,'staging':3600,'training':19800,'upload':3000,'cleanup':300},'resume':None}

class FakeSidecar:
 pid=987654
 def wait(self,timeout=None):return 0

class Tests(unittest.TestCase):
 def orchestration(self,fail=None):
  stack=contextlib.ExitStack();tmp=stack.enter_context(tempfile.TemporaryDirectory());root=Path(tmp);bind=root/'binding.json';bind.write_text(json.dumps(binding()))
  run=root/'run';run.mkdir();calls=[];persist=[]
  def mkdtemp(**_):return str(run)
  def guarded(argv,env,cwd,deadline,log,floor=8*1024**3):
   log=Path(log);name=log.stem;calls.append(name);log.parent.mkdir(parents=True,exist_ok=True);log.write_text(name+' ok\n')
   if name==fail:raise ValueError(name+' injected failure')
   if name=='bootstrap':(run/'bootstrap/bin').mkdir(parents=True);(run/'bootstrap/bin/uv').write_text('uv')
   if name=='uv-version':log.write_text('uv 0.11.23\n')
   if name=='python-probe':log.write_text('{}')
   if name=='venv':(run/'venv/bin').mkdir(parents=True);(run/'venv/bin/python').write_text('python')
   if name=='staging':
    recipe=run/'relocated-recipe.json';recipe.write_text('{}');(run/'artifacts/staging-receipt.json').write_text(json.dumps({'relocated_recipe':str(recipe)}))
   if name=='training':
    q=run/'artifacts/training';q.mkdir(parents=True);(q/'terminal.json').write_text(json.dumps({'step':1902,'status':'schedule_complete'}))
   return 0
  def persist_early(b,r):persist.append(r);return 'f'*40
  discover_path=run/'python/cpython-3.10.19/bin/python';discover_path.parent.mkdir(parents=True);discover_path.write_text('python')
  patches=[patch.object(entry,'validate_binding',return_value=time.time()+28800),patch.object(entry,'verify_payload'),patch.object(entry,'verify_root_receipts'),patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=64*1024**3)),patch.object(Path,'is_dir',return_value=True),patch.object(entry.tempfile,'mkdtemp',side_effect=mkdtemp),patch.object(entry,'guarded',side_effect=guarded),patch.object(entry,'discover',return_value=discover_path),patch.object(entry,'validate_python'),patch.object(entry.subprocess,'Popen',return_value=FakeSidecar()),patch.object(entry,'persist_early',side_effect=persist_early),patch.object(sys,'argv',['cloud_entry.py',str(bind)])]
  for p in patches:stack.enter_context(p)
  return stack,calls,persist
 def test_complete_startup_staging_training_and_final_path(self):
  stack,calls,persist=self.orchestration()
  with stack:entry.main()
  self.assertEqual(calls,['bootstrap','uv-version','python','python-probe','venv','packages','sentinel','staging','training','final-upload']);self.assertEqual(persist,[])
 def test_sentinel_and_staging_failures_persist_small_terminal(self):
  for failed in ('sentinel','staging'):
   stack,calls,persist=self.orchestration(failed)
   with stack,self.subTest(failed=failed),self.assertRaises(SystemExit):entry.main()
   self.assertEqual(persist[0]['failure']['phase'],failed);self.assertFalse(persist[0]['training_complete'])
 def test_payload_failure_before_managed_python_persists(self):
  stack,calls,persist=self.orchestration()
  stack.enter_context(patch.object(entry,'verify_payload',side_effect=ValueError('payload injected failure')))
  with stack,self.assertRaises(SystemExit):entry.main()
  self.assertEqual(calls,[]);self.assertEqual(persist[0]['failure']['phase'],'payload')
 def test_guarded_phase_emits_sanitized_failure_tail(self):
  with tempfile.TemporaryDirectory() as d,patch.object(entry,'mem_available',return_value=64*1024**3):
   log=Path(d)/'probe.log';capture=io.StringIO()
   with contextlib.redirect_stdout(capture),self.assertRaises(ValueError):entry.guarded([sys.executable,'-c',"print('Bearer hf_ABCsecret');raise SystemExit(7)"],dict(os.environ),H,time.time()+10,log)
   text=capture.getvalue();self.assertIn('"status": "START"',text);self.assertIn('"status": "EXCEPTION"',text);self.assertIn('[REDACTED]',text);self.assertNotIn('hf_ABCsecret',text)
 def test_clean_bootstrap_environment_removes_credentials(self):
  with patch.dict(os.environ,{'HF_TOKEN':'hf_secret','MY_API_KEY':'secret','SAFE':'yes'},clear=True):clean=entry.clean_env('/tmp/run')
  self.assertNotIn('HF_TOKEN',clean);self.assertNotIn('MY_API_KEY',clean);self.assertEqual(clean['SAFE'],'yes')
 def test_early_terminal_uses_stdlib_private_commit(self):
  class Response:
   def __enter__(self):return self
   def __exit__(self,*_):pass
   def read(self):return json.dumps({'commitOid':'f'*40}).encode()
  record={'status':'FAIL','error_message':'bounded'}
  with patch.dict(os.environ,{'HF_TOKEN':'hf_testsecret'},clear=True),patch.object(entry.urllib.request,'urlopen',return_value=Response()) as call:
   self.assertEqual(entry.persist_early(binding(),record),'f'*40)
  request=call.call_args.args[0];self.assertLess(len(request.data),4096);self.assertNotIn(b'hf_testsecret',request.data);self.assertIn(b'early-failure-terminal.json',request.data)
 def test_absolute_binding_opens_from_real_child_payload_cwd(self):
  with tempfile.TemporaryDirectory() as d:
   packet=Path(d);payload=packet/'payload';payload.mkdir();binding_path=packet/'binding.root.json';binding_path.write_text('{"ok":true}\n')
   relative=Path('binding.root.json');resolved=(packet/relative).resolve(strict=True)
   code='import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())'
   good=__import__('subprocess').run([sys.executable,'-c',code,str(resolved)],cwd=payload,capture_output=True,text=True)
   bad=__import__('subprocess').run([sys.executable,'-c',code,str(relative)],cwd=payload,capture_output=True,text=True)
   self.assertEqual(good.returncode,0);self.assertIn('"ok":true',good.stdout);self.assertNotEqual(bad.returncode,0)
  source=(H/'payload/cloud_entry.py').read_text();self.assertIn('binding_path=a.binding.resolve(strict=True)',source);self.assertEqual(source.count('str(binding_path)'),4);self.assertNotIn('str(a.binding)',source)
 def test_artifact_failure_reports_sanitized_path_and_location(self):
  with tempfile.TemporaryDirectory() as d:
   result=__import__('subprocess').run([sys.executable,'-B',str(H/'payload/artifact_upload.py'),'sentinel','missing-binding.json',d],cwd=H/'payload',capture_output=True,text=True,env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1'})
  self.assertEqual(result.returncode,1);report=json.loads(result.stdout);self.assertEqual(report['error_type'],'FileNotFoundError');self.assertIn('missing-binding.json',report['error_message']);self.assertTrue(any(row['file']=='artifact_upload.py' for row in report['stack']))
 def test_full_admission_accepts_admit_and_rejects_diagnostic(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);payload=root/'payload';payload.mkdir();old=entry.HERE;entry.HERE=payload
   try:
    b=binding();ad=root/'root-recipe-admission.json';armed=root/'watchdog-armed.json';armed.write_text(json.dumps({'name':'sepalith-cpt-'+b['run_id'],'deadline':b['absolute_deadline_utc']}));b['watchdog_armed_receipt_sha256']=entry.sha(armed)
    for status,accepted in [('ADMITTED_FULL_TRAINING',True),('ADMITTED_DIAGNOSTIC_ONLY',False)]:
     ad.write_text(json.dumps({'status':status,'admitted':True,'recipe':b['recipe_sha256']}));b['root_recipe_admission_sha256']=entry.sha(ad)
     if accepted:entry.verify_root_receipts(b)
     else:
      with self.assertRaisesRegex(ValueError,'full training'):entry.verify_root_receipts(b)
   finally:entry.HERE=old
 def test_prearm_and_finalize_bind_exact_fresh_watchdog(self):
  prepare=module('v3_prepare',H/'prepare_prearm_binding.py');finalize=module('v3_finalize',H/'finalize_binding_after_arm.py')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'binding.template.json').write_bytes((H/'binding.template.json').read_bytes());(root/'payload-manifest.json').write_bytes((H/'payload-manifest.json').read_bytes())
   admission=root/'root-recipe-admission.json';admission.write_text(json.dumps({'status':'ADMITTED_FULL_TRAINING','admitted':True,'recipe':'254de1150b0fcc8e8566178905542bdadc1255610a2dc3cb8a9168dc7017d538'}));pre=root/'binding.prearm.json';now=time.time();iso=lambda x:datetime.datetime.fromtimestamp(x,datetime.timezone.utc).isoformat();old=prepare.H;prepare.H=root
   try:
    with patch.object(sys,'argv',['prepare','--run-id','a'*32,'--absolute-deadline-utc',iso(now+28800),'--watchdog-armed-at-utc',iso(now-1),'--root-admission',str(admission),'--output',str(pre)]):prepare.main()
   finally:prepare.H=old
   state=root/'watchdog-state';state.mkdir();(state/'armed.json').write_text(json.dumps({'name':'sepalith-cpt-'+'a'*32,'deadline':iso(now+28800),'pid':os.getpid()}));armed=root/'watchdog-armed.json';out=root/'binding.root.json';old=finalize.HERE;finalize.HERE=root
   try:
    with patch.object(sys,'argv',['finalize','--prearm',str(pre),'--armed-state',str(state/'armed.json'),'--armed-receipt',str(armed),'--output',str(out)]):finalize.main()
   finally:finalize.HERE=old
   bound=json.loads(out.read_text());self.assertEqual(bound['input_revision'],'1e4df9c9db3cee1a7df5bbcc7caa62920d2ab2db');self.assertEqual(bound['watchdog_armed_receipt_sha256'],entry.sha(armed))
 def test_root_entry_and_scientific_lifecycle_unchanged(self):
  self.assertIn('cloud_entry.py', (H/'payload/root-bound-entry.sh').read_text())
  contract=(H/'payload/cloud_contract.py').read_text();entry_source=(H/'payload/cloud_entry.py').read_text();upload=(H/'payload/artifact_upload.py').read_text()
  for token in ('28800','29100','28.0','317,634,951,1268,1585'):self.assertIn(token,contract)
  self.assertIn('1902',entry_source);self.assertIn('step%317==0',upload);self.assertIn("spec.get('full') is not True",upload);self.assertIn("'optimizer.pt'",upload);self.assertIn("'rng_state.pth'",upload)
  for name in ('checkpoint_sidecar.py','cloud_contract.py','deadline_watchdog.py','managed_python.py','requirements.txt','root-bound-entry.sh','root_arm_watchdog.py','stage_inputs.py'):
   self.assertEqual((H/'payload'/name).read_bytes(),(V1/name).read_bytes(),name)
 def test_no_model_framework_imported(self):self.assertFalse(set(sys.modules)&{'torch','transformers','peft','gguf','ray'})
if __name__=='__main__':unittest.main(verbosity=2)
