import datetime,importlib.util,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'payload'))
import artifact_upload as upload
import checkpoint_sidecar as sidecar
import cloud_contract as contract
import cloud_entry as entry
import deadline_watchdog as watchdog
import stage_inputs as stage

_uploader_spec=importlib.util.spec_from_file_location('cloud_cpt_private_uploader',HERE/'upload_private_inputs.py')
private_uploader=importlib.util.module_from_spec(_uploader_spec);_uploader_spec.loader.exec_module(private_uploader)

def binding(now=1_800_000_000):
 iso=lambda x:datetime.datetime.fromtimestamp(x,datetime.timezone.utc).isoformat()
 return {'schema':'sepalith.cloud-cpt.binding.v1','admitted':True,'run_id':'a'*32,
 'image_uri':contract.IMAGE,'instance_type':contract.INSTANCE,'input_repo':contract.REPO,
 'artifact_repo':contract.REPO,'artifact_prefix':'r2-cpt/'+'a'*32,
 'input_manifest_sha256':contract.TRANSPORT_SHA,'recipe_sha256':contract.RECIPE_SHA,
 'input_revision':'b'*40,'root_recipe_admission_sha256':'c'*64,'root_recipe_admission_relative_path':'root-recipe-admission.json',
 'watchdog_armed_receipt_sha256':'d'*64,'watchdog_armed_receipt_relative_path':'watchdog-armed.json','payload_manifest_sha256':'e'*64,'watchdog_armed_at_utc':iso(now-1),
 'absolute_deadline_utc':iso(now+28800),'provider_timeout_seconds':28800,
 'watchdog_timeout_seconds':29100,'max_incremental_charge_usd':28.0,
 'phase_seconds':{'bootstrap':1800,'staging':3600,'training':19800,'upload':3000,'cleanup':300},'resume':None}

class ContractTests(unittest.TestCase):
 def test_binding_and_charge_pass(self):self.assertGreater(contract.validate_binding(binding(),now=1_800_000_000),1_800_000_000)
 def test_missing_root_admission_rejected(self):
  b=binding();b['root_recipe_admission_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'root recipe admission'):contract.validate_binding(b,now=1_800_000_000)
 def test_secret_field_rejected(self):
  b=binding();b['HF_TOKEN']='nope'
  with self.assertRaisesRegex(ValueError,'credential'):contract.validate_binding(b,now=1_800_000_000)
 def test_resume_requires_full_cadence(self):
  b=binding();b['resume']={'revision':'e'*40,'inventory_path':'r2-cpt/'+'f'*32+'/checkpoint-receipts/checkpoint-318.json','inventory_sha256':'1'*64,'step':318}
  with self.assertRaisesRegex(ValueError,'full cadence'):contract.validate_binding(b,now=1_800_000_000)
 def test_frozen_payload_manifest_verifies(self):
  b=binding();b['payload_manifest_sha256']=contract.sha(HERE/'payload-manifest.json');entry.verify_payload(b)
 def test_absent_remote_prefix_is_empty_upload(self):
  RemoteEntryNotFoundError=type('RemoteEntryNotFoundError',(Exception,),{})
  class Api:
   def list_repo_tree(self,*args,**kwargs):
    def lazy():
     raise RemoteEntryNotFoundError()
     yield
    return lazy()
  self.assertEqual(private_uploader.remote_files(Api(),'missing-prefix'),{})

class RelocationTests(unittest.TestCase):
 def recipe(self):return {'stage':'cpt_raw_r_v1','model_path':'/old/model','output_dir':'/old/out','archive_dir':'/old/archive','parameters':{'max_steps':1902,'per_device_batch':2,'gradient_accumulation':8},'identity':{'policy':{'initialization':'new_lora_on_merged_cpt_parent'}},'inputs':[{'path':'/old/data','sha256':'a'}],'resume_from':None}
 def test_fresh_relocation_preserves_identity(self):
  with tempfile.TemporaryDirectory() as d:
   old=self.recipe();new=stage.relocate(old,{'/old/data':'/new/data'},Path(d),{},'2030-01-01T00:00:00+00:00',100)
   self.assertEqual(new['inputs'][0]['path'],'/new/data');self.assertEqual(new['model_path'],'/old/model');self.assertEqual(new['identity'],old['identity']);self.assertIsNone(new['resume_from']);self.assertTrue(new['launch_authorized'])
 def test_exact_materialization_rejects_unbound_root(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'source';source.write_text('x')
   with self.assertRaisesRegex(ValueError,'outside frozen roots'):stage.materialize_exact(source,'/tmp/unbound')
 def test_resume_binding_is_exact(self):
  with tempfile.TemporaryDirectory() as d:
   rb={'path':'/resume','binding':{'schema':'sepalith.cpt.full-cadence-resume.v1'}}
   new=stage.relocate(self.recipe(),{'/old/data':'/new/data'},Path(d),{'relocated_resume':rb},'2030-01-01T00:00:00+00:00',100)
   self.assertEqual(new['resume_from'],'/resume');self.assertEqual(new['resume_binding'],rb['binding'])

class PersistenceTests(unittest.TestCase):
 def complete_checkpoint(self,folder,step=317):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  names=['adapter_config.json','adapter_model.safetensors','campaign-state.json','optimizer.pt','rng_state.pth','scheduler.pt','trainer_state.json']
  for name in names:
   value=json.dumps({'sampler':{'schedule_sha256':'f'*64}}) if name=='campaign-state.json' else name
   (folder/name).write_text(value)
  files={name:{'bytes':(folder/name).stat().st_size,'sha256':contract.sha(folder/name)} for name in names}
  (folder/'campaign-manifest.json').write_text(json.dumps({'step':step,'full':True,'files':files}))
 def test_incomplete_checkpoint_not_ready(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'campaign-manifest.json').write_text(json.dumps({'step':317,'full':True,'files':{'missing':{'bytes':1,'sha256':'0'*64}}}))
   self.assertIsNone(upload.checkpoint_complete(p))
 def test_complete_checkpoint_ready_for_sidecar(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'checkpoint-317';self.complete_checkpoint(p)
   self.assertEqual(sidecar.ready(root,set())[0][0],317)
 def test_checkpoint_retry_after_local_receipt_preserves_revision(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';folder=Path(d)/'checkpoint-317';self.complete_checkpoint(folder)
   class Api:
    def __init__(self):self.folder_uploads=0;self.receipt_attempts=0
    def upload_folder(self,**kwargs):self.folder_uploads+=1;return SimpleNamespace(oid='a'*40)
    def upload_file(self,**kwargs):
     self.receipt_attempts+=1
     if self.receipt_attempts==1:raise RuntimeError('injected receipt failure')
     return SimpleNamespace(oid='b'*40)
    def file_exists(self,*args,**kwargs):return False
   api=Api()
   def download(*args,**kwargs):return folder/'campaign-manifest.json' if kwargs['filename'].endswith('/campaign-manifest.json') else run/'artifacts/cloud-persistence/checkpoint-317.json'
   b={'artifact_repo':'private/repo','artifact_prefix':'r2-cpt/'+'c'*32}
   with patch.object(upload,'client',return_value=(api,download,'token')),patch.object(upload,'validate_remote'):
    with self.assertRaisesRegex(RuntimeError,'injected'):upload.upload_checkpoint(run,b,folder)
    local=run/'artifacts/cloud-persistence/checkpoint-317.json';self.assertTrue(local.is_file())
    result=upload.upload_checkpoint(run,b,folder)
   self.assertEqual(api.folder_uploads,1);self.assertEqual(result['revision'],'a'*40);self.assertEqual(result['receipt_revision'],'b'*40)
 def test_terminal_requires_full_schedule(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'artifacts/training';p.mkdir(parents=True);(p/'terminal.json').write_text(json.dumps({'step':317,'status':'lead_decision'}))
   with self.assertRaisesRegex(ValueError,'incomplete'):entry.training_terminal(d)

class LifecycleTests(unittest.TestCase):
 def test_watchdog_timeout_requests_termination(self):
  calls=[];clock=[0]
  def now():return clock[0]
  def sleep(n):clock[0]+=n
  def call(op,name):calls.append((op,name));return {'ok':True,'termination_requested':True} if op=='terminate' else {'ok':False}
  result=watchdog.monitor('sepalith-cpt-'+'a'*32,20,lambda _:None,clock=now,sleep=sleep,call=call)
  self.assertEqual(result['status'],'termination_requested');self.assertEqual(calls[-1][0],'terminate')
 def test_final_cleanup_kills_owned_group(self):
  child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
  try:
   result=entry.cleanup_group(child.pid,grace=.1);self.assertEqual(result['remaining'],[])
  finally:
   if child.poll() is None:child.kill()

if __name__=='__main__':unittest.main()
