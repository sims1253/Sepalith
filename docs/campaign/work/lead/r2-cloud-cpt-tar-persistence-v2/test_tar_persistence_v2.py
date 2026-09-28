import importlib.util,io,json,os,tarfile,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tp',HERE/'tar_persistence_v2.py');tp=importlib.util.module_from_spec(spec);spec.loader.exec_module(tp)

class TarPersistenceTest(unittest.TestCase):
 def fixture(self,root):
  p=Path(root)/'checkpoint-317';p.mkdir();files={
   'README.md':b'---\nbase_model: /home/m0hawk/private/invalid-local-parent\n---\n',
   'adapter_config.json':b'{}','adapter_model.safetensors':b'weights','optimizer.pt':b'optimizer',
   'rng_state.pth':b'rng','scheduler.pt':b'scheduler','trainer_state.json':b'{}',
   'campaign-state.json':json.dumps({'sampler':{'schedule_sha256':'a'*64}}).encode()}
  manifest={}
  for name,data in files.items():(p/name).write_bytes(data);manifest[name]={'bytes':len(data),'sha256':__import__('hashlib').sha256(data).hexdigest()}
  value={'full':True,'step':317,'files':manifest};(p/'campaign-manifest.json').write_text(json.dumps(value))
  return p,files
 def test_tar_preserves_every_byte_including_invalid_readme_and_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   p,files=self.fixture(d);scratch=Path(d)/'scratch';before=tp.file_inventory(p);meta=tp.build_tar(p,317,scratch);self.assertEqual(before,meta['files'])
   with tarfile.open(meta['tar_path']) as tf:
    got={m.name.split('/',1)[1]:tf.extractfile(m).read() for m in tf.getmembers()}
   self.assertEqual(got['README.md'],files['README.md']);self.assertEqual(got['campaign-manifest.json'],(p/'campaign-manifest.json').read_bytes())
   self.assertEqual(before,{n:{'bytes':len(v),'sha256':__import__('hashlib').sha256(v).hexdigest()} for n,v in got.items()})
 def test_member_paths_are_safe(self):
  for name in ('/absolute','../escape','x/../../escape','x\\escape',''):
   self.assertFalse(tp.safe_name(name),name)
  self.assertTrue(tp.safe_name('checkpoint-317/nested/file.bin'))
 def test_validated_tar_extracts_to_exact_resume_tree(self):
  with tempfile.TemporaryDirectory() as d:
   p,_=self.fixture(d);meta=tp.build_tar(p,317,Path(d)/'scratch');dest=Path(d)/'restore'
   with tarfile.open(meta['tar_path']) as tf:
    for m in tf.getmembers():self.assertTrue(m.isfile() and tp.safe_name(m.name))
    tf.extractall(dest)
   self.assertEqual(tp.file_inventory(p),tp.file_inventory(dest/'checkpoint-317'))
 def test_bounded_retry_reports_each_failure_and_backoff(self):
  calls=[];sleeps=[]
  def fail(run,step):calls.append(step);raise ValueError('https://host/path?token=secret')
  result,errors=tp.attempt_step(Path('/unused'),317,3,(2,5),fail,sleeps.append)
  self.assertIsNone(result);self.assertEqual(len(calls),3);self.assertEqual(sleeps,[2,5]);self.assertEqual([x['attempt'] for x in errors],[1,2,3]);self.assertNotIn('secret',json.dumps(errors));self.assertIn('REDACTED_QUERY',json.dumps(errors))
 def test_monitor_does_not_reenter_abandoned_step(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'artifacts/archive/full/checkpoint-317';p.mkdir(parents=True);(p/'campaign-manifest.json').write_text('{}')
   originals=tp.STEPS,tp.validate_checkpoint,tp.attempt_step
   calls=[]
   try:
    tp.STEPS=(317,);tp.validate_checkpoint=lambda folder,step: True
    def exhausted(run,step,*args,**kwargs):
     calls.append(step);row={'step':step,'attempt':3,'error_type':'Failure','error_message':'safe'}
     if kwargs.get('reporter'):kwargs['reporter'](row)
     return None,[row]
    tp.attempt_step=exhausted;result=tp.monitor(Path(d),poll=0,max_retries=3,backoffs=(0,0))
    self.assertEqual(calls,[317]);self.assertEqual(result['abandoned_steps'],[317]);self.assertEqual(result['status'],'bounded_exit')
   finally:tp.STEPS,tp.validate_checkpoint,tp.attempt_step=originals
 def test_present_corrupt_manifest_is_reported_once_and_abandoned(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);p=run/'artifacts/archive/full/checkpoint-317';p.mkdir(parents=True);(p/'campaign-manifest.json').write_text('{bad')
   old=tp.STEPS;tp.STEPS=(317,)
   try:result=tp.monitor(run,poll=0)
   finally:tp.STEPS=old
   self.assertEqual(result['abandoned_steps'],[317]);self.assertEqual(len(result['failures']),1);self.assertEqual(result['failures'][0]['attempt'],0)
   events=[json.loads(x) for x in (run/'persistence-staging/events.jsonl').read_text().splitlines()]
   self.assertEqual(sum(x['status']=='CHECKPOINT_INVALID' for x in events),1)
 def test_campaign_manifest_must_cover_all_nonmanifest_files(self):
  with tempfile.TemporaryDirectory() as d:
   p,_=self.fixture(d);(p/'untracked.bin').write_bytes(b'x')
   with self.assertRaisesRegex(ValueError,'inventory differs'):tp.validate_checkpoint(p,317)
 def test_retry_after_receipt_failure_does_not_reupload_tar(self):
  class Api:
   def __init__(self):self.files={};self.tar_uploads=0;self.fail_receipt=True;self.n=0
   def repo_info(self,*a,**k):return SimpleNamespace(private=True)
   def upload_file(self,*,path_or_fileobj,path_in_repo,**kwargs):
    if 'checkpoint-tar-receipts' in path_in_repo and self.fail_receipt:self.fail_receipt=False;raise RuntimeError('receipt unavailable')
    data=Path(path_or_fileobj).read_bytes();self.files[path_in_repo]=data;self.n+=1
    if path_in_repo.endswith('.tar'):self.tar_uploads+=1
    return SimpleNamespace(oid='r'+str(self.n).zfill(40))
   def list_repo_tree(self,*,path_in_repo,**kwargs):
    out=[]
    for name,data in self.files.items():
     if str(PurePath(name).parent)!=path_in_repo:continue
     if name.endswith('.tar'):out.append(SimpleNamespace(path=name,size=len(data),lfs={'sha256':__import__('hashlib').sha256(data).hexdigest()},blob_id=None))
     else:
      h=__import__('hashlib').sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();out.append(SimpleNamespace(path=name,size=len(data),lfs=None,blob_id=h))
    return out
  from pathlib import PurePosixPath as PurePath
  with tempfile.TemporaryDirectory() as d:
   run=Path(d);(run/'artifacts/archive/full').mkdir(parents=True);p,_=self.fixture(run/'artifacts/archive/full');api=Api();old=os.environ.get('HF_TOKEN');os.environ['HF_TOKEN']='hf_test'
   try:
    with self.assertRaisesRegex(RuntimeError,'receipt unavailable'):tp.upload_step(run,317,api)
    result=tp.upload_step(run,317,api);self.assertEqual(result['step'],317);self.assertEqual(api.tar_uploads,1)
   finally:
    if old is None:os.environ.pop('HF_TOKEN',None)
    else:os.environ['HF_TOKEN']=old
 def test_controller_remote_and_nested_source_compile(self):
  import ast
  ast.parse((HERE/'root_tar_monitor_controller_v2.py').read_text());ast.parse((HERE/'tar_persistence_v2.py').read_text())
  c={'__file__':str(HERE/'root_tar_monitor_controller_v2.py'),'__name__':'test_controller'};exec(compile((HERE/'root_tar_monitor_controller_v2.py').read_text(),str(HERE/'root_tar_monitor_controller_v2.py'),'exec'),c)
  compile(c['REMOTE'],'REMOTE','exec')
 def test_authorizations_are_disabled_and_action_scoped(self):
  monitor=json.loads((HERE/'root-authorization-monitor.template.json').read_text())
  self.assertIs(monitor['authorized'],False);self.assertEqual(monitor['action'],'launch_bounded_tar_monitor_v2');self.assertEqual(monitor['run_id'],tp.RUN_ID)

if __name__=='__main__':unittest.main()
