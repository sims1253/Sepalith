"""Independent synthetic run_session success test. No subprocess/network/model execution."""
import contextlib,io,importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
HERE=Path(__file__).resolve().parent
PACKET=Path(os.environ['DAILY_REVIEW_PACKET'])
sys.path.insert(0,str(PACKET))
spec=importlib.util.spec_from_file_location('reviewed_daily_lan',PACKET/'daily_lan.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Success(unittest.TestCase):
 def test_real_run_session_reaches_notebook_ready_and_owned_terminal(self):
  with tempfile.TemporaryDirectory(dir=HERE) as temp:
   root=Path(temp);state=root/'state';lock=root/'lock';lock.touch();ledger=root/'ledger';ledger.write_text('| RTX 5090 | None | synthetic |\n')
   model=root/'synthetic-model-metadata';binary=root/'synthetic-bundle/llama-server';profile=json.loads((PACKET/'primary-manifest.json').read_text());bundle=profile['bundles'][0]
   metadata={model:profile['model'],**{binary.parent/item['name']:item for item in bundle['files']}}
   calls=[];processes={};uploaded=[];session=[None];real_sha=d.sha;real_stat=Path.stat;real_lock=d.exclusive_lock
   class Pipe(io.BytesIO):
    def close(self):uploaded.append(self.getvalue());super().close()
   class Proc:
    def __init__(self,argv,kwargs):
     self.pid=9000000+len(calls);self.argv=list(argv);self.returncode=None;self.stdin=Pipe() if kwargs['stdin']==d.subprocess.PIPE else None
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=0;return 0
   def spawn(argv,**kwargs):
    calls.append(list(argv));proc=Proc(argv,kwargs);processes[proc.pid]=proc;session[0]=Path(kwargs['stdout'].name).parent
    if len(calls)==1:kwargs['stdout'].write(b'offloaded 43/43 layers to GPU\n');kwargs['stdout'].flush()
    return proc
   def identity(pid):
    if pid in processes and processes[pid].returncode is not None:return None
    return {'pid':pid,'startTick':'17','uid':os.getuid()}
   def sha(path):
    path=Path(path)
    if path in metadata:return metadata[path]['sha256']
    return real_sha(path)
   def stat(path,*args,**kwargs):
    if path in metadata:return types.SimpleNamespace(st_size=metadata[path]['bytes'])
    return real_stat(path,*args,**kwargs)
   def fetch(url,expected_instance=None):
    if url.endswith('/props'):return {'model_path':str(model),'default_generation_settings':{'n_ctx':4096},'total_slots':1}
    binding=json.loads((session[0]/'binding.json').read_text());self.assertEqual(expected_instance,binding['instanceId'])
    return {'schema':1,'instanceId':binding['instanceId'],'modelSha256':profile['model']['sha256'],'serverSha256':bundle['launchProfile']['serverSha256'],'backend':'cuda','contextSize':4096,'maxOutputTokens':192,'renderer':profile['modelProfile']['renderer'],'cudaGraphOptimization':0}
   def sleep(seconds):
    if seconds==.5:(session[0]/'stop').touch()
   class Socket:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def bind(self,address):pass
    def setsockopt(self,*args):pass
   admission=root/'admission.json';admission.write_text(json.dumps({'schema':1,'status':'admitted','purpose':'daily-lan-deployment','sourceManifestSha256':real_sha(PACKET/'source-manifest.json'),'cudaLockPath':str(d.LOCK),'initialLeaseLedgerSha256':real_sha(ledger)}))
   with contextlib.ExitStack() as stack:
    for patch in [mock.patch.object(d,'MODEL',model),mock.patch.object(d,'BIN',binary),mock.patch.object(d,'LEDGER',ledger),mock.patch.object(d,'exclusive_lock',lambda:real_lock(lock)),mock.patch.object(d,'sha',side_effect=sha),mock.patch.object(Path,'stat',stat),mock.patch.object(d.subprocess,'Popen',side_effect=spawn),mock.patch.object(d,'identity',side_effect=identity),mock.patch.object(d,'signal_owned',return_value=True),mock.patch.object(d,'fetch_json',side_effect=fetch),mock.patch.object(d.socket,'socket',return_value=Socket()),mock.patch.object(d.signal,'signal'),mock.patch.object(d.time,'sleep',side_effect=sleep)]:stack.enter_context(patch)
    self.assertEqual(d.run_session(state,admission),0)
   self.assertEqual(len(calls),5);native,gateway,upload,tunnel,probe=calls
   self.assertIn('--daily-max-run-ms',gateway);self.assertEqual(gateway[-1],'28800000')
   self.assertIn('bind',upload[-1]);self.assertIn('probe',probe[-1]);self.assertIn('127.0.0.1:18403:127.0.0.1:18423',tunnel)
   self.assertEqual(len(uploaded),1);binding=json.loads(uploaded[0]);self.assertEqual(binding,json.loads((session[0]/'binding.json').read_text()))
   ready=json.loads((session[0]/'ready.json').read_text());terminal=json.loads((session[0]/'terminal.json').read_text())
   self.assertEqual(ready['instanceId'],binding['instanceId']);self.assertEqual(ready['status'],'desktop_and_notebook_forward_verified_extension_handshake_pending')
   self.assertIsNone(terminal['failure']);self.assertTrue(terminal['resourceReleaseProven']);self.assertEqual(len(terminal['cleanup']),5)
if __name__=='__main__':unittest.main(verbosity=2)
