"""Root-admitted CPT cloud orchestration; no provider submission API."""
import argparse,json,os,shutil,signal,subprocess,sys,tempfile,time
from pathlib import Path
from cloud_contract import IMAGE,remaining,require,sha,stamp,utc,validate_binding,write
from managed_python import PROBE,discover,validate as validate_python

HERE=Path(__file__).resolve().parent;STOP=False
def stop_requested(*_):
 global STOP;STOP=True
def mem_available():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise ValueError('Linux MemAvailable unavailable')
def proc(pid):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();v=raw[raw.rfind(')')+2:].split();return {'pid':pid,'state':v[0],'pgrp':int(v[2]),'session':int(v[3]),'start_tick':int(v[19])}
 except (OSError,ValueError,IndexError):return None
def members(group):
 out=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   r=proc(int(p.name))
   if r and r['pgrp']==group and r['session']==group and r['state']!='Z':out.append(r)
 return out
def cleanup_group(group,grace=10):
 owned={r['pid']:r for r in members(group)}
 for sig in (signal.SIGTERM,signal.SIGKILL):
  for pid,old in list(owned.items()):
   current=proc(pid)
   if current and current['start_tick']==old['start_tick']:
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
  until=time.monotonic()+(grace if sig==signal.SIGTERM else 2)
  while time.monotonic()<until:
   live=members(group)
   if not live:return {'owned':list(owned.values()),'remaining':[]}
   owned.update({r['pid']:r for r in live});time.sleep(.2)
 return {'owned':list(owned.values()),'remaining':members(group)}
def guarded(argv,env,cwd,deadline,log,floor=8*1024**3):
 require(deadline>time.time() and mem_available()>=floor,'phase admission failed')
 with Path(log).open('xb') as f:
  child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);reason=None
  while child.poll() is None:
   if STOP:reason='termination requested'
   elif time.time()>=deadline:reason='phase deadline'
   elif mem_available()<floor:reason='host memory floor'
   elif Path(log).stat().st_size>64*1024**2:reason='log size ceiling'
   if reason:break
   time.sleep(.5)
  release=cleanup_group(child.pid);code=child.poll()
 require(reason is None and code==0 and not release['remaining'],'guarded phase failed')
 return code
def training_terminal(run):
 q=Path(run)/'artifacts/training/terminal.json';require(q.is_file(),'training terminal absent');x=json.loads(q.read_text())
 require(x.get('step')==1902 and x.get('status')=='schedule_complete','full CPT schedule incomplete');return x
def clean_env(run):
 e=dict(os.environ);e.pop('HF_TOKEN',None);e.pop('HUGGING_FACE_HUB_TOKEN',None)
 e.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',UV_CACHE_DIR=str(Path(run)/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(Path(run)/'python'),HF_HOME=str(Path(run)/'hf-cache'))
 return e
def verify_payload(binding):
 manifest=HERE.parent/'payload-manifest.json';require(sha(manifest)==binding['payload_manifest_sha256'],'payload manifest differs')
 spec=json.loads(manifest.read_text())
 for row in spec['files']:
  p=HERE.parent/row['path'];require(p.is_file() and not p.is_symlink() and sha(p)==row['sha256'],'payload source differs')
def verify_root_receipts(binding):
 root=HERE.parent
 admission=root/binding['root_recipe_admission_relative_path'];armed=root/binding['watchdog_armed_receipt_relative_path']
 require(admission.is_file() and not admission.is_symlink() and sha(admission)==binding['root_recipe_admission_sha256'],'root recipe admission receipt differs')
 a=json.loads(admission.read_text());status=str(a.get('status','')).lower();require(a.get('admitted') is True or any(x in status for x in ('accept','admit','pass')),'root recipe admission is not affirmative')
 frozen=json.dumps(a,sort_keys=True);require(binding['recipe_sha256'] in frozen,'root admission does not bind frozen recipe')
 require(armed.is_file() and not armed.is_symlink() and sha(armed)==binding['watchdog_armed_receipt_sha256'],'watchdog armed receipt differs')
 w=json.loads(armed.read_text());require(w.get('name')=='sepalith-cpt-'+binding['run_id'] and utc(w.get('deadline',''))==utc(binding['absolute_deadline_utc']),'watchdog armed identity differs')
def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);a=p.parse_args();b=json.loads(a.binding.read_text());hard=validate_binding(b);verify_payload(b);verify_root_receipts(b)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_requested)
 require(Path('/mnt/local_storage').is_dir() and shutil.disk_usage('/mnt/local_storage').free>=24*1024**3,'local storage below24GiB')
 require(shutil.disk_usage('/').free>=16*1024**3,'root filesystem below16GiB')
 run=Path(tempfile.mkdtemp(prefix='sepalith-cpt-'+b['run_id']+'-',dir='/mnt/local_storage'));(run/'artifacts').mkdir();write(run/'artifacts/binding.json',b)
 env=clean_env(run);python=None;sidecar=None;stop=run/'stop-sidecar';success=False;failure=None
 try:
  bootstrap=remaining(hard,b['phase_seconds']['bootstrap'],b['phase_seconds']['staging']+b['phase_seconds']['training']+b['phase_seconds']['upload']+b['phase_seconds']['cleanup'])
  guarded([sys.executable,'-m','pip','--disable-pip-version-check','--no-input','install','--target',str(run/'bootstrap'),'uv==0.11.23'],env,HERE,bootstrap,run/'bootstrap.log')
  uv=run/'bootstrap/bin/uv';require(uv.is_file(),'pinned uv absent');guarded([str(uv),'--version'],env,HERE,bootstrap,run/'uv-version.log');require((run/'uv-version.log').read_text().strip().split()[:2]==['uv','0.11.23'],'uv version differs')
  guarded([str(uv),'python','install','3.10.19'],env,HERE,bootstrap,run/'python.log')
  interpreter=discover(uv,run/'python',env,bootstrap,guarded,HERE,run/'python-discovery.log');guarded([str(interpreter),'-I','-c',PROBE],env,HERE,bootstrap,run/'python-probe.log');validate_python(json.loads((run/'python-probe.log').read_text()),interpreter,run/'python')
  guarded([str(uv),'venv','--python',str(interpreter),str(run/'venv')],env,HERE,bootstrap,run/'venv.log');python=run/'venv/bin/python'
  guarded([str(uv),'pip','install','--python',str(python),'-r',str(HERE/'requirements.txt')],env,HERE,bootstrap,run/'packages.log')
  token_env=dict(env)
  if 'HF_TOKEN' in os.environ:token_env['HF_TOKEN']=os.environ['HF_TOKEN']
  stage_deadline=remaining(hard,b['phase_seconds']['staging'],b['phase_seconds']['training']+b['phase_seconds']['upload']+b['phase_seconds']['cleanup'])
  guarded([str(python),str(HERE/'artifact_upload.py'),'sentinel',str(a.binding),str(run)],token_env,HERE,stage_deadline,run/'sentinel.log')
  train_deadline=remaining(hard,b['phase_seconds']['training'],b['phase_seconds']['upload']+b['phase_seconds']['cleanup'],now=stage_deadline)
  deadline_text=__import__('datetime').datetime.fromtimestamp(train_deadline,__import__('datetime').timezone.utc).isoformat()
  guarded([str(python),str(HERE/'stage_inputs.py'),str(a.binding),str(run),'--deadline',deadline_text,'--max-attempt',str(train_deadline-time.time())],token_env,HERE,stage_deadline,run/'staging.log')
  stage=json.loads((run/'artifacts/staging-receipt.json').read_text());recipe=Path(stage['relocated_recipe'])
  training_source=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1/source/experiments/training')
  package_source=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1/source/packages/sepalith/src')
  soft_text=__import__('datetime').datetime.fromtimestamp(train_deadline-60,__import__('datetime').timezone.utc).isoformat()
  trainer_env=dict(env,PYTHONPATH=str(training_source)+os.pathsep+str(package_source),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SEPALITH_CAMPAIGN_SOFT_DEADLINE=soft_text,SEPALITH_CAMPAIGN_HARD_DEADLINE=deadline_text)
  sidecar=subprocess.Popen([str(python),str(HERE/'checkpoint_sidecar.py'),str(a.binding),str(run),'--stop-file',str(stop)],env=token_env,cwd=HERE,stdout=(run/'sidecar.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
  guarded([str(python),str(training_source/'campaign_cpt.py'),str(recipe)],trainer_env,HERE,train_deadline,run/'training.log')
  training_terminal(run);stop.touch(exist_ok=False);require(sidecar.wait(timeout=1800)==0,'checkpoint persistence sidecar failed');sidecar=None;success=True
 except Exception as e:failure=type(e).__name__;write(run/'artifacts/entry-failure.json',{'at':stamp(),'status':'failed','error_type':failure})
 finally:
  if not stop.exists():stop.touch()
  if sidecar is not None:
   try:sidecar.wait(timeout=900)
   except subprocess.TimeoutExpired:cleanup_group(sidecar.pid)
  write(run/'artifacts/local-terminal.json',{'at':stamp(),'training_success':success,'error_type':failure})
  if python and python.is_file() and time.time()<hard-60:
   token_env=dict(clean_env(run));
   if 'HF_TOKEN' in os.environ:token_env['HF_TOKEN']=os.environ['HF_TOKEN']
   try:guarded([str(python),str(HERE/'artifact_upload.py'),'final',str(a.binding),str(run)],token_env,HERE,remaining(hard,b['phase_seconds']['upload'],b['phase_seconds']['cleanup']),run/'final-upload.log')
   except Exception:success=False
 if not success:raise SystemExit(1)
if __name__=='__main__':main()
