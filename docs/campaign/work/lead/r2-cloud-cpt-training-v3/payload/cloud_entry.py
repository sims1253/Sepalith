"""Root-admitted CPT cloud orchestration; no provider submission API."""
import argparse,base64,datetime,json,os,re,shutil,signal,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from cloud_contract import IMAGE,remaining,require,sha,stamp,utc,validate_binding,write
from managed_python import PROBE,discover,validate as validate_python

HERE=Path(__file__).resolve().parent;STOP=False;MAX_TAIL_BYTES=4096
SECRET=re.compile(rb'hf_[A-Za-z0-9]+')
def sanitized(data,limit):
 data=SECRET.sub(b'[REDACTED]',data);data=re.sub(rb'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',rb'\1[REDACTED]',data);data=re.sub(rb'(://)[^/@\s:]+:[^/@\s]+@',rb'\1[REDACTED]@',data)
 text=data[-limit:].decode('utf-8','replace')
 while len(text.encode('utf-8'))>limit:text=text[1:]
 return text
def safe_message(error):return sanitized(str(error).encode('utf-8','replace'),1024)
def safe_tail(path):
 try:return sanitized(Path(path).read_bytes()[-MAX_TAIL_BYTES:],MAX_TAIL_BYTES)
 except OSError:return ''
def event(phase,status,**fields):print(json.dumps({'schema':'sepalith.cloud-cpt.training-event.v3','at':stamp(),'phase':phase,'status':status,**fields},sort_keys=True),flush=True)
def persist_early(binding,record):
 """Persist one small failure terminal using only the provider Python stdlib."""
 token=os.environ.get('HF_TOKEN');require(bool(token),'private persistence token absent')
 require(binding['artifact_repo']=='scholzmx/sepalith-lora' and binding['artifact_prefix']=='r2-cpt/'+binding['run_id'],'private failure route differs')
 remote=binding['artifact_prefix']+'/early-failure-terminal.json';body=json.dumps(record,sort_keys=True).encode()
 payload=[{'key':'header','value':{'summary':'CPT training early failure','description':''}},{'key':'file','value':{'content':base64.b64encode(body).decode(),'path':remote,'encoding':'base64'}}]
 data=b''.join(json.dumps(row,separators=(',',':')).encode()+b'\n' for row in payload)
 request=urllib.request.Request('https://huggingface.co/api/models/'+binding['artifact_repo']+'/commit/main',data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/x-ndjson'},method='POST')
 with urllib.request.urlopen(request,timeout=30) as response:result=json.loads(response.read())
 revision=result.get('commitOid','');require(re.fullmatch('[0-9a-f]{40,64}',revision or ''),'early failure persistence revision missing');return revision
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
 phase=Path(log).stem;event(phase,'START',argv=[sanitized(str(value).encode('utf-8','replace'),1024) for value in argv],memory_floor_bytes=floor)
 child=None;reason=None;release={'remaining':[]};code=None
 try:
  available=mem_available();require(deadline>time.time(),phase+' deadline exhausted before launch');require(available>=floor,phase+f' memory admission failed: available={available} floor={floor}')
  with Path(log).open('xb') as f:
   child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    if STOP:reason='termination requested'
    elif time.time()>=deadline:reason='phase deadline'
    elif mem_available()<floor:reason='host memory floor'
    elif Path(log).stat().st_size>64*1024**2:reason='log size ceiling'
    if reason:break
    time.sleep(.5)
   release=cleanup_group(child.pid);code=child.poll()
  require(reason is None,phase+' stopped: '+str(reason));require(code==0,phase+f' exit code {code}');require(not release['remaining'],phase+' owned processes remain')
  event(phase,'END',exit_code=code,log_tail=safe_tail(log));return code
 except BaseException as error:
  if child is not None and child.poll() is None:release=cleanup_group(child.pid)
  event(phase,'EXCEPTION',exit_code=code,error_type=type(error).__name__,error_message=safe_message(error),log_tail=safe_tail(log));raise
def training_terminal(run):
 q=Path(run)/'artifacts/training/terminal.json';require(q.is_file(),'training terminal absent');x=json.loads(q.read_text())
 require(x.get('step')==1902 and x.get('status')=='schedule_complete','full CPT schedule incomplete');return x
def clean_env(run):
 markers=('TOKEN','SECRET','CREDENTIAL','PASSWORD','PASSWD','API_KEY','ACCESS_KEY','PRIVATE_KEY','AUTH')
 e={k:v for k,v in os.environ.items() if not any(marker in k.upper() for marker in markers)}
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
 a=json.loads(admission.read_text());status=str(a.get('status','')).lower();require('diagnostic' not in status and (a.get('admitted') is True or any(x in status for x in ('accept','admit','pass'))),'full training root recipe admission is not affirmative')
 frozen=json.dumps(a,sort_keys=True);require(binding['recipe_sha256'] in frozen,'root admission does not bind frozen recipe')
 require(armed.is_file() and not armed.is_symlink() and sha(armed)==binding['watchdog_armed_receipt_sha256'],'watchdog armed receipt differs')
 w=json.loads(armed.read_text());require(w.get('name')=='sepalith-cpt-'+binding['run_id'] and utc(w.get('deadline',''))==utc(binding['absolute_deadline_utc']),'watchdog armed identity differs')
def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);a=p.parse_args()
 b=None;hard=None;run=None;python=None;sidecar=None;sidecar_log=None;stop=None;success=False;failure=None;current='binding';trusted=False
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_requested)
 try:
  event(current,'START');b=json.loads(a.binding.read_text());hard=validate_binding(b);trusted=True;event(current,'END')
  current='payload';event(current,'START');verify_payload(b);event(current,'END')
  current='root_receipts';event(current,'START');verify_root_receipts(b);event(current,'END')
  current='storage';event(current,'START');require(Path('/mnt/local_storage').is_dir() and shutil.disk_usage('/mnt/local_storage').free>=24*1024**3,'local storage below24GiB');require(shutil.disk_usage('/').free>=16*1024**3,'root filesystem below16GiB');event(current,'END')
  run=Path(tempfile.mkdtemp(prefix='sepalith-cpt-'+b['run_id']+'-',dir='/mnt/local_storage'));(run/'artifacts').mkdir();write(run/'artifacts/binding.json',b)
  env=clean_env(run);stop=run/'stop-sidecar'
  bootstrap=remaining(hard,b['phase_seconds']['bootstrap'],b['phase_seconds']['staging']+b['phase_seconds']['training']+b['phase_seconds']['upload']+b['phase_seconds']['cleanup'])
  current='pip_uv';guarded([sys.executable,'-m','pip','--disable-pip-version-check','--no-input','install','--target',str(run/'bootstrap'),'uv==0.11.23'],env,HERE,bootstrap,run/'bootstrap.log')
  uv=run/'bootstrap/bin/uv';require(uv.is_file(),'pinned uv absent');current='uv_version';guarded([str(uv),'--version'],env,HERE,bootstrap,run/'uv-version.log');require((run/'uv-version.log').read_text().strip().split()[:2]==['uv','0.11.23'],'uv version differs')
  current='python_install';guarded([str(uv),'python','install','3.10.19'],env,HERE,bootstrap,run/'python.log')
  current='python_discovery';interpreter=discover(uv,run/'python',env,bootstrap,guarded,HERE,run/'python-discovery.log');current='python_probe';guarded([str(interpreter),'-I','-c',PROBE],env,HERE,bootstrap,run/'python-probe.log');validate_python(json.loads((run/'python-probe.log').read_text()),interpreter,run/'python')
  current='venv';guarded([str(uv),'venv','--python',str(interpreter),str(run/'venv')],env,HERE,bootstrap,run/'venv.log');python=run/'venv/bin/python'
  current='requirements';guarded([str(uv),'pip','install','--python',str(python),'-r',str(HERE/'requirements.txt')],env,HERE,bootstrap,run/'packages.log')
  token_env=dict(env)
  if 'HF_TOKEN' in os.environ:token_env['HF_TOKEN']=os.environ['HF_TOKEN']
  stage_deadline=remaining(hard,b['phase_seconds']['staging'],b['phase_seconds']['training']+b['phase_seconds']['upload']+b['phase_seconds']['cleanup'])
  current='sentinel';guarded([str(python),str(HERE/'artifact_upload.py'),'sentinel',str(a.binding),str(run)],token_env,HERE,stage_deadline,run/'sentinel.log')
  train_deadline=remaining(hard,b['phase_seconds']['training'],b['phase_seconds']['upload']+b['phase_seconds']['cleanup'],now=stage_deadline)
  deadline_text=__import__('datetime').datetime.fromtimestamp(train_deadline,__import__('datetime').timezone.utc).isoformat()
  current='staging';guarded([str(python),str(HERE/'stage_inputs.py'),str(a.binding),str(run),'--deadline',deadline_text,'--max-attempt',str(train_deadline-time.time())],token_env,HERE,stage_deadline,run/'staging.log')
  stage=json.loads((run/'artifacts/staging-receipt.json').read_text());recipe=Path(stage['relocated_recipe'])
  training_source=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1/source/experiments/training')
  package_source=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-remaining-v1/source/packages/sepalith/src')
  soft_text=__import__('datetime').datetime.fromtimestamp(train_deadline-60,__import__('datetime').timezone.utc).isoformat()
  trainer_env=dict(env,PYTHONPATH=str(training_source)+os.pathsep+str(package_source),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SEPALITH_CAMPAIGN_SOFT_DEADLINE=soft_text,SEPALITH_CAMPAIGN_HARD_DEADLINE=deadline_text)
  current='checkpoint_sidecar_start';sidecar_log=(run/'sidecar.log').open('xb');sidecar=subprocess.Popen([str(python),str(HERE/'checkpoint_sidecar.py'),str(a.binding),str(run),'--stop-file',str(stop)],env=token_env,cwd=HERE,stdout=sidecar_log,stderr=subprocess.STDOUT,start_new_session=True)
  current='training';guarded([str(python),str(training_source/'campaign_cpt.py'),str(recipe)],trainer_env,HERE,train_deadline,run/'training.log')
  training_terminal(run);stop.touch(exist_ok=False);require(sidecar.wait(timeout=1800)==0,'checkpoint persistence sidecar failed');sidecar=None;success=True
 except BaseException as e:
  failure={'phase':current,'error_type':type(e).__name__,'error_message':safe_message(e),'log_tail':safe_tail(run/(current+'.log')) if run else ''};event(current,'EXCEPTION',error_type=failure['error_type'],error_message=failure['error_message'],log_tail=failure['log_tail'])
  if run:write(run/'artifacts/entry-failure.json',dict({'at':stamp(),'status':'failed'},**failure))
 finally:
  training_complete=success
  if stop is not None and not stop.exists():stop.touch()
  if sidecar is not None:
   try:sidecar.wait(timeout=900)
   except subprocess.TimeoutExpired:cleanup_group(sidecar.pid)
  if sidecar_log is not None:sidecar_log.close()
  if run:write(run/'artifacts/local-terminal.json',{'at':stamp(),'training_success':success,'failure':failure})
  if run and python and python.is_file() and time.time()<hard-60:
   token_env=dict(clean_env(run));
   if 'HF_TOKEN' in os.environ:token_env['HF_TOKEN']=os.environ['HF_TOKEN']
   try:guarded([str(python),str(HERE/'artifact_upload.py'),'final',str(a.binding),str(run)],token_env,HERE,remaining(hard,b['phase_seconds']['upload'],b['phase_seconds']['cleanup']),run/'final-upload.log')
   except BaseException as e:
    success=False
    if failure is None:failure={'phase':'final_upload','error_type':type(e).__name__,'error_message':safe_message(e),'log_tail':safe_tail(run/'final-upload.log')}
  if not success and trusted:
   terminal={'schema':'sepalith.cloud-cpt.early-failure-terminal.v3','status':'FAIL','run_id':b['run_id'],'at':stamp(),'failure':failure,'training_complete':training_complete,'credential_persisted':False}
   try:terminal['persistence_revision']=persist_early(b,terminal);event('early_failure_persistence','END',revision=terminal['persistence_revision'])
   except BaseException as e:event('early_failure_persistence','EXCEPTION',error_type=type(e).__name__,error_message=safe_message(e))
 if not success:raise SystemExit(1)
if __name__=='__main__':main()
