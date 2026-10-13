"""Root-admitted cloud orchestration. No provider API or credential persistence."""
import argparse, datetime, hashlib, importlib.util, json, math, os, platform, re, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from entry_observer import emit,safe_reason,setup_tail,telemetry_snapshot,rss_bytes
HERE=Path(__file__).resolve().parent
IMAGE='docker.io/anyscale/ray@sha256:3b904e7cbb1736a8a17359708197d62333daa0bc8285d7de66f1d22aa78754f9'
REVISION='8dc5f6055b90fe4b9422340810b270b9569f37f3'
MODEL_PINS={'model.safetensors':'38a28680f6208242a0de7c84627343d44cee517b49c2b39d1afd706be6beabad',
 'config.json':'59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180',
 'generation_config.json':'9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1',
 'tokenizer.json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
 'tokenizer_config.json':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'}
STOP=False
def stop_requested(*_):
 global STOP;STOP=True
def require(ok,reason):
 if not ok:raise ValueError(reason)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def utc(value):
 d=datetime.datetime.fromisoformat(value.replace('Z','+00:00'));require(d.tzinfo is not None,'UTC offset missing');return d.timestamp()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bounded_path(root,relative):
 p=Path(relative);require(not p.is_absolute() and '..' not in p.parts and str(p) not in ('','.','/'),'unsafe payload path')
 candidate=root/p
 require(not any(q.is_symlink() for q in [candidate,*candidate.parents] if q!=root.parent),'symlink payload path')
 require(candidate.resolve().is_relative_to(root.resolve()),'payload escaped root')
 return candidate
def validate_binding(b,now=None):
 now=time.time() if now is None else now
 require(b.get('schema')==1 and b.get('admitted') is True,'root payload admission is pending')
 require(re.fullmatch('[0-9a-f]{32}',b.get('run_id','')),'fresh UUID32 required')
 require(b.get('image_uri')==IMAGE and b.get('instance_type')=='g5.2xlarge','image or instance differs')
 require(b.get('provider_timeout_seconds')==6900 and b.get('watchdog_timeout_seconds')==7200,'deadline layers differ')
 require(re.fullmatch('[0-9a-f]{64}',b.get('watchdog_armed_receipt_sha256','')),'root watchdog binding missing')
 armed=utc(b['watchdog_armed_at_utc']);hard=min(armed+6900,utc(b['absolute_deadline_utc']))
 require(armed<=now<hard and hard<=utc('2026-09-14T06:15:00Z'),'expired or invalid cloud window')
 require(utc(b['absolute_deadline_utc'])<=armed+7200,'watchdog limit expanded')
 require(b.get('artifact_repo')=='scholzmx/sepalith-lora' and b.get('artifact_prefix')=='r2-control/'+b['run_id'],'private unique artifact prefix differs')
 require(b.get('setup_seconds')==1800 and b.get('training_seconds')==4200 and b.get('upload_seconds')==600 and b.get('cleanup_seconds')==60,'phase budgets differ')
 require(not any('TOKEN' in str(k).upper() or 'SECRET' in str(k).upper() for k in b),'credential fields forbidden')
 return hard
def remaining_phase(hard,cap,reserve,now=None):
 now=time.time() if now is None else now;remaining=min(cap,hard-now-reserve)
 require(math.isfinite(remaining) and remaining>0,'no phase budget remains')
 return now+remaining
def mem_available():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise ValueError('Linux MemAvailable unavailable')
def proc(pid):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
  return {'pid':pid,'start_tick':int(parts[19]),'pgrp':int(parts[2]),'session':int(parts[3]),'state':parts[0]}
 except (OSError,ValueError,IndexError):return None
def group_members(group):
 members=[]
 for path in Path('/proc').iterdir():
  if path.name.isdigit():
   row=proc(int(path.name))
   if row and row['pgrp']==group and row['session']==group and row['state']!='Z':members.append(row)
 return members
def cleanup_group(group,grace=10):
 observed={r['pid']:r for r in group_members(group)}
 for sig in (signal.SIGTERM,signal.SIGKILL):
  for pid,identity in list(observed.items()):
   current=proc(pid)
   if current and current['start_tick']==identity['start_tick'] and current['pgrp']==group:
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
  until=time.monotonic()+(grace if sig==signal.SIGTERM else 2)
  while time.monotonic()<until:
   live=group_members(group)
   if not live:return {'owned':list(observed.values()),'remaining':[]}
   observed.update({r['pid']:r for r in live});time.sleep(.2)
 return {'owned':list(observed.values()),'remaining':group_members(group)}
def _guarded_run(argv,*,env,cwd,deadline,log,mem_probe=mem_available,floor=8*1024**3,phase=None,training_telemetry=None,heartbeat_interval=15):
 admission_remaining=deadline-time.time();admission_available=mem_probe()
 emit('phase_admission',phase=phase,phase_remaining_seconds=admission_remaining,linux_mem_available_bytes=admission_available,required_mem_available_bytes=floor)
 require(admission_remaining>0 and admission_available>=floor,'prelaunch deadline or Linux memory floor')
 with Path(log).open('wb') as output:
  child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  identity=proc(child.pid);reason=None;minimum=None;next_heartbeat=0
  try:
   while child.poll() is None:
    available=mem_probe();minimum=available if minimum is None else min(minimum,available)
    if training_telemetry is not None and time.monotonic()>=next_heartbeat:
     emit('training_heartbeat',phase=phase,child_identity=identity,child_rss_bytes=rss_bytes(identity),linux_mem_available_bytes=available,phase_remaining_seconds=max(0,deadline-time.time()),**telemetry_snapshot(training_telemetry))
     next_heartbeat=time.monotonic()+heartbeat_interval
    if STOP:reason='entry termination requested'
    elif time.time()>=deadline:reason='phase deadline'
    elif available<floor:reason='Linux MemAvailable below8GiB'
    elif Path(log).stat().st_size>64*1024**2:reason='phase log exceeded64MiB'
    if reason:break
    time.sleep(.25)
  finally:
   release=cleanup_group(child.pid)
   try:code=child.wait(timeout=2)
   except subprocess.TimeoutExpired:code=None
 result={'at':stamp(),'argv_program':Path(argv[0]).name,'pid_identity':identity,'exit_code':code,'reason':reason,'minimum_mem_available_bytes':minimum,'release':release}
 write(str(log)+'.guard.json',result)
 require(reason is None and code==0 and not release['remaining'],'guarded phase failed')
 return result
def guarded_run(argv,*,env,cwd,deadline,log,mem_probe=mem_available,floor=8*1024**3,phase=None,training_telemetry=None,heartbeat_interval=15):
 phase=phase or Path(log).stem
 emit('phase_start',phase=phase)
 try:
  result=_guarded_run(argv,env=env,cwd=cwd,deadline=deadline,log=log,mem_probe=mem_probe,floor=floor,phase=phase,training_telemetry=training_telemetry,heartbeat_interval=heartbeat_interval)
 except Exception as error:
  guard={}
  try:guard=json.loads(Path(str(log)+'.guard.json').read_text())
  except (OSError,ValueError):pass
  emit('phase_terminal',phase=phase,status='failed',reason=safe_reason(error),guard_reason=guard.get('reason'),exit_code=guard.get('exit_code'),setup_log_tail=setup_tail(log,phase))
  raise
 emit('phase_terminal',phase=phase,status='succeeded',exit_code=result['exit_code'])
 return result
def relocate_recipe(recipe,binding,run):
 r=json.loads(json.dumps(recipe));mapping={}
 for item in binding['payload_files']:
  path=bounded_path(HERE,item['relative_path']);require(path.is_file() and sha(path)==item['sha256'],'payload hash differs')
  require(item['original_path'] not in mapping,'duplicate relocation');mapping[item['original_path']]={'path':str(path),'sha256':item['sha256']}
 old_model=Path(r['model_path'])
 for name,digest in MODEL_PINS.items():mapping[str(old_model/name)]={'path':str(run/'model'/name),'sha256':digest}
 def record(old):
  require(old['path'] in mapping,'unbound input relocation');new=mapping[old['path']]
  require(new['sha256']==old['sha256'],'relocation changed content identity');return dict(new)
 r['inputs']=[record(x) for x in r['inputs']]
 for key in ('token_rows','draw_schedule','development_panel'):r[key]=record(r[key])
 r['model_path']=str(run/'model');r['output_dir']=str(run/'artifacts/training');r['archive_dir']=str(run/'artifacts/archive')
 p=r['parameters'];require(p['max_steps']==1000 and p['per_device_batch']==2 and p['gradient_accumulation']==8,'comparison geometry differs')
 require(r['identity']['schedule']==p and r['stage']=='task_sft_prm03_v1','task horizon identity differs')
 require(r['identity']['parent']['kind']=='midtrain_control' and r['identity']['parent']['weights_sha256']==MODEL_PINS['model.safetensors'] and r['identity']['parent']['revision']==REVISION,'fresh Midtrain control differs')
 require(r['identity']['policy']['initialization']=='new_lora_on_midtrain_control' and r.get('resume_from') is None,'control must initialize fresh')
 require(r['mandatory_stop_steps']==[250] and r['decision_steps']==[250],'first full250 stop differs')
 require(r['launch_authorized'] is True,'recipe launch admission pending')
 return r
def clean_env(run):
 env=dict(os.environ);env.pop('HF_TOKEN',None);env.pop('HUGGING_FACE_HUB_TOKEN',None)
 env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',UV_CACHE_DIR=str(run/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(run/'python'),HF_HOME=str(run/'hf-cache'))
 return env
def main():
 parser=argparse.ArgumentParser();parser.add_argument('binding',type=Path);args=parser.parse_args()
 emit('entry_start',phase='binding-preflight')
 b=json.loads(args.binding.read_text());hard=validate_binding(b)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_requested)
 run=Path(tempfile.mkdtemp(prefix='sepalith-r2-'+b['run_id']+'-',dir='/tmp'));art=run/'artifacts';art.mkdir();write(art/'binding.json',b)
 env=clean_env(run);success=False;failure=None;python=None;upload_success=False;phase='image-preflight'
 def command(argv,name,deadline=None,command_env=env):
  nonlocal phase
  phase=name
  return guarded_run(argv,env=command_env,cwd=HERE,deadline=setup if deadline is None else deadline,log=art/(name+'.log'),phase=name,training_telemetry=art/'training/telemetry.jsonl' if name=='training' else None)
 try:
  emit('phase_start',phase=phase)
  setup=remaining_phase(hard,1800,1260)
  emit('image_assumptions',python_version=list(sys.version_info[:3]),machine=platform.machine(),cc_available=shutil.which('cc') is not None,timeout_file_present=Path('/usr/bin/timeout').is_file(),pip_module_present=importlib.util.find_spec('pip') is not None)
  require(shutil.which('cc') is not None and Path('/usr/bin/timeout').is_file(),'image compiler or GNU timeout missing')
  source_manifest=json.loads((HERE/'trainer-source-manifest.json').read_text())
  require(sha(HERE/'trainer-source-manifest.json')==b['trainer_source_manifest_sha256'],'trainer manifest pin differs')
  for row in source_manifest['files']:
   q=bounded_path(HERE,row['path']);require(sha(q)==row['sha256'],'trainer source differs')
  emit('phase_terminal',phase=phase,status='succeeded')
  command([sys.executable,'-m','pip','--disable-pip-version-check','--no-input','install','--target',str(run/'bootstrap-tools'),'uv==0.11.23'],'uv-bootstrap')
  uv=run/'bootstrap-tools/bin/uv';emit('uv_layout_check',expected_relative_path='bootstrap-tools/bin/uv',executable_exists=uv.is_file());require(uv.is_file(),'pinned uv executable missing')
  command([str(uv),'python','install','3.10.19'],'python-bootstrap')
  interpreters=list((run/'python').glob('*/bin/python3.10'));require(len(interpreters)==1,'managed Python3.10 missing or ambiguous')
  command([str(uv),'venv','--python',str(interpreters[0]),str(run/'venv')],'venv-bootstrap')
  python=run/'venv/bin/python'
  command([str(uv),'pip','install','--python',str(python),'-r',str(HERE/'requirements.txt')],'packages-bootstrap')
  command([str(uv),'pip','check','--python',str(python)],'packages-check')
  command([str(python),str(HERE/'runtime_setup.py'),str(run)],'runtime-setup')
  raw_recipe=bounded_path(HERE,b['recipe']['relative_path']);require(sha(raw_recipe)==b['recipe']['sha256'],'recipe pin differs')
  recipe=relocate_recipe(json.loads(raw_recipe.read_text()),b,run)
  upload_env=dict(env)
  if 'HF_TOKEN' in os.environ:upload_env['HF_TOKEN']=os.environ['HF_TOKEN']
  command([str(python),str(HERE/'artifact_upload.py'),'sentinel',str(run)],'sentinel',command_env=upload_env)
  train_deadline=remaining_phase(hard,4200,660);require(train_deadline-time.time()>recipe['checkpoint_reserve_seconds']+30,'insufficient guarded training time')
  recipe['deadline']=datetime.datetime.fromtimestamp(train_deadline,datetime.timezone.utc).isoformat();recipe['max_attempt_seconds']=train_deadline-time.time();recipe['termination_grace_seconds']=30
  write(art/'recipe.json',recipe)
  env.update(PYTHONPATH=str(HERE/'source/experiments/training')+os.pathsep+str(HERE/'source/packages/sepalith/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  command([str(python),str(HERE/'cloud_launch.py'),str(art/'recipe.json'),'--receipt',str(art/'supervision.json')],'training',deadline=train_deadline+1)
  terminal=json.loads((art/'training/terminal.json').read_text());require(terminal['step']==250 and terminal['status']=='lead_decision','full250 task stop not reached')
  profile=json.loads((art/'training/cloud-backward-profile.json').read_text());require(profile['status']=='pass' and profile['optimizer_updates']==0,'backward profile missing')
  success=True
 except Exception as error:
  failure=type(error).__name__;emit('entry_failure',phase=phase,error_type=failure,reason=safe_reason(error))
  if phase=='image-preflight':emit('phase_terminal',phase=phase,status='failed',reason=safe_reason(error))
  write(art/'entry-failure.json',{'at':stamp(),'error_type':failure,'status':'failed; no success admission','phase':phase,'reason':safe_reason(error),'details':'See bounded provider phase events and local guard/training logs.'})
 finally:
  write(art/'local-terminal.json',{'at':stamp(),'training_success':success,'error_type':failure,'upload_success':False})
  if python is not None and python.is_file() and time.time()<hard-60:
   upload_env=dict(clean_env(run))
   if 'HF_TOKEN' in os.environ:upload_env['HF_TOKEN']=os.environ['HF_TOKEN']
   try:
    deadline=remaining_phase(hard,600,60)
    guarded_run([str(python),str(HERE/'artifact_upload.py'),'final',str(run)],env=upload_env,cwd=HERE,deadline=deadline,log=run/'upload.log',phase='artifact-upload')
    uploaded=json.loads((run/'persistence-receipt.json').read_text());upload_success=True
    print(json.dumps({'run_id':b['run_id'],'training_success':success,'persistence':uploaded}),flush=True)
   except Exception as error:
    success=False;print(json.dumps({'run_id':b['run_id'],'status':'upload_failed','error_type':type(error).__name__}),flush=True)
  else:
   success=False
   emit('upload_skipped',reason='managed_python_unavailable' if python is None or not python.is_file() else 'insufficient_deadline')
  emit('entry_terminal',training_success=success,upload_success=upload_success,status='succeeded' if success and upload_success else 'failed')
 if not success:raise SystemExit(1)
if __name__=='__main__':
 try:main()
 except Exception as error:
  emit('entry_rejected',phase='binding-or-entry-preflight',error_type=type(error).__name__,reason=safe_reason(error));emit('entry_terminal',training_success=False,upload_success=False,status='failed');raise SystemExit(1)
