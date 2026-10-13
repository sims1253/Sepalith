"""Bounded continuation of the verified production CPT lineage."""
import datetime, fcntl, hashlib, importlib.util, json, math, os, pathlib, shutil, signal, subprocess, sys, time, traceback
R=pathlib.Path(__file__).resolve().parent
LEAD=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
PYTHON='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
SOURCE=LEAD/'r2-cpt450-cadence64-continuation-preparation-v2/source/experiments/training'
GUARD=LEAD/'host-memory-guard-v4/cuda_host_guard.py'
BASE=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json')
RECIPE=R/'recipe.json'
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def status(state,**kw):
 v={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'supervisor_pid':os.getpid(),**kw};write(R/'status.json',v);print(json.dumps(v),flush=True)
def release(files):
 for p in files:
  p=pathlib.Path(p)
  if not p.is_file(): continue
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally: os.close(fd)
def cpfiles(cp): return [cp/'model.safetensors',cp/'optimizer.pt']
recipe=read(RECIPE);archive=pathlib.Path(recipe['outputs']['archive']);native=pathlib.Path(recipe['outputs']['trainer'])
env=dict(os.environ);env.update(read(R/'environment.json'))
stage=read(BASE);data=[pathlib.Path(o['staged_path'])/n for o in stage['objects'] for n in o['files']]
deadline=1790243068.8309097
stop_requested=False
def stop(*_):
 global stop_requested
 stop_requested=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def attested(cp,command):
 return [PYTHON,'-B',str(SOURCE/'native_launch_attestation.py'),'--base-receipt',str(BASE),'--base-receipt-sha256',sha(BASE),'--resume',str(cp),'--resume-manifest-sha256',sha(cp/'campaign-manifest.json'),'--',*command]
def guarded(command,d,seconds,cp,training=False):
 d.mkdir();write(d/'command.json',command)
 release(cpfiles(cp)+data)
 guard=[PYTHON,'-B',str(GUARD),'--command-json',str(d/'command.json'),'--output',str(d/'guard'),'--seconds',str(seconds),'--minimum-free-mib','8192','--admission-free-mib','16384']
 e=env.copy();e['SEPALITH_ALLOCATOR_REPORT']=str(d/'allocator.json');e['SEPALITH_RESUME_MANIFEST_SHA256']=sha(cp/'campaign-manifest.json')
 started=time.time();seen=set();offsets={};restoring=False;loaded=False
 with (d/'guard.log').open('x') as log:
  child=subprocess.Popen(guard,env=e,stdout=log,stderr=subprocess.STDOUT)
  write(d/'launch.json',{'pid':child.pid,'at':started,'command':guard})
  try:
   while child.poll() is None:
    if training:
     if time.time()>=deadline-1800 or stop_requested:
      write(recipe['outputs']['graceful_stop'],{'action':'save_and_stop','bound_recipe_sha256':sha(RECIPE)})
     for p in [archive/'startup-telemetry.jsonl',archive/'telemetry.jsonl']:
      if not p.exists():continue
      with p.open() as f:
       f.seek(offsets.get(str(p),0))
       while True:
        pos=f.tell();line=f.readline()
        if not line:break
        if not line.endswith('\n'):f.seek(pos);break
        event=json.loads(line)
        if event.get('at',0)<started:continue
        name=event.get('event');step=event.get('step')
        if name=='trainer_resume_start':restoring=True
        if name=='dataset_preflight_end':release(data)
        if name=='pre_optimizer' and not loaded:release(cpfiles(cp));loaded=True
        if name=='log' and 'loss' in event.get('logs',{}):
         logs=event['logs'];assert math.isfinite(logs['loss']) and math.isfinite(logs['grad_norm'])
         status('training',step=step,loss=logs['loss'],segment=str(d),deadline=deadline)
        if name in ('native_seal_end_publish_start','durable_E_publish_end'):
         key=(name,step)
         if key not in seen:
          release(cpfiles(native/f'checkpoint-{step}')+cpfiles(archive/f'full/checkpoint-{step}'));seen.add(key)
       offsets[str(p)]=f.tell()
     if restoring and not loaded:release(cpfiles(cp))
    elif stop_requested:child.terminate()
    # Linux headroom complements the guard's Windows host-memory checks.
    mem={x.split(':')[0]:int(x.split()[1]) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
    if mem['MemAvailable']<6*1024*1024:raise RuntimeError('Linux available RAM below 6 GiB')
    assert not cache_errors, f'Cache maintenance failed: {cache_errors}'
    time.sleep(2)
   code=child.wait()
  except BaseException:
   child.terminate()
   try:child.wait(timeout=40)
   except subprocess.TimeoutExpired:raise RuntimeError('Guard did not stop; manual process inspection required')
   raise
 assert code==0, f'guard failed: {d}'
 assert read(d/'guard/terminal.json')['status']=='completed'
 return d

def evaluate(cp,step,d):
 binding=read(R/'review706/binding.review.json');binding.update(status='admitted',model_path=str(cp),checkpoint_step=step)
 manifest=read(cp/'campaign-manifest.json')
 binding['model_files']={n:manifest['files'][n]['sha256'] for n in binding['model_files']}
 binding['runner_sha256']=sha(R/'review706/evaluate_matched.py');binding['dtype_restoration_sha256']=sha(R/'review706/saved_precision.py')
 bp=d.with_name(d.name+'-binding.json');write(bp,binding)
 out=pathlib.Path('/mnt/e/sepalith/campaign-20260915/evaluations')/('resume-20260921-'+str(step))
 command=[PYTHON,'-B',str(R/'review706/evaluate_matched.py'),'--binding',str(bp),'--output',str(out)]
 status('evaluating',step=step,output=str(out));guarded(command,d,1800,cp)
 spec=importlib.util.spec_from_file_location('review',R/'review706/review_eval.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=read(out/'result.json');assert result['status']=='evaluations_complete' and result['binding_sha256']==sha(bp)
 values={}
 for label in m.FIXTURES:
  panel=read(out/(label+'.json'));assert panel['step']==step
  # Reuse exact fixture/denominator/numerical validation; step is checked above.
  panel['step']=706;values[label]=m.validate_panel(panel,label,sha(bp))
 baseline=read(LEAD.parent.parent/'receipts/SFT-11-cpt450-root-matched-acceptance.json')['panels']
 assert all(values[k]<=baseline[k]['nll']*1.01 for k in values),f'Validation regressed >1% from accepted450: {values}'
 write(d/'accepted.json',{'step':step,'nll':values,'checkpoint_manifest_sha256':sha(cp/'campaign-manifest.json'),'decision':'continue_cpt_only','release_promoted':False})
 return values

def main():
 lock=open(R/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(R/'recovery-supervisor-launch.json',{'pid':os.getpid(),'start':time.time(),'deadline':deadline,'original_availability_hours':72,'recovery_from_step':4866,'policy':'same scientific identity; stop on any failure, >1% anchor regression, RAM guard, or deadline; no release promotion'})
 assert read(R/'review706/checkpoint-review.json')['native_and_durable_payloads_verified']
 assert read(R/'checkpoint-4866-recovered.json')['status']=='native_and_durable_payloads_verified'
 cp=native/'checkpoint-4866';step=4866
 evaluate(cp,step,R/'eval-4866')
 while time.time()<deadline-3*3600 and not stop_requested:
  target=min(step+512,11586)
  if target<=step:break
  d=R/f'recovery-train-{step}-{target}';d.mkdir()
  continuation={'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'admitted','decision':'continue','launch_authorized':True,'bound_recipe_sha256':sha(RECIPE),'checkpoint':str(cp),'checkpoint_manifest_sha256':sha(cp/'campaign-manifest.json'),'step':step}
  write(d/'continuation.json',continuation)
  write(d/'stop.json',{'schema':'sepalith.sft11.native-cpt-execution-stop.v1','status':'admitted','launch_authorized':True,'bound_recipe_sha256':sha(RECIPE),'resume_global_step':step,'stop_at_global_step':target})
  archive.mkdir(parents=True,exist_ok=True)
  args=['--recipe',str(RECIPE),'--resume',str(cp),'--continuation-admission',str(d/'continuation.json'),'--execution-stop-admission',str(d/'stop.json')]
  pre=attested(cp,[PYTHON,'-B',str(SOURCE/'full_weight_cpt_trainer.py'),'preflight-resume',*args])
  status('preflight',step=step,target=target)
  with (d/'preflight.log').open('x') as f:subprocess.run(pre,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=True)
  # One new full checkpoint plus 70 GiB free reserve, matching prior admission.
  assert shutil.disk_usage('/home').free-recipe['checkpoint_storage']['expected_full_checkpoint_bytes']>=70*1024**3
  command=attested(cp,[PYTHON,'-B',str(R/'run_with_allocator.py'),'run',*args])
  guarded(command,d/'execution',21600,cp,True)
  result=read(archive/'run-result.json');write(d/'run-result.json',result)
  assert result['global_step']>step and math.isfinite(result['train_loss'])
  expected_cursor=(result['global_step']-66)*16
  assert result['last_draw_position']+1==expected_cursor
  assert result['observed_draws']==(result['global_step']-step)*16
  step=result['global_step'];cp=native/f'checkpoint-{step}'
  assert read(cp/'campaign-state.json')['sampler']['cursor']==expected_cursor
  if stop_requested or time.time()>=deadline-1800:break
  evaluate(cp,step,R/f'eval-{step}')
  if pathlib.Path(recipe['outputs']['graceful_stop']).exists():break
 status('stopped_at_safe_boundary',step=step,checkpoint=str(cp),reason='time_limit_or_schedule_boundary_or_stop_request',deadline=deadline)
if __name__=='__main__':
 from cache_maintenance import start
 cache_stop, cache_worker, cache_errors = start(native, archive, data)
 try:main()
 except BaseException as error:
  status('stopped_on_failure',error=str(error),traceback=traceback.format_exc());raise
