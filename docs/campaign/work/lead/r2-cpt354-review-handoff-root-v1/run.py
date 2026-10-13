import pathlib,json,hashlib,subprocess,datetime,time,os
R=pathlib.Path(__file__).resolve().parent;P=R.parent/'r2-cpt354-recovery-review-preparation-v2';PLAN=R.parents[4]
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
pids=[2630015,2630027,2630218,2634519];write('launch.json',{'at':at(),'pid':os.getpid(),'waiting_for':pids,'purpose':'verify full354 after training terminal, then matched8K16Keval under soleCUDAguard','training_continuation_authorized':False})
phase='waiting_training_terminal'
try:
 until=time.monotonic()+7200
 while any(pathlib.Path(f'/proc/{pid}').exists()for pid in pids):
  if time.monotonic()>until:raise TimeoutError('training wait deadline')
  time.sleep(3)
 assert sha(P/'artifact-manifest.json')=='ad122479ede0225a4443dd28cfe49aec26d5105e2795573d7e21d5eb6794152c'
 for x in read(P/'artifact-manifest.json')['files']:
  f=P/x['path'];assert f.stat().st_size==x['bytes']and sha(f)==x['sha256']
 handles=read(P/'runtime-handles.template.json');assert read(pathlib.Path(handles['controller_terminal']))['exit_code']==0 and read(pathlib.Path(handles['guard_terminal']))['child_exit_code']==0
 write('runtime-handles.json',handles);phase='verifying_terminal_payloads'
 cmd=['timeout','--signal=TERM','--kill-after=30s','1800','taskset','-c','12,14','env','CUDA_VISIBLE_DEVICES=','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1','python3','-B',str(P/'prepare.py'),'--runtime-handles',str(R/'runtime-handles.json')]
 with(R/'verify.log').open('x')as f:code=subprocess.run(cmd,cwd=PLAN,stdout=f,stderr=subprocess.STDOUT).returncode
 write('verify-terminal.json',{'at':at(),'exit_code':code});assert code==0,'terminal verification failed'
 review=read(P/'checkpoint-review.json');assert review['updates_verified']==24 and review['draws_verified']==384 and review['native_and_durable_payloads_verified']is True
 binding=read(P/'binding.review.json');binding['status']='admitted'
 with(P/'binding.admitted.json').open('x')as f:json.dump(binding,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 cp=pathlib.Path(binding['model_path']);m=read(cp/'campaign-manifest.json');released=[]
 for root in (cp,pathlib.Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-selected330-recovery-cadence24-v2/full/checkpoint-354')):
  for n in ('model.safetensors','optimizer.pt'):
   f=root/n;s=f.stat();assert s.st_size==m['files'][n]['bytes'];fd=os.open(f,os.O_RDONLY|os.O_NOFOLLOW)
   try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
   t=f.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns);released.append(str(f))
 write('verified-checkpoint-cache-release.json',{'at':at(),'files':released,'content_unchanged':True})
 phase='evaluating_long_panels'
 command=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(R.parent/'host-memory-guard-v4/cuda_host_guard.py'),'--command-json',str(P/'command.json'),'--output','/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt354-long-eval-root-v1-host-supervision','--seconds','2400','--minimum-free-mib','6144','--admission-free-mib','12288']
 write('eval-guard-command.json',command)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',UNSLOTH_RETURN_LOGITS='0')
 with(R/'eval-guard.log').open('x')as f:
  child=subprocess.Popen(command,cwd=PLAN,env=env,stdout=f,stderr=subprocess.STDOUT);write('eval-launch.json',{'at':at(),'guard_pid':child.pid});code=child.wait()
 assert code==0,'long evaluation guard failed'
 write('terminal.json',{'at':at(),'status':'commands_complete_requires_root_metric_review','exit_code':0,'promotion_authorized':False})
except BaseException as e:
 write('terminal.json',{'at':at(),'status':'stopped_without_retry','phase':phase,'error':type(e).__name__+': '+str(e)});raise
