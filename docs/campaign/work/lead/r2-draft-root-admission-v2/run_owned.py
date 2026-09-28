"""Root-owned draft profile lock and host supervision. Admission is separate."""
import pathlib,json,hashlib,fcntl,subprocess,os,datetime,time,signal
W=pathlib.Path(__file__).resolve().parent
N=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,value):
 with (W/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
a=json.loads((W/'launch-admission.json').read_text())
assert a['status']=='admitted_bounded_local_draft_profile'
for path,h in a['pins'].items():assert sha(pathlib.Path(path))==h,path
for row in json.loads(pathlib.Path(a['dependency_manifest']).read_text())['files']:assert sha(pathlib.Path(row['path']))==row['sha256'],row['path']
source=pathlib.Path(a['source_root'])
for row in json.loads(pathlib.Path(a['inventory']).read_text())['files']:assert sha(source/row['path'])==row['sha256'],row['path']
with (N/'resource-locks/cuda0.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 ps='$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory;[pscustomobject]@{AvailableMBytes=$m.AvailableMBytes;PagesOutputPersec=$m.PagesOutputPersec}|ConvertTo-Json'
 host=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',ps],text=True,timeout=15))
 assert host['AvailableMBytes']>=24576 and host['PagesOutputPersec']==0,host
 now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime.fromisoformat(a['hard_deadline'])
 seconds=min(1700,int((deadline-now).total_seconds()));assert seconds>=a['minimum_remaining_seconds']
 assert not pathlib.Path(a['run_dir']).exists()
 argv=['python3','-B',a['guard'],'--command-json',str(W/'command.json'),'--output',a['guard_output'],'--seconds',str(seconds),'--minimum-free-mib','8192']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',SEPALITH_PROFILE_ADMITTED='1',SEPALITH_PUBLIC_DRAFT_WEIGHTS=str(N/'models/released-dspark-hf-v1/model.safetensors'))
 started=time.monotonic()
 with (W/'guard-console.log').open('xb') as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),),env=env)
  def stop(*_):
   if child.poll() is None:child.send_signal(signal.SIGTERM)
  for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,stop)
  put('supervisor-launch.json',{'at':now.isoformat(),'pid':os.getpid(),'guard_pid':child.pid,'host':host,'argv':argv,'lock_held':True,'seconds':seconds})
  code=child.wait();put('supervisor-terminal.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'seconds':time.monotonic()-started,'acceptance':'requires independent stage/checkpoint/descendant verification'})
 raise SystemExit(code)
