import pathlib,json,hashlib,fcntl,subprocess,os,datetime,time
W=pathlib.Path(__file__).resolve().parent
N=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
P=W.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,d):
 with (W/name).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
a=json.loads((W/'launch-admission.json').read_text())
assert a['status']=='admitted_bounded_exact_gate'
for name,h in a['pins'].items():assert sha(W/name)==h,name
for f in json.loads((W/'source-manifest.json').read_text())['files']:assert sha(pathlib.Path(a['source_root'])/f['path'])==f['sha256'],f['path']
lock=(N/'resource-locks/cuda0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
ps='$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory;[pscustomobject]@{AvailableMBytes=$m.AvailableMBytes;PagesOutputPersec=$m.PagesOutputPersec}|ConvertTo-Json'
m=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',ps],text=True,timeout=15));assert m['AvailableMBytes']>=24576 and m['PagesOutputPersec']==0,m
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,9,14,3,55,tzinfo=datetime.timezone.utc)
guard=P/'work/lead/host_memory_guard_v3.py';assert sha(guard)==a['guard_sha256']
argv=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(guard),'--command-json',str(W/'command.json'),'--output',str(N/'training/RL-R2-step500-exact-tenstep-v1-host-supervision'),'--seconds','2520','--minimum-free-mib','8192','--release-cache-file',str(N/'models/SFT11-task-global-b-500-merged/model.safetensors')]
start=time.monotonic()
with (W/'guard-console.log').open('xb') as log:
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),),env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
 put('supervisor-launch.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'start_tick':pathlib.Path('/proc/self/stat').read_text().split()[21],'guard_pid':child.pid,'guard_start_tick':pathlib.Path(f'/proc/{child.pid}/stat').read_text().split()[21],'host':m,'argv':argv,'lock_held':True})
 code=child.wait()
 put('supervisor-terminal.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'seconds':time.monotonic()-start,'scientific_acceptance':'pending checkpoint rollout and DEV review'})
