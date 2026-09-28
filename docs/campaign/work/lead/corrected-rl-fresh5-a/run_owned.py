import datetime, hashlib, json, os, pathlib, signal, subprocess, time
W=pathlib.Path(__file__).parent
P=W.parents[2];N=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(name,d):
 with (W/name).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
a=json.loads((P/'receipts/RL-08-corrected-fresh5-root-admission.json').read_text())
assert a['status']=='admitted_bounded_fresh5'
assert sha(W/'recipe.json')==a['recipe_sha256'] and sha(W/'command.json')==a['command_sha256']
assert sha(pathlib.Path(__file__))==a['supervisor_sha256']
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,9,13,21,20,tzinfo=datetime.timezone.utc)
guard=P/'work/lead/host_memory_guard_v3.py';assert sha(guard)==a['guard_sha256']
argv=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',str(guard),'--command-json',str(W/'command.json'),'--output',str(N/'training/RL-corrected-theta0-fresh5-v1-host-supervision'),'--seconds','1860','--minimum-free-mib','8192','--release-cache-file',str(N/'models/SFT-primary-step1000-theta0/model.safetensors')]
start=time.monotonic();put('supervisor-launch.json',{'at':now(),'pid':os.getpid(),'start_tick':pathlib.Path('/proc/self/stat').read_text().split()[21],'task':'RL-08','owner':'lead','argv':argv})
with (W/'guard-console.log').open('xb') as log:
 p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
 put('guard-process.json',{'at':now(),'pid':p.pid,'start_tick':pathlib.Path(f'/proc/{p.pid}/stat').read_text().split()[21]})
 signal.signal(signal.SIGTERM,lambda *_:os.kill(p.pid,signal.SIGTERM) if p.poll() is None else None)
 try:code=p.wait(timeout=1900)
 except subprocess.TimeoutExpired:
  os.kill(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=40)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
 put('supervisor-terminal.json',{'at':now(),'seconds':time.monotonic()-start,'guard_exit_code':code,'scientific_acceptance':'pending root checkpoint, DEV and resource review'})
