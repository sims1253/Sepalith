import pathlib,json,time,subprocess,hashlib,datetime,os
R=pathlib.Path(__file__).resolve().parent;P=R.parent/'r2-cpt706-from450-review-preparation-v1';A=json.loads((R/'admission.json').read_text())
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
write('launch.json',{'controller_pid':os.getpid(),'training_controller_pid':3038867,'maximum_wait_seconds':10800,'verification_timeout_seconds':1800,'cpu_cores':[12,14]})
t=R.parent/'r2-cpt450-to706-cadence64-root-v3/terminal.json';deadline=time.monotonic()+10800
while not t.exists():
 if time.monotonic()>deadline:raise SystemExit('training terminal observation timeout; no restart')
 time.sleep(5)
if json.loads(t.read_text())['exit_code']!=0:
 write('terminal.json',{'exit_code':1,'status':'training_failed_verification_not_launched'});raise SystemExit(1)
while any(pathlib.Path('/proc/'+str(pid)).exists()for pid in(3038867,3040840,3042470,3044342)):
 if time.monotonic()>deadline:raise SystemExit('runtime handles remain live; no verifier launched')
 time.sleep(2)
assert hashlib.sha256((P/'artifact-manifest.json').read_bytes()).hexdigest()==A['artifact_manifest_sha256']
for x in json.loads((P/'artifact-manifest.json').read_text())['files']:assert hashlib.sha256((P/x['path']).read_bytes()).hexdigest()==x['sha256']
command=['timeout','--signal=TERM','--kill-after=30s','1800','taskset','-c','12,14','nice','-n','10',*json.loads((P/'root-commands.json').read_text())['prepare_manual_after_terminal']]
with(R/'verification.log').open('x')as f:
 c=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'));write('verification-launch.json',{'pid':c.pid,'command':command});code=c.wait()
write('terminal.json',{'exit_code':code,'status':'fullstate_verification_complete_requires_root_review' if code==0 else 'verification_failed','cuda_launched':False,'checkpoint_accepted':False});raise SystemExit(code)
