import pathlib,json,subprocess,hashlib,datetime,os
R=pathlib.Path(__file__).resolve().parent;A=json.loads((R/'admission.json').read_text());P=R.parent/'r2-noop4100-64k-preparation-v1'
for x in A['frozen_files']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
command=['timeout','--signal=TERM','--kill-after=30s','3660',*json.loads((P/'commands.json').read_text())['render_after_root_admission_only']]
with(R/'process.log').open('x')as f:
 p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));write('launch.json',{'controller_pid':os.getpid(),'timeout_pid':p.pid,'command':command});code=p.wait()
write('terminal.json',{'exit_code':code,'status':'command_complete_requires_root_review' if code==0 else 'failed_preserve_partial','training_admitted':False});raise SystemExit(code)
