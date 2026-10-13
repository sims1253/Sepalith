import pathlib,json,hashlib,subprocess,datetime,os
R=pathlib.Path(__file__).resolve().parent
A=json.loads((R/'admission.json').read_text())
for x in A['frozen_files']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
command=['timeout','--signal=TERM','--kill-after=30s','3600','bash',str(R/'run_lanes.sh'),'/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/fallback32-postrender-v1','/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/render32-postrender-v2']
with(R/'render32.log').open('x')as f:
 p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));write('launch.json',{'controller_pid':os.getpid(),'child_pid':p.pid,'command':command});code=p.wait()
write('terminal.json',{'exit_code':code,'status':'command_complete_requires_root_review' if code==0 else 'failed_preserve_partial','training_admitted':False})
raise SystemExit(code)
