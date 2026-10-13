import pathlib,json,subprocess,time,hashlib,datetime,os
R=pathlib.Path(__file__).resolve().parent;L=R.parent;A=json.loads((R/'admission.json').read_text())
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n')
write('launch.json',{'controller_pid':os.getpid(),'waiting_for_controller':2726253,'maximum_wait_seconds':1200})
t=L/'r2-semantic9534-rich-retry-root-v1/terminal.json';deadline=time.monotonic()+1200
while not t.exists():
 if time.monotonic()>deadline:raise SystemExit('producer terminal observation timeout; no restart')
 time.sleep(3)
assert json.loads(t.read_text())['exit_code']==0
for x in A['frozen_files']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
p=L/'r2-semantic9534-postrender-preparation-v2';base='/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/'
command=['timeout','180','taskset','-c','12','nice','-n','10','ionice','-c','3','python3','-B',str(p/'source/verify_render16.py'),'--input-manifest',base+'render-inputs-v1/manifest.json','--render16',base+'render-16k-retry-rich-v1','--output',str(R/'render16-terminal-merge.json')]
with(R/'verification.log').open('x')as f:c=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')).returncode
write('terminal.json',{'exit_code':c,'status':'verification_complete_requires_root_review' if c==0 else 'verification_failed','training_admitted':False});raise SystemExit(c)
