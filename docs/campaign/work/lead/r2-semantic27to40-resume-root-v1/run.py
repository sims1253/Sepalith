import datetime,hashlib,json,os,pathlib,subprocess,importlib.util
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];S=L/'r2-semantic27to40-resume-preparation-v1';F=L/'r2-semantic-queue-root-launch-v3'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (P/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt=PLAN/'docs/campaign/receipts/DAT-10-semantic27to40-resume-preparation.json';assert sha(receipt)=='394d4fffa5e5fa7e851729326fe00acd8d16242041f46d3e21e391452c5af840'
for x in read(F/'source-manifest.json')['files']:
 p=F/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
spec=importlib.util.spec_from_file_location('semantic_resume',S/'validate_resume.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);v=m.validate();assert v['semantic']['completed_shards']==list(range(27,35)) and v['semantic']['missing_shards']==list(range(35,41));write('preflight.json',v)
command=v['resume']['command'];command[command.index('--shards')+1]=','.join(map(str,range(27,41)))
write('launch.json',{'at':at(),'controller_pid':os.getpid(),'command':command,'root_adjustment':'Request all14 original shards so final subset manifest accounts for all14. Frozen worker reusable() fully verifies existing8 and computes only missing6.','expected_reused':list(range(27,35)),'expected_computed':list(range(35,41)),'training_admission':False})
env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/parser-dependencies-root-v1')
with (P/'process.log').open('x') as log:
 proc=subprocess.Popen(command,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write('process.json',{'at':at(),'pid':proc.pid});code=proc.wait()
write('terminal.json',{'at':at(),'exit_code':code,'status':'commands_complete_requires_root_output_review' if code==0 else 'stopped_without_retry','training_admission':False})
if code:raise SystemExit(code)
