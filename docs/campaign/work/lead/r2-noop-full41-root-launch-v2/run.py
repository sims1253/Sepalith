import datetime,hashlib,json,os,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];S=L/'r2-sourcewalk-noop-expansion-preparation-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):
 with (P/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert sha(S/'source-manifest.json')=='b7b03e63f533c8035d896352dc1416d4304b56744d8683a53b2b443fb734a842'
for x in read(S/'source-manifest.json')['files']:
 f=S/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256']
command=['timeout','--signal=TERM','--kill-after=30s','1800','ionice','-c3','nice','-n','10','taskset','-c','4,6']+read(S/'commands.json')['prepare_full_41_shard_scope']
write('launch.json',{'at':at(),'controller_pid':os.getpid(),'command':command,'scope':'Full41 no-op target-free inputs only; no provider or training admission','tests_passed':13})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
with (P/'process.log').open('x') as log:
 proc=subprocess.Popen(command,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('process.json',{'at':at(),'pid':proc.pid});code=proc.wait()
result={'at':at(),'exit_code':code,'status':'command_complete_requires_root_geometry_output_review' if code==0 else 'stopped_without_retry','training_admission':False}
write('terminal.json',result)
if code:raise SystemExit(code)
