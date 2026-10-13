import pathlib,json,hashlib,subprocess,datetime,os,concurrent.futures
R=pathlib.Path(__file__).resolve().parent;Q=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-noop4100-provider-v1/fallback128-root-v1');O=Q.parent/'render128-root-v1';A=json.loads((R/'admission.json').read_text());M=json.loads((Q/'manifest.json').read_text())
for x in A['frozen_files']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
for x in M['entries']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
O.mkdir()
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n')
write('launch.json',{'controller_pid':os.getpid(),'cores':[4,6],'rows':9,'maximum_seconds_each_shard':300})
def lane(core):
 with(R/f'lane-{core}.log').open('x')as f:
  for x in M['entries']:
   if x['core']==core:
    c=subprocess.run(['timeout','--signal=TERM','--kill-after=20s','300','bash',str(R.parent/'r2-noop4100-postrender-root-v2/run_shard.sh'),f"{x['shard']:04d}",str(core),'131072','2048',str(Q/'inputs'),str(O)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='')).returncode
    if c:return c
 return 0
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:codes=list(pool.map(lane,[4,6]))
code=0 if codes==[0,0] else 1;write('terminal.json',{'exit_code':code,'lane_codes':codes,'status':'commands_complete_requires_root_review' if code==0 else 'failed_preserve_partial','training_admitted':False});raise SystemExit(code)
