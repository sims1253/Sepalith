import datetime,json,os,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];N=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')
def read(p):return json.loads(p.read_text())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):
 with (P/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
old=N/'SFT11-varlen322-varlen_candidate-root-v1-host-supervision';ordinary=N/'SFT11-varlen322-ordinary_reference-root-v1-host-supervision';assert read(old/'terminal.json')=={'status':'not_launched','reason':'host_free_memory_below_admission_floor'};assert not (old/'launch.json').exists();t=read(ordinary/'terminal.json');assert t['status']=='completed' and t['child_exit_code']==0
recipe=read(L/'r2-native-varlen322-root-v1/varlen_candidate/recipe.json');assert not (pathlib.Path(recipe['outputs']['archive'])/'run-result.json').exists();assert not pathlib.Path(recipe['outputs']['trainer']).exists()
write('launch.json',{'at':at(),'pid':os.getpid(),'retry_reason':'Only previous admission failed; clean completed checkpoint cache evicted, Windows available24835MiB observed. Same packed recipe and thresholds.','ordinary_terminal':t,'production_admitted':False})
try:
 env=dict(os.environ);env.update(read(P/'environment.json'));g=read(P/'guard-command.json')
 with (P/'guard.log').open('x') as log:
  p=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('guard-process.json',{'at':at(),'pid':p.pid});code=p.wait()
 assert code==0,f'guard exit {code}'
 t=read(pathlib.Path(g[g.index('--output')+1])/'terminal.json');assert t['status']=='completed' and t['child_exit_code']==0
 write('terminal.json',{'at':at(),'status':'both_commands_completed_requires_root_payload_metric_review','promotion':False,'ordinary_guard':str(ordinary),'packed_guard':g[g.index('--output')+1],'original_admission_failure_preserved':str(old)})
except Exception as exc:
 write('terminal.json',{'at':at(),'status':'stopped_without_retry','error':str(exc)});raise
