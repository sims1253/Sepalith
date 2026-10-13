import datetime,hashlib,json,os,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];S=L/'r2-full-weight-edit-context-profile-preparation-v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert read(L/'r2-sm120-varlen-probe-root-v2/terminal.json')['exit_code']==0
assert sha(S/'source-manifest.json')=='609c599d144f1afd3887f2921e3723cea124696d85f2a61c9526e2b178979f38'
for x in read(S/'source-manifest.json')['files']:
 q=S/x['path'];assert q.stat().st_size==x['bytes'] and sha(q)==x['sha256']
write(P/'launch.json',{'at':at(),'controller_pid':os.getpid(),'caps':[2048,4096,8192,16384,32768],'scope':'ascending isolated full-state target-only memory probes; no optimizer updates; stop after first failure','source_checkpoint':322})
completed=[];cap=None
try:
 for cap in (2048,4096,8192,16384,32768):
  A=P/str(cap);ad=read(A/'admission.json');assert ad['status']=='prepared_requires_sequential_root_launch' and ad['launch_authorized'] is False
  write(A/'admission-before-root-launch.json',ad);ad.update(status='admitted',launch_authorized=True);(A/'admission.json').write_text(json.dumps(ad,indent=2)+'\n')
  g=read(A/'guard-command.json');env=dict(os.environ);env.update(read(A/'environment.json'))
  with (A/'guard.log').open('x') as log:
   child=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write(A/'guard-process.json',{'at':at(),'pid':child.pid});code=child.wait()
  assert code==0,f'cap{cap} guard exit {code}'
  terminal=read(pathlib.Path(g[g.index('--output')+1])/'terminal.json');assert terminal['status']=='completed' and terminal['child_exit_code']==0
  report=read(A/'report.json');assert report['status']=='completed_no_optimizer_update' and report['optimizer_step_called'] is False and report['finite_gradient_tensors']==report['nonzero_gradient_tensors']==381 and report['sampled_parameters_unchanged'] is True and report['rng_restored']==report['rng_before'] and report['memory']['free_after']>0
  completed.append(cap)
 write(P/'terminal.json',{'at':at(),'status':'all_caps_completed_requires_root_resource_review','completed_caps':completed,'training_admitted':False})
except Exception as exc:
 write(P/'terminal.json',{'at':at(),'status':'stopped_escalation_after_failure','failed_cap':cap,'completed_caps':completed,'error':str(exc),'training_admitted':False});raise
