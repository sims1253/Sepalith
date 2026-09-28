import datetime,hashlib,json,os,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];B=L/'r2-edit-context322-root-preparation-v1';S=L/'r2-full-weight-edit-context-profile-preparation-v2';CP=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def release():
 assert sha(CP/'campaign-manifest.json')=='81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf';m=read(CP/'campaign-manifest.json');out=[]
 for name in ('model.safetensors','optimizer.pt'):
  p=CP/name;s=p.stat();assert s.st_size==m['files'][name]['bytes'];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns);out.append({'path':str(p),'bytes':s.st_size,'metadata_unchanged':True})
 return out
assert read(B/'terminal.json')['failed_cap']==4096 and read(B/'terminal.json')['completed_caps']==[2048]
old=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-edit-context322-4096-root-v1-host-supervision');assert read(old/'terminal.json')=={'status':'not_launched','reason':'host_free_memory_below_admission_floor'} and not (old/'launch.json').exists()
assert sha(S/'source-manifest.json')=='609c599d144f1afd3887f2921e3723cea124696d85f2a61c9526e2b178979f38'
for x in read(S/'source-manifest.json')['files']:
 q=S/x['path'];assert q.stat().st_size==x['bytes'] and sha(q)==x['sha256']
write(P/'launch.json',{'at':at(),'controller_pid':os.getpid(),'caps':[4096,8192,16384,32768],'scope':'ascending no-update full-state memory probes, with exact clean checkpoint cache release before each unchanged admission','previous_2048_pass_preserved':True,'previous_4096_not_launched_preserved':str(old)})
completed=[2048];cap=None
try:
 for cap in (4096,8192,16384,32768):
  A=B/str(cap);O=P/str(cap);O.mkdir();write(O/'clean-cache-release.json',{'at':at(),'files':release(),'operation':'POSIX_FADV_DONTNEED only exact immutable checkpoint model and optimizer files'})
  ad=read(A/'admission.json')
  if ad['status']!='admitted':
   assert ad['launch_authorized'] is False;write(O/'admission-before-launch.json',ad);ad.update(status='admitted',launch_authorized=True);(A/'admission.json').write_text(json.dumps(ad,indent=2)+'\n')
  assert ad['launch_authorized'] is True and not (A/'report.json').exists()
  g=read(A/'guard-command.json');g[g.index('--output')+1]+='-b';write(O/'guard-command.json',g);env=dict(os.environ);env.update(read(A/'environment.json'))
  with (O/'guard.log').open('x') as log:
   child=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write(O/'guard-process.json',{'at':at(),'pid':child.pid});code=child.wait()
  assert code==0,f'cap{cap} guard exit {code}'
  t=read(pathlib.Path(g[g.index('--output')+1])/'terminal.json');assert t['status']=='completed' and t['child_exit_code']==0
  r=read(A/'report.json');assert r['status']=='completed_no_optimizer_update' and r['optimizer_step_called'] is False and r['finite_gradient_tensors']==r['nonzero_gradient_tensors']==381 and r['sampled_parameters_unchanged'] is True and r['rng_restored']==r['rng_before'] and r['memory']['free_after']>0
  completed.append(cap)
 write(P/'terminal.json',{'at':at(),'status':'all_caps_completed_requires_root_resource_review','completed_caps':completed,'training_admitted':False})
except Exception as e:
 write(P/'terminal.json',{'at':at(),'status':'stopped_escalation_after_failure','failed_cap':cap,'completed_caps':completed,'error':str(e),'training_admitted':False});raise
