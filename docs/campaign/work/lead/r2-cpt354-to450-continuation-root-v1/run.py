import pathlib,json,os,subprocess,datetime,time,shutil,hashlib
R=pathlib.Path(__file__).resolve().parent;PLAN=R.parents[4]
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(n,v):
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def advise(files,label):
 rows=[]
 for p,size in files:
  s=p.stat();assert p.is_file()and not p.is_symlink()and s.st_size==size;fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  t=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns);rows.append(str(p))
 write(label+'.json',{'at':at(),'operation':'targeted_clean_cache_advice','files':rows,'content_unchanged':True})
def cpfiles(cp):
 m=read(cp/'campaign-manifest.json');return[(cp/n,m['files'][n]['bytes'])for n in ('model.safetensors','optimizer.pt')]
assert read(PLAN/'docs/campaign/receipts/SFT-11-cpt354-root-decision.json')['status']=='checkpoint354_accepted_for_continuation_not_release'
assert read(R.parent/'r2-cpt354-review-handoff-root-v1/terminal.json')['exit_code']==0
recipe=read(R/'runtime-recipe.json');assert hashlib.sha256((R/'runtime-recipe.json').read_bytes()).hexdigest()=='26b78532f907ca5e3a34827857ad58a47e0d56a167fb26a61dbf9b555b50a1d5'
archive=pathlib.Path(recipe['outputs']['archive']);native=pathlib.Path(recipe['outputs']['trainer']);cp=pathlib.Path(read(R/'continuation-354.json')['checkpoint']);stage=read(pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json'));data=[(pathlib.Path(o['staged_path'])/n,x['bytes'])for o in stage['objects']for n,x in o['files'].items()]
env=dict(os.environ);env.update(read(R.parent/'r2-selected330-ordinary-root-v1/environment.json'));env['SEPALITH_ALLOCATOR_REPORT']=str(R/'allocator.json');write('environment.json',{k:env[k]for k in read(R.parent/'r2-selected330-ordinary-root-v1/environment.json')})
command=['timeout','--signal=TERM','--kill-after=30s','900','taskset','-c','12,14',*read(R/'cpu-command.json')]
with(R/'cpu-preflight.log').open('x')as log:
 child=subprocess.Popen(command,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('cpu-launch.json',{'at':at(),'pid':child.pid,'controller_pid':os.getpid()});code=child.wait()
write('cpu-terminal.json',{'at':at(),'exit_code':code});assert code==0,'CPU preflight failed'
d=readline=None
for line in (R/'cpu-preflight.log').read_text().splitlines()[::-1]:
 try:d=json.loads(line);break
 except json.JSONDecodeError:pass
assert d['status']=='pass'and d['resume_step']==354 and d['initial_cursor']==4608 and d['cohort']['cohort_rows']==185318 and d['execution_stop_admission']['stop_at_global_step']==450
advise(cpfiles(cp)+data,'pre_guard_cache_release')
free=shutil.disk_usage('/home').free;assert free-recipe['checkpoint_storage']['expected_full_checkpoint_bytes']>=70*1024**3
g=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(R.parent/'host-memory-guard-v4/cuda_host_guard.py'),'--command-json',str(R/'training-command.json'),'--output','/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-recovery354-to450-cadence24-v1-host-supervision','--seconds','10800','--minimum-free-mib','6144','--admission-free-mib','14336'];write('guard-command.json',g)
with(R/'guard.log').open('x')as log:
 child=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('launch.json',{'at':at(),'controller_pid':os.getpid(),'guard_pid':child.pid,'command':g,'resume_step':354,'stop_at':450,'native_free_bytes':free});seen=set()
 while child.poll()is None:
  events=[]
  for p in (archive/'startup-telemetry.jsonl',archive/'telemetry.jsonl'):
   if p.exists():
    for line in p.read_text().splitlines():
     try:events.append(json.loads(line))
     except json.JSONDecodeError:pass
  for event in events:
   name=event.get('event');step=event.get('step')
   if name=='dataset_preflight_end'and'data'not in seen:advise(data,'post_dataset_preflight_cache_release');seen.add('data')
   if name=='pre_optimizer'and'loaded'not in seen:advise(cpfiles(cp),'post_optimizer_load_cache_release');seen.add('loaded')
   if name=='native_seal_end_publish_start'and('sealed',step)not in seen:advise(cpfiles(native/f'checkpoint-{step}'),f'post_native_seal_{step}_cache_release');seen.add(('sealed',step))
   if name=='durable_E_publish_end'and('published',step)not in seen:advise(cpfiles(native/f'checkpoint-{step}')+cpfiles(archive/f'full/checkpoint-{step}'),f'post_publish_{step}_cache_release');seen.add(('published',step))
  time.sleep(2)
 code=child.wait()
write('terminal.json',{'at':at(),'exit_code':code,'status':'commands_complete_requires_root_review'if code==0 else'failed_requires_root_review','checkpoint_accepted':False})
raise SystemExit(code)
