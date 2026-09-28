import pathlib,json,os,subprocess,datetime,time,shutil
R=pathlib.Path(__file__).resolve().parent;B=R.parent/'r2-selected330-recovery-root-v2';PLAN=R.parents[4]
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,v):
 with(R/name).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def read(p):return json.loads(p.read_text())
def advise(files,label):
 entries=[]
 for p,size in files:
  s=p.stat();assert p.is_file()and not p.is_symlink()and s.st_size==size
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  t=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns);entries.append({'path':str(p),'bytes':size})
 write(label+'.json',{'at':at(),'operation':'targeted_clean_cache_advice','files':entries,'content_unchanged':True})
def cp_files(cp):
 m=read(cp/'campaign-manifest.json');return[(cp/n,m['files'][n]['bytes'])for n in ('model.safetensors','optimizer.pt')]
assert read(B/'cpu-terminal.json')['exit_code']==0
recipe=read(B/'runtime-recipe.json');cp=pathlib.Path(read(B/'continuation-330.admitted.json')['checkpoint']);archive=pathlib.Path(recipe['outputs']['archive']);native=pathlib.Path(recipe['outputs']['trainer']);assert archive.is_dir()and native.is_dir()
stage=read(pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json'));data=[(pathlib.Path(o['staged_path'])/n,x['bytes'])for o in stage['objects']for n,x in o['files'].items()]
free=shutil.disk_usage('/home').free;assert free-recipe['checkpoint_storage']['expected_full_checkpoint_bytes']>=70*1024**3
advise(cp_files(cp)+data,'pre_guard_cache_release')
g=read(B/'guard-command.json');assert not pathlib.Path(g[g.index('--output')+1]).exists()
with(R/'guard.log').open('x')as log:
 child=subprocess.Popen(g,cwd=PLAN,stdout=log,stderr=subprocess.STDOUT);write('launch.json',{'at':at(),'controller_pid':os.getpid(),'guard_pid':child.pid,'command':g,'recipe':str(B/'runtime-recipe.json'),'resume_step':330,'stop_at':354,'native_free_bytes':free})
 seen=set()
 while child.poll()is None:
  events=[]
  for p in (archive/'startup-telemetry.jsonl',archive/'telemetry.jsonl'):
   if p.exists():
    for line in p.read_text().splitlines():
     try:events.append(json.loads(line))
     except json.JSONDecodeError:pass
  if any(x.get('event')=='dataset_preflight_end'for x in events)and'data'not in seen:advise(data,'post_dataset_preflight_cache_release');seen.add('data')
  if any(x.get('event')=='pre_optimizer'for x in events)and'loaded'not in seen:advise(cp_files(cp),'post_optimizer_load_cache_release');seen.add('loaded')
  if any(x.get('event')=='durable_E_publish_end'for x in events)and'published'not in seen:
   advise(cp_files(native/'checkpoint-354')+cp_files(archive/'full/checkpoint-354'),'post_publish_cache_release');seen.add('published')
  time.sleep(2)
 code=child.wait()
write('terminal.json',{'at':at(),'exit_code':code,'status':'commands_complete_requires_root_review'if code==0 else'failed_requires_root_review','checkpoint_accepted':False})
raise SystemExit(code)
