import datetime,hashlib,json,os,pathlib,subprocess,shutil
R=pathlib.Path(__file__).resolve().parent;B=R.parent/'r2-selected330-ordinary-root-v1';PLAN=R.parents[4]
def read(p):return json.loads(p.read_text())
def write(n,v):
 with (R/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert read(B/'terminal.json')['exit_code']==1
old=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-ordinary-to450-root-v1-host-supervision');assert read(old/'terminal.json')['child_exit_code']!=0
assert "FileNotFoundError" in (old/'process.log').read_text() and 'startup-telemetry.jsonl' in (old/'process.log').read_text()
recipe=read(B/'runtime-recipe.json');archive=pathlib.Path(recipe['outputs']['archive']);output=pathlib.Path(recipe['outputs']['trainer']);assert not archive.exists() and not output.exists();archive.mkdir(parents=True);output.mkdir(parents=True)
free=shutil.disk_usage('/home').free;assert free-recipe['checkpoint_storage']['expected_full_checkpoint_bytes']>=70*1024**3
cp=pathlib.Path(read(B/'continuation.json')['checkpoint']);m=read(cp/'campaign-manifest.json')
for name in ('model.safetensors','optimizer.pt'):
 p=cp/name;s=p.stat();assert s.st_size==m['files'][name]['bytes'];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
 finally:os.close(fd)
 t=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
g=read(B/'guard-command.json');g[g.index('--output')+1]+='-b';write('guard-command.json',g)
env=dict(os.environ);env.update(read(B/'environment.json'));env['SEPALITH_ALLOCATOR_REPORT']=str(R/'allocator.json');write('environment.json',{k:env[k]for k in read(B/'environment.json')})
write('launch.json',{'at':at(),'controller_pid':os.getpid(),'source':'selected_packed330','stop_at':450,'change':'create empty owned archive/trainer output directories before unchanged trainer startup telemetry','scientific_recipe_unchanged':True,'original_failed_guard':str(old),'free_bytes':free})
with (R/'guard.log').open('x') as log:
 p=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('guard-process.json',{'at':at(),'pid':p.pid});code=p.wait()
write('terminal.json',{'at':at(),'exit_code':code,'status':'commands_complete_requires_root_review' if code==0 else 'failed_requires_root_review','training_complete':False})
