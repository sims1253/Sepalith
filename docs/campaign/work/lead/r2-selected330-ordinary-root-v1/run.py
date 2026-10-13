import datetime,hashlib,json,os,pathlib,subprocess,shutil
R=pathlib.Path(__file__).resolve().parent;PLAN=R.parents[4]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,v):
 with (R/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert read(R/'cpu-preflight-terminal.json')['exit_code']==0
pre=json.loads((R/'cpu-preflight.log').read_text().splitlines()[-1]);assert pre['status']=='pass' and pre['resume_step']==330 and pre['initial_cursor']==4224
recipe=read(R/'runtime-recipe.json');assert sha(R/'runtime-recipe.json')=='eff266cdb287e8b099990f265760539e024bc20c14dd0707b6e225abce1a6981'
for key in ('trainer','archive'):assert not pathlib.Path(recipe['outputs'][key]).exists()
free=shutil.disk_usage('/home').free;reserve=recipe['checkpoint_storage']['expected_full_checkpoint_bytes'];assert free-reserve>=70*1024**3
write('capacity.json',{'at':at(),'free_before':free,'checkpoint_reserve':reserve,'free_after_projected':free-reserve,'minimum_free':70*1024**3})
base=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training');released=[]
for rel,h in [('SFT11-native-CPT-prefix-extension-v3-from194/runtime/checkpoint-322','81dfdfb4af6d0690ebb114f68910fdbe270d20ad3393a84c47c1c337ce2dbbcf'),('SFT11-native-varlen-canary-322-ordinary_reference-v1/runtime/checkpoint-330','f7bd7b819584a8abeadc37c677914bd8775f350bd097479d326425292ba8e943'),('SFT11-native-varlen-canary-322-varlen_candidate-v1/runtime/checkpoint-330','2b35854537588499fb703af2127029ff649e410cbab65374948723273b520951')]:
 cp=base/rel;assert sha(cp/'campaign-manifest.json')==h;m=read(cp/'campaign-manifest.json')
 for name in ('model.safetensors','optimizer.pt'):
  p=cp/name;b=p.stat();assert b.st_size==m['files'][name]['bytes'];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  a=p.stat();assert (b.st_ino,b.st_size,b.st_mtime_ns)==(a.st_ino,a.st_size,a.st_mtime_ns);released.append(str(p))
write('clean-cache-release.json',{'at':at(),'paths':released,'content_unchanged':True})
env=dict(os.environ);env.update(read(R/'environment.json'));g=read(R/'guard-command.json');write('launch.json',{'at':at(),'controller_pid':os.getpid(),'source':'selected_packed330','runtime':'ordinary','stop_at':450,'guard_command_sha256':sha(R/'guard-command.json'),'wrapper_sha256':sha(R/'run_with_allocator.py')})
with (R/'guard.log').open('x') as log:
 p=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write('guard-process.json',{'pid':p.pid,'at':at()});code=p.wait()
write('terminal.json',{'at':at(),'exit_code':code,'status':'commands_complete_requires_root_review' if code==0 else 'failed_requires_root_review','training_complete':False})
