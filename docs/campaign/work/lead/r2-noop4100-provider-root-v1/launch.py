import pathlib,json,hashlib,subprocess,os,time,datetime,signal
R=pathlib.Path(__file__).resolve().parent
P=R.parents[4]
def write(name,value):
 with(R/name).open('x')as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
c=json.loads((R/'commands.json').read_text());assert not pathlib.Path(c['output_root']).exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');children=[];logs=[]
try:
 launches=[]
 for lane in c['lanes']:
  base=R.parent/'r2-semantic10948-provider-materialization-v2';m=base/'source-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='7546d5457c299edfbf69bc0681327f08bad5041e2a146fe60fd66278dd58c770'
  for x in json.loads(m.read_text())['files']:
   f=pathlib.Path(x['path']);f=f if f.is_absolute()else base/f;assert f.stat().st_size==x['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256']
  command=['timeout','--signal=TERM','--kill-after=30s','9000',*lane['command']];log=(R/f"lane{lane['lane']}.log").open('x');logs.append(log);child=subprocess.Popen(command,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);launches.append(dict(lane,pid=child.pid,guarded_command=command))
 write('launch.json',{'at':at(),'controller_pid':os.getpid(),'lanes':launches,'plan_sha256':c['plan_sha256'],'CPU_only':True,'rows':4100,'training_admitted':False})
 while True:
  codes=[x.poll()for x in children]
  if any(x not in(None,0)for x in codes):raise RuntimeError('lane exit codes '+str(codes))
  if all(x==0 for x in codes):break
  time.sleep(2)
 write('terminal.json',{'at':at(),'status':'commands_complete_requires_root_review','exit_codes':codes,'training_admitted':False})
except BaseException as e:
 for child in children:
  if child.poll()is None:os.killpg(child.pid,signal.SIGTERM)
 for child in children:
  try:child.wait(timeout=35)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 write('terminal.json',{'at':at(),'status':'failed_no_retry','error':type(e).__name__+': '+str(e),'exit_codes':[x.returncode for x in children]});raise
finally:
 for log in logs:log.close()
