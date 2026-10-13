#!/usr/bin/env python3
"""Root-owned bounded two-session daily LAN lifecycle smoke, TRAIN input only."""
import datetime,fcntl,hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACK=HERE.parents[1]/'daily-lan-launch-preparation-v2'
STATE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-01-daily-lan-smoke-a')
LOCK=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
HOST='m0hawk@192.168.178.40'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
def ident(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return None if s[0] in ('Z','X','x') else {'pid':pid,'startTick':s[19],'uid':p.stat().st_uid}
 except FileNotFoundError:return None

def stop_signal(*_):raise InterruptedError('root_controller_interrupted')
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop_signal)
pins=json.loads((HERE/'smoke-client-pins.json').read_text())
assert not STATE.exists();STATE.mkdir()
records=[];child=None;failure=None;started=time.monotonic();current=None
try:
 for phase in ('first','reconnect'):
  prior=set(STATE.iterdir());log=(HERE/(phase+'-launcher.log')).open('xb')
  child=subprocess.Popen([sys.executable,'-B',str(PACK/'daily_lan.py'),'start','--state-dir',str(STATE),'--admission',str(HERE/'deployment-admission.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  supervisor=ident(child.pid);save(phase+'-root-launch.json',{'at':stamp(),'supervisor':supervisor,'controller':ident(os.getpid())})
  until=time.monotonic()+100;current=None
  while time.monotonic()<until:
   if child.poll() is not None:raise RuntimeError('launcher exited before ready: '+str(child.returncode))
   dirs=[p for p in STATE.iterdir() if p.is_dir() and p not in prior]
   assert len(dirs)<=1
   if dirs:current=dirs[0]
   if current and (current/'ready.json').exists():break
   time.sleep(.2)
  else:raise TimeoutError('launch readiness exceeded 100 seconds')
  instance=current.name
  command=shlex.join(['python3','-B',pins['remote']+'/daily_smoke_client.py',instance,pins['files']['train-fixture.jsonl']])
  ssh=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',HOST,command],capture_output=True,text=True,timeout=35)
  (HERE/(phase+'-client.stdout')).write_text(ssh.stdout);(HERE/(phase+'-client.stderr')).write_text(ssh.stderr)
  if ssh.returncode:raise RuntimeError('notebook TRAIN client exited '+str(ssh.returncode))
  result=json.loads(ssh.stdout);assert result['status']=='complete' and result['instance_id']==instance
  child.send_signal(signal.SIGTERM);exitcode=child.wait(timeout=25);log.close()
  terminal=json.loads((current/'terminal.json').read_text());assert exitcode==0 and terminal['failure'] is None and terminal['resourceReleaseProven']
  identities=[x['identity'] for x in terminal['cleanup']]+[supervisor]
  assert all(ident(x['pid'])!=x for x in identities)
  # Independent kernel lock probe after all recorded processes have exited.
  with LOCK.open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  records.append({'phase':phase,'session':str(current),'result':result,'terminal':terminal,'owned_identities_absent':identities,'global_lock_reacquired':True})
  save(phase+'-accepted-smoke.json',records[-1]);child=None
 assert records[0]['result']['instance_id']!=records[1]['result']['instance_id']
 assert records[0]['result']['outputs'][0]['raw_sha256']==records[1]['result']['outputs'][0]['raw_sha256']
except Exception as exc:failure=type(exc).__name__+': '+str(exc)
finally:
 if child is not None and child.poll() is None:
  child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=25)
  except subprocess.TimeoutExpired:failure=(failure or '')+'; supervisor failed cleanup; root identity review required'
 save('controller-terminal.json',{'at':stamp(),'failure':failure,'seconds':time.monotonic()-started,'completed_sessions':len(records),'sessions':[r['session'] for r in records],'last_session':str(current) if current else None,'last_supervisor_alive':child is not None and child.poll() is None})
print(json.dumps({'failure':failure,'completed_sessions':len(records),'seconds':time.monotonic()-started}),flush=True)
raise SystemExit(1 if failure else 0)
