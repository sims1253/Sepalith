#!/usr/bin/env python3
"""Root-owned actual daily editor smoke; native launched only by reviewed daily v2."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent
PACK=HERE.parents[1]/'daily-lan-launch-preparation-v2'
STATE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-01-daily-editor-smoke-b')
REMOTE='/home/m0hawk/.local/share/sepalith-campaign-20260915'
HOST='m0hawk@192.168.178.40'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'startTick':fields[19],'uid':p.stat().st_uid}
 except FileNotFoundError:return None

def interrupted(*_):raise InterruptedError('root editor controller interrupted')
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
assert not STATE.exists();STATE.mkdir();started=time.monotonic();child=None;ssh=None;current=None;failure=None
try:
 with (HERE/'daily-launcher.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B',str(PACK/'daily_lan.py'),'start','--state-dir',str(STATE),'--admission',str(HERE/'deployment-admission.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  save('root-launch.json',{'at':stamp(),'controller':identity(os.getpid()),'daily_supervisor':identity(child.pid)})
  until=time.monotonic()+90
  while time.monotonic()<until:
   if child.poll() is not None:raise RuntimeError('daily launcher exited before ready')
   dirs=[p for p in STATE.iterdir() if p.is_dir()];assert len(dirs)<=1
   if dirs:current=dirs[0]
   if current and (current/'ready.json').exists():break
   time.sleep(.2)
  else:raise TimeoutError('native/daily readiness 90 second bound')
  digest=hashlib.sha256((current/'binding.json').read_bytes()).hexdigest()
  upload=json.loads((current/'binding-upload.log').read_text());assert upload['bindingSha256']==digest and upload['instanceId']==current.name
  command=shlex.join(['exec','python3','-B',REMOTE+'/daily-editor-smoke-a-capsule/notebook_editor_supervisor.py','--run-root',REMOTE+'/runs/daily-editor-smoke-b','--binding-sha256',digest,'--instance-id',current.name])
  with (HERE/'notebook.stdout.log').open('xb') as out,(HERE/'notebook.stderr.log').open('xb') as err:
   ssh=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',HOST,command],stdin=subprocess.DEVNULL,stdout=out,stderr=err)
   save('notebook-launch.json',{'at':stamp(),'local_ssh':identity(ssh.pid),'session':str(current),'binding_sha256':digest,'remote_run':REMOTE+'/runs/daily-editor-smoke-b'})
   deadline=time.monotonic()+320
   while ssh.poll() is None:
    if child.poll() is not None:raise RuntimeError('daily runtime failed during editor smoke')
    if time.monotonic()>deadline:raise TimeoutError('remote supervisor exceeded 320 seconds')
    time.sleep(.5)
   if ssh.returncode:failure='Notebook supervisor SSH exit '+str(ssh.returncode)+'; evidence review required'
except Exception as exc:failure=type(exc).__name__+': '+str(exc)
finally:
 if ssh is not None and ssh.poll() is None:
  ssh.terminate()
  try:ssh.wait(timeout=10)
  except subprocess.TimeoutExpired:failure=(failure or '')+'; SSH still active, notebook bounded supervisor requires review'
 if child is not None and child.poll() is None:
  child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=25)
  except subprocess.TimeoutExpired:failure=(failure or '')+'; daily cleanup timed out'
 terminal=json.loads((current/'terminal.json').read_text()) if current and (current/'terminal.json').exists() else None
 save('controller-terminal.json',{'at':stamp(),'seconds':time.monotonic()-started,'failure':failure,'session':str(current) if current else None,'daily_exit':child.returncode if child else None,'ssh_exit':ssh.returncode if ssh else None,'daily_terminal':terminal,'scope':'Process completion only; root must review actual editor evidence.'})
print(json.dumps({'failure':failure,'seconds':time.monotonic()-started,'session':str(current) if current else None}),flush=True)
raise SystemExit(1 if failure else 0)
