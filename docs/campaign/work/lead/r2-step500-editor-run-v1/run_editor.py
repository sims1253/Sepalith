#!/usr/bin/env python3
"""Root-owned selected500 editor run. Requires separately issued fresh admission."""
import datetime,fcntl,hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACK=HERE.parent/'r2-step500-lan-preparation-v1'
STATE=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-01-r2-step500-editor-a')
REMOTE='/home/m0hawk/.local/share/sepalith-r2-step500-checks'
HOST='m0hawk@192.168.178.40'
LOCK=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
def save(name,obj): (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
def ssh(args,timeout=25):
 return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',HOST,shlex.join(args)],capture_output=True,text=True,timeout=timeout)
def ident(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return None if s[0] in ('Z','X','x') else {'pid':pid,'startTick':s[19],'uid':p.stat().st_uid}
 except FileNotFoundError:return None
def interrupted(*_):raise InterruptedError('root controller interrupted')
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
assert (HERE/'deployment-admission.json').is_file()
assert not STATE.exists()
STATE.mkdir()
child=None;current=None;failure=None;started=time.monotonic();cleanup=None
try:
 with (HERE/'launcher.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B',str(PACK/'daily_lan.py'),'start','--state-dir',str(STATE),'--admission',str(HERE/'deployment-admission.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  supervisor=ident(child.pid);save('launch.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor':supervisor,'controller':ident(os.getpid())})
  until=time.monotonic()+100
  while time.monotonic()<until:
   if child.poll() is not None:raise RuntimeError('LAN launcher exited before ready')
   dirs=[p for p in STATE.iterdir() if p.is_dir()];assert len(dirs)<=1
   current=dirs[0] if dirs else None
   if current and (current/'ready.json').exists():break
   time.sleep(.2)
  else:raise TimeoutError('LAN readiness')
  raw=(current/'binding.json').read_bytes();binding=json.loads(raw)
  assert binding['instanceId']==current.name
  assert binding['manifest']['model']['sha256']=='d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db'
  remote=ssh(['sha256sum',REMOTE+'/current-binding.json']);assert remote.returncode==0
  binding_sha=hashlib.sha256(raw).hexdigest();assert remote.stdout.split()[0]==binding_sha
  run=REMOTE+'/runs/'+current.name
  args=['timeout','--signal=TERM','--kill-after=15s','230s','xvfb-run','-a','--server-args=-screen 0 1280x800x24','node',REMOTE+'/capsule/run_remote_editor.mjs','--code','/usr/bin/code','--vsix',REMOTE+'/package/candidate.vsix','--vsix-sha256','b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1','--binding',REMOTE+'/current-binding.json','--binding-sha256',binding_sha,'--instance-id',current.name,'--debug-port','19403','--renderer-source','/usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js','--renderer-sha256','e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78','--run-root',run,'--timeout-ms','180000','--debounce-ms','350']
  save('editor-command.json',{'argv':args,'binding_sha256':binding_sha,'instance':current.name})
  result=ssh(args,250);save('editor-command-result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
  for script,extra in [('analyze_renderer.mjs',[run+'/renderer-analysis.json']),('analyze_auto.mjs',[])]:
   r=ssh(['node',REMOTE+'/capsule/'+script,run]+extra,30);save(script+'.result.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if result.returncode:raise RuntimeError('Editor launcher failed; inspect retained evidence')
except Exception as exc:failure=type(exc).__name__+': '+str(exc)
finally:
 if child is not None:
  if child.poll() is None:child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:failure=(failure or '')+'; LAN cleanup timed out'
 if current and (current/'terminal.json').exists():
  cleanup=json.loads((current/'terminal.json').read_text())
  if not cleanup.get('resourceReleaseProven'):failure=(failure or '')+'; resource release unproven'
 if child is not None and child.poll() is not None and cleanup and cleanup.get('resourceReleaseProven'):
  try:
   with LOCK.open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   cleanup['root_lock_reacquired']=True
  except BlockingIOError:failure=(failure or '')+'; CUDA lock still held'
 save('controller-terminal.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failure':failure,'seconds':time.monotonic()-started,'session':str(current) if current else None,'cleanup':cleanup,'release_acceptance':'requires independent notebook owned-process and editor evidence review'})
raise SystemExit(1 if failure else 0)
