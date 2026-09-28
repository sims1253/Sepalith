#!/usr/bin/env python3
"""Root-owned wall-clock supervisor for the immutable notebook quality runner."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
BASE=Path.home()/'.local/share/sepalith-e750-notebook-cpu-quality-v1'
ENTRY=BASE/'quality6-v1/remote_entry.sh'
MAX_SECONDS=28800
TERM_GRACE_SECONDS=150
STOP=None

def proc_tick(pid):
 p=Path(f'/proc/{pid}/stat')
 if not p.exists():return None
 text=p.read_text();return text[text.rfind(')')+2:].split()[19]
def owned(pid,tick):return isinstance(pid,int) and isinstance(tick,str) and proc_tick(pid)==tick
def atomic(path,value):
 tmp=path.with_suffix(path.suffix+'.new')
 with tmp.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def arm_servers(run_root):
 out=[]
 for launch in sorted(run_root.glob('cap-*/launch.json')):
  try:
   x=json.loads(launch.read_text());pid=x['pid'];tick=x['start_tick']
   if not isinstance(pid,int) or not isinstance(tick,str):raise TypeError('bad identity')
   out.append((pid,tick,str(launch)))
  except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError):out.append((None,None,str(launch)))
 return out
def stop_server(pid,tick,source):
 result={'pid':pid,'start_tick':tick,'source':source,'matched':False,'action':'absent'}
 if not isinstance(pid,int) or not isinstance(tick,str):result['action']='invalid_identity_preserved';return result
 if not owned(pid,tick):return result
 result['matched']=True;result['action']='TERM'
 try:os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:return result
 until=time.monotonic()+10
 while time.monotonic()<until and owned(pid,tick):time.sleep(.1)
 if owned(pid,tick):
  result['action']='TERM_then_KILL'
  try:os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 return result
def supervise(run_id,admission):
 run_root=BASE/'quality6-runs'/run_id;terminal=BASE/'quality6-logs'/f'{run_id}.watchdog-terminal.json'
 if run_root.exists() or terminal.exists():raise FileExistsError('fresh watchdog outputs required')
 a=json.loads(admission.read_text())
 if a.get('status')!='admitted' or a.get('run_id')!=run_id or a.get('maximum_seconds')!=MAX_SECONDS:raise ValueError('admission mismatch')
 started=time.monotonic();proc=subprocess.Popen([str(ENTRY),run_id,str(admission)],start_new_session=True);tick=proc_tick(proc.pid);reason='runner_exit'
 while proc.poll() is None:
  if STOP:reason=STOP;break
  if time.monotonic()-started>=MAX_SECONDS:reason='maximum_seconds_elapsed';break
  time.sleep(.25)
 cleanup=[]
 if proc.poll() is None:
  if owned(proc.pid,tick):
   try:os.kill(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  until=time.monotonic()+TERM_GRACE_SECONDS
  while proc.poll() is None and time.monotonic()<until:time.sleep(.25)
  if proc.poll() is None:
   cleanup=[stop_server(*x) for x in arm_servers(run_root)]
   if owned(proc.pid,tick):
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
 cleanup.extend(stop_server(*x) for x in arm_servers(run_root) if owned(x[0],x[1]))
 result={'schema':'sepalith.run06.e750-notebook-root-watchdog.v1','status':'terminal','reason':reason,'maximum_seconds':MAX_SECONDS,'term_grace_seconds':TERM_GRACE_SECONDS,'runner':{'pid':proc.pid,'start_tick':tick,'returncode':proc.poll(),'absent':not owned(proc.pid,tick)},'server_cleanup':cleanup,'elapsed_seconds':time.monotonic()-started}
 atomic(terminal,result);return 0 if reason=='runner_exit' and proc.returncode==0 else 1
def main():
 global STOP
 def handler(signum,_):
  global STOP;STOP=signal.Signals(signum).name
 for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,handler)
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--admission',type=Path,required=True);a=p.parse_args()
 return supervise(a.run_id,a.admission.resolve())
if __name__=='__main__':raise SystemExit(main())
