#!/usr/bin/env python3
"""Bounded server/probe lifecycle; every descendant retains the root CUDA lease fd."""
import argparse,ctypes,json,os,signal,socket,subprocess,time
from pathlib import Path
from urllib.request import urlopen
from prepare_refresh_v4 import require,sha,validate_binding,validate_norm_audit,write_new
from record_prelaunch_verification import validate as validate_prelaunch

class LifecycleSignal(Exception):pass
def start_ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().split()[21])
def identity_alive(pid,ticks):
 try:return start_ticks(pid)==ticks
 except (FileNotFoundError,ProcessLookupError,ValueError):return False
def parent_death_term():
 if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('PR_SET_PDEATHSIG failed')
def require_lock_fd(fd):
 require(type(fd) is int and fd>=3,'stable CUDA lock fd required');os.fstat(fd)
def port_is_free(host,port):
 s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:s.bind((host,port));return True
 except OSError:return False
 finally:s.close()
def replace_port(argv,port):return [str(port) if x=='$PORT' else x for x in argv]
def server_argv(b,mode):
 key=b['selected_quant_artifact_key'];validate_norm_audit(b,key);model=Path(b['outputs'][key]['path']);require(model.is_file() and not model.is_symlink(),'selected quant missing/symlink');base=[b['tools']['server']['path'],'-m',str(model),*b['runtime']['common_server_argv']]
 if mode=='ordinary':return base
 if mode=='ngram_mod':return [*base,*b['runtime']['ngram_argv']]
 require(mode in {'released_dspark','existing_trained_dspark'},'server mode unsupported');d=b['drafts'][mode];require(d.get('available') and Path(d['path']).is_file(),'draft unavailable');return [*base,'-md',d['path'],*b['runtime']['dspark_argv']]
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def process_telemetry(pid,ticks):
 require(identity_alive(pid,ticks),'process identity changed during telemetry');values={}
 for line in Path(f'/proc/{pid}/status').read_text().splitlines():
  if line.startswith(('State:','VmRSS:','VmSize:','Threads:')):k,v=line.split(':',1);values[k]=v.strip()
 return values
def terminate_owned(proc,ticks,grace,events):
 if proc is None:return None
 # Before /proc start_ticks are captured, an unreaped Popen still owns this exact child PID.
 owned_alive=(ticks is None and proc.poll() is None) or (ticks is not None and identity_alive(proc.pid,ticks))
 if owned_alive:
  try:os.kill(proc.pid,signal.SIGTERM);append(events,{'event':'TERM','pid':proc.pid,'start_ticks':ticks,'at':time.time()})
  except ProcessLookupError:pass
  try:proc.wait(timeout=grace)
  except subprocess.TimeoutExpired:
   if identity_alive(proc.pid,ticks):os.kill(proc.pid,signal.SIGKILL);append(events,{'event':'KILL','pid':proc.pid,'start_ticks':ticks,'at':time.time()})
   proc.wait(timeout=5)
 return proc.returncode
def child_env(lock_fd):
 env=os.environ.copy();env['SEPALITH_CUDA_LOCK_FD']=str(lock_fd);return env
def spawn_owned(argv,log,lock_fd):
 return subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,close_fds=True,pass_fds=(lock_fd,),start_new_session=False,preexec_fn=parent_death_term,env=child_env(lock_fd))

def run(binding_path,verification_path,verification_sha256,profile,mode,run_dir,cuda_lock_fd):
 bpath=Path(binding_path).resolve();b=json.loads(bpath.read_text());validate_binding(b,True,False,True);validate_prelaunch(b,bpath,verification_path,verification_sha256);require_lock_fd(cuda_lock_fd)
 life=b['server_lifecycle'];host,port=life['host'],life['port'];require(port_is_free(host,port),'selected port already bound');run_dir=Path(run_dir);require(run_dir.is_absolute() and not run_dir.exists(),'fresh absolute run directory required')
 proc=child=None;ticks=child_ticks=None;server_log=None;events=run_dir/'events.jsonl';status='failed';probe_exit=None;old_handlers={}
 def on_signal(signum,_frame):raise LifecycleSignal(signal.Signals(signum).name)
 for sig in (signal.SIGTERM,signal.SIGINT):old_handlers[sig]=signal.signal(sig,on_signal)
 try:
  run_dir.mkdir(parents=True);server_log=(run_dir/'server.log').open('wb');deadline=time.monotonic()+life['deadline_seconds'];argv=replace_port(server_argv(b,mode),port);proc=spawn_owned(argv,server_log,cuda_lock_fd);ticks=start_ticks(proc.pid);append(events,{'event':'START','pid':proc.pid,'start_ticks':ticks,'port':port,'artifact_key':b['selected_quant_artifact_key'],'mode':mode,'cuda_lock_fd':cuda_lock_fd,'deadline_seconds':life['deadline_seconds'],'at':time.time()})
  health_deadline=min(deadline,time.monotonic()+life['health_timeout_seconds'])
  while time.monotonic()<health_deadline:
   require(identity_alive(proc.pid,ticks),'server exited before health')
   try:
    with urlopen(f'http://{host}:{port}{life["health_path"]}',timeout=1) as response:
     if response.status==200:append(events,{'event':'HEALTHY','pid':proc.pid,'start_ticks':ticks,'port':port,'at':time.time()});break
   except Exception:time.sleep(.25)
  else:raise RuntimeError('health deadline exceeded')
  probe_out=run_dir/'probe.json';probe=[os.environ.get('SEPALITH_SERVING_PYTHON','/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'),'-B',str(Path(__file__).with_name('probe_bound_v4.py')),'--binding',str(bpath),'--verification',str(Path(verification_path).resolve()),'--verification-sha256',verification_sha256,'--cuda-lock-fd',str(cuda_lock_fd),'--profile',profile,'--mode',mode,'--url',f'http://{host}:{port}','--out',str(probe_out)]
  require(deadline-time.monotonic()>0,'server deadline elapsed before probe');child=spawn_owned(probe,None,cuda_lock_fd);child_ticks=start_ticks(child.pid);append(events,{'event':'PROBE_START','pid':child.pid,'start_ticks':child_ticks,'at':time.time()});next_sample=0
  while child.poll() is None:
   if time.monotonic()>=deadline:raise TimeoutError('paired probe exceeded lifecycle deadline')
   if time.monotonic()>=next_sample:append(events,{'event':'TELEMETRY','server':process_telemetry(proc.pid,ticks),'probe':process_telemetry(child.pid,child_ticks),'at':time.time()});next_sample=time.monotonic()+5
   time.sleep(.25)
  probe_exit=child.returncode;require(probe_exit in (0,1) and probe_out.is_file(),'paired probe failed without bounded result');status='completed'
 except LifecycleSignal as exc:status='signal_'+str(exc).lower();append(events,{'event':'SIGNAL','signal':str(exc),'at':time.time()})
 except TimeoutError:status='deadline_exceeded'
 except Exception as exc:append(events,{'event':'ERROR','error_type':type(exc).__name__,'at':time.time()})
 finally:
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
  if run_dir.exists():
   terminate_owned(child,child_ticks,5,events);rc=terminate_owned(proc,ticks,life['terminate_grace_seconds'],events)
   if server_log is not None:server_log.close()
   append(events,{'event':'CLEANUP','pid':None if proc is None else proc.pid,'start_ticks':ticks,'process_exit_code':rc,'identity_alive_after':False if proc is None else identity_alive(proc.pid,ticks),'at':time.time()})
 terminal={'schema':'sepalith.run06.server-lifecycle-terminal.v2','status':status,'binding_sha256':sha(bpath),'prelaunch_verification_sha256':verification_sha256,'pid':None if proc is None else proc.pid,'start_ticks':ticks,'host':host,'port':port,'artifact_key':b['selected_quant_artifact_key'],'mode':mode,'profile':profile,'probe_exit_code':probe_exit,'server_exit_code':None if proc is None else proc.returncode,'health_path':life['health_path'],'deadline_seconds':life['deadline_seconds'],'cleanup_verified':proc is None or not identity_alive(proc.pid,ticks),'events_sha256':sha(events),'server_log_sha256':sha(run_dir/'server.log')};write_new(run_dir/'terminal.json',terminal);return terminal

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--verification',required=True);p.add_argument('--verification-sha256',required=True);p.add_argument('--cuda-lock-fd',required=True,type=int);p.add_argument('--profile',choices=('ngram','draft'),required=True);p.add_argument('--mode',choices=('ordinary','ngram_mod','released_dspark','existing_trained_dspark'),required=True);p.add_argument('--run-dir',required=True);a=p.parse_args();print(json.dumps(run(a.binding,a.verification,a.verification_sha256,a.profile,a.mode,a.run_dir,a.cuda_lock_fd),sort_keys=True))
