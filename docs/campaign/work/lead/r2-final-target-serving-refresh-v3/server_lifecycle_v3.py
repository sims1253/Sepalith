#!/usr/bin/env python3
"""Root-owned bounded llama-server plus paired-probe lifecycle."""
import argparse,json,os,signal,socket,subprocess,tempfile,time
from pathlib import Path
from urllib.request import urlopen
from prepare_refresh_v3 import artifact,require,sha,validate_binding,validate_norm_audit,write_new

def start_ticks(pid):
 fields=Path(f'/proc/{pid}/stat').read_text().split();return int(fields[21])
def identity_alive(pid,ticks):
 try:return start_ticks(pid)==ticks
 except (FileNotFoundError,ProcessLookupError,ValueError):return False
def port_is_free(host,port):
 s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:s.bind((host,port));return True
 except OSError:return False
 finally:s.close()
def replace_port(argv,port):
 out=[]
 for value in argv:out.append(str(port) if value=='$PORT' else value)
 return out
def server_argv(b,mode):
 key=b['selected_quant_artifact_key'];validate_norm_audit(b,key);model=artifact(b['outputs'][key],key);base=[b['tools']['server']['path'],'-m',str(model),*b['runtime']['common_server_argv']]
 if mode=='ordinary':return base
 if mode=='ngram_mod':return [*base,*b['runtime']['ngram_argv']]
 require(mode in {'released_dspark','existing_trained_dspark'},'server mode unsupported');d=b['drafts'][mode];artifact(d,mode);artifact({'path':d['header_receipt_path'],'sha256':d['header_receipt_sha256']},mode+' header');return [*base,'-md',d['path'],*b['runtime']['dspark_argv']]
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def process_telemetry(pid,ticks):
 require(identity_alive(pid,ticks),'process identity changed during telemetry');values={}
 for line in Path(f'/proc/{pid}/status').read_text().splitlines():
  if line.startswith(('State:','VmRSS:','VmSize:','Threads:')):
   key,value=line.split(':',1);values[key]=value.strip()
 return values
def terminate_owned(proc,ticks,grace,events):
 if identity_alive(proc.pid,ticks):
  os.killpg(proc.pid,signal.SIGTERM);append(events,{'event':'TERM','pid':proc.pid,'start_ticks':ticks,'at':time.time()})
  try:proc.wait(timeout=grace)
  except subprocess.TimeoutExpired:
   if identity_alive(proc.pid,ticks):os.killpg(proc.pid,signal.SIGKILL);append(events,{'event':'KILL','pid':proc.pid,'start_ticks':ticks,'at':time.time()})
   proc.wait(timeout=5)
 return proc.returncode
def run(binding_path,profile,mode,run_dir):
 bpath=Path(binding_path).resolve();b=json.loads(bpath.read_text());validate_binding(b,True,True);life=b['server_lifecycle'];host,port=life['host'],life['port'];require(port_is_free(host,port),'selected port already bound');run_dir=Path(run_dir);require(run_dir.is_absolute() and not run_dir.exists(),'fresh absolute run directory required');run_dir.mkdir(parents=True);events=run_dir/'events.jsonl';server_log=(run_dir/'server.log').open('wb');deadline=time.monotonic()+life['deadline_seconds'];argv=replace_port(server_argv(b,mode),port);proc=subprocess.Popen(argv,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True);ticks=start_ticks(proc.pid);append(events,{'event':'START','pid':proc.pid,'start_ticks':ticks,'port':port,'artifact_key':b['selected_quant_artifact_key'],'mode':mode,'deadline_seconds':life['deadline_seconds'],'at':time.time()})
 status='failed';probe_exit=None;child=None;child_ticks=None
 try:
  health_deadline=min(deadline,time.monotonic()+life['health_timeout_seconds'])
  while time.monotonic()<health_deadline:
   require(identity_alive(proc.pid,ticks),'server exited before health')
   try:
    with urlopen(f'http://{host}:{port}{life["health_path"]}',timeout=1) as response:
     if response.status==200:append(events,{'event':'HEALTHY','pid':proc.pid,'start_ticks':ticks,'port':port,'at':time.time()});break
   except Exception:time.sleep(.25)
  else:raise RuntimeError('health deadline exceeded')
  probe_out=run_dir/'probe.json';probe=[os.environ.get('SEPALITH_SERVING_PYTHON','/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'),'-B',str(Path(__file__).with_name('probe_bound_v3.py')),'--binding',str(bpath),'--profile',profile,'--mode',mode,'--url',f'http://{host}:{port}','--out',str(probe_out)]
  require(deadline-time.monotonic()>0,'server deadline elapsed before probe');child=subprocess.Popen(probe,start_new_session=True);child_ticks=start_ticks(child.pid);append(events,{'event':'PROBE_START','pid':child.pid,'start_ticks':child_ticks,'at':time.time()});next_sample=0
  while child.poll() is None:
   if time.monotonic()>=deadline:
    if identity_alive(child.pid,child_ticks):os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
     if identity_alive(child.pid,child_ticks):os.killpg(child.pid,signal.SIGKILL)
     child.wait(timeout=5)
    raise TimeoutError('paired probe exceeded lifecycle deadline')
   if time.monotonic()>=next_sample:
    append(events,{'event':'TELEMETRY','server':process_telemetry(proc.pid,ticks),'probe':process_telemetry(child.pid,child_ticks),'at':time.time()});next_sample=time.monotonic()+5
   time.sleep(.25)
  probe_exit=child.returncode;require(probe_exit in (0,1) and probe_out.is_file(),'paired probe failed without bounded result');status='completed'
 except TimeoutError:status='deadline_exceeded'
 except Exception as exc:append(events,{'event':'ERROR','error_type':type(exc).__name__,'at':time.time()})
 finally:
  if child is not None and child.poll() is None:terminate_owned(child,child_ticks,5,events)
  rc=terminate_owned(proc,ticks,life['terminate_grace_seconds'],events);server_log.close();append(events,{'event':'CLEANUP','pid':proc.pid,'start_ticks':ticks,'process_exit_code':rc,'identity_alive_after':identity_alive(proc.pid,ticks),'at':time.time()})
 terminal={'schema':'sepalith.run06.server-lifecycle-terminal.v1','status':status,'binding_sha256':sha(bpath),'pid':proc.pid,'start_ticks':ticks,'host':host,'port':port,'artifact_key':b['selected_quant_artifact_key'],'mode':mode,'profile':profile,'probe_exit_code':probe_exit,'server_exit_code':proc.returncode,'health_path':life['health_path'],'deadline_seconds':life['deadline_seconds'],'cleanup_verified':not identity_alive(proc.pid,ticks),'events_sha256':sha(events),'server_log_sha256':sha(run_dir/'server.log')};write_new(run_dir/'terminal.json',terminal);return terminal

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--profile',choices=('ngram','draft'),required=True);p.add_argument('--mode',choices=('ordinary','ngram_mod','released_dspark','existing_trained_dspark'),required=True);p.add_argument('--run-dir',required=True);a=p.parse_args();print(json.dumps(run(a.binding,a.profile,a.mode,a.run_dir),sort_keys=True))
