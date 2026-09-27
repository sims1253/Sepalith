#!/usr/bin/env python3
"""Read-only allowlisted sidecar process observation for the owned CPT run."""
import json,time
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'r2-cloud-cpt-live-observer-v1'
sys.path.insert(0,str(BASE))
import live_observer as o
REMOTE=r'''import glob,json,os,sys,time
run=sys.argv[1];paths=glob.glob('/mnt/local_storage/sepalith-cpt-'+run+'-*')
if len(paths)!=1:raise SystemExit('owned run directory count differs')
root=paths[0]
def ints(path,names):
 out={}
 try:
  for line in open(path):
   key=line.split(':',1)[0]
   if key in names:
    value=line.split(':',1)[1].strip().split()[0];out[key]=int(value) if value.isdigit() else value
 except OSError:pass
 return out
def snapshot():
 found=[]
 for q in glob.glob('/proc/[0-9]*/cmdline'):
  try:
   raw=open(q,'rb').read().decode(errors='replace');parts=[x for x in raw.split(chr(0)) if x]
   if not any(os.path.basename(x)=='checkpoint_sidecar.py' for x in parts) or run not in raw:continue
   pid=int(q.split('/')[2]);base='/proc/'+str(pid);fds={'socket':0,'pipe':0,'regular':0,'checkpoint_file':0};socket_inodes=[]
   for f in glob.glob(base+'/fd/*'):
    try:target=os.readlink(f)
    except OSError:continue
    if target.startswith('socket:['):fds['socket']+=1;socket_inodes.append(target[8:-1])
    elif target.startswith('pipe:['):fds['pipe']+=1
    elif target.startswith(root):fds['regular']+=1;fds['checkpoint_file']+=('/artifacts/archive/full/checkpoint-' in target)
   status=ints(base+'/status',{'PPid','VmRSS','voluntary_ctxt_switches','nonvoluntary_ctxt_switches'})
   io=ints(base+'/io',{'rchar','wchar','read_bytes','write_bytes'})
   stat=open(base+'/stat').read().split();wchan=open(base+'/wchan').read().strip()[:80]
   env=open(base+'/environ','rb').read().split(chr(0).encode())
   keys={x.split(b'=',1)[0].decode(errors='ignore') for x in env if b'=' in x}
   found.append({'pid':pid,'ppid':status.get('PPid'),'state':stat[2],'utime_ticks':int(stat[13]),'stime_ticks':int(stat[14]),'start_ticks':int(stat[21]),'wchan':wchan,'vmrss_kib':status.get('VmRSS'),'voluntary_context_switches':status.get('voluntary_ctxt_switches'),'nonvoluntary_context_switches':status.get('nonvoluntary_ctxt_switches'),'io':io,'fds':fds,'owned_socket_inodes':len(socket_inodes),'hf_token_present':'HF_TOKEN' in keys,'python':os.path.basename(os.readlink(base+'/exe')),'argv_shape':[os.path.basename(x) if i<2 else ('--stop-file' if x=='--stop-file' else ('owned_run_path' if run in x else os.path.basename(x))) for i,x in enumerate(parts)]})
  except (OSError,ValueError):pass
 cp=root+'/artifacts/cloud-persistence';entries=[]
 if os.path.isdir(cp):
  entries=[{'name':n,'bytes':os.stat(cp+'/'+n).st_size} for n in sorted(os.listdir(cp)) if os.path.isfile(cp+'/'+n)]
 check=root+'/artifacts/archive/full/checkpoint-317';manifest=check+'/campaign-manifest.json'
 checkpoint={'exists':os.path.isdir(check),'manifest_exists':os.path.isfile(manifest),'manifest_mtime_ns':os.stat(manifest).st_mtime_ns if os.path.isfile(manifest) else None}
 return {'at_ns':time.time_ns(),'processes':found,'cloud_persistence':entries,'sidecar_log_bytes':os.stat(root+'/sidecar.log').st_size if os.path.isfile(root+'/sidecar.log') else None,'stop_file_exists':os.path.exists(root+'/stop-sidecar'),'checkpoint317':checkpoint}
a=snapshot();time.sleep(7);b=snapshot();print(json.dumps({'run_id':run,'run_directory_match_count':1,'first':a,'second':b},sort_keys=True))'''
o.REMOTE=REMOTE
value=o.observe(use_hf=False)
remote=value['remote'];a=remote['first'];b=remote['second']
for field in ('provider','private_artifacts'):pass
out={'schema':'sepalith.cloud-cpt.persistence-diagnosis-observation.v1','observed_at':value['observed_at'],'run_id':value['run_id'],'provider':value['provider'],'private_artifacts':{'status':'checked_separately'},'remote':remote}
p=HERE/'live-sidecar-observation.json';
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'provider':out['provider']['provider_run_status'],'processes':b['processes'],'persistence':b['cloud_persistence'],'private_status':out['private_artifacts'].get('status')}))
