#!/usr/bin/env python3
"""Root-authorized exact-sidecar stop and one checkpoint-317 upload diagnostic."""
import argparse,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'r2-cloud-cpt-live-observer-v1';sys.path.insert(0,str(BASE));import live_observer as o
JOB=o.JOB_ID;RUN=o.RUN_ID;PID=5498;START=18486
ACTION='stop_exact_sidecar_then_upload_checkpoint317_once'
REMOTE=r'''import base64,hashlib,json,os,signal,subprocess,sys,time
run=sys.argv[1];pid=5498;start=18486
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def fail(s):raise SystemExit(s)
base='/proc/'+str(pid);raw=open(base+'/cmdline','rb').read();parts=[x.decode() for x in raw.split(b'\0') if x]
stat=open(base+'/stat').read().split()
if int(stat[21])!=start or int(stat[3])!=4588:fail('exact sidecar process identity differs')
if len(parts)!=6 or os.path.basename(parts[1])!='checkpoint_sidecar.py' or parts[4]!='--stop-file' or run not in parts[3] or run not in parts[5]:fail('exact sidecar argv differs')
sidecar=os.path.realpath(parts[1]);binding=os.path.realpath(parts[2]);root=os.path.realpath(parts[3]);payload=os.path.dirname(sidecar);checkpoint=root+'/artifacts/archive/full/checkpoint-317';diagnostic=root+'/artifacts/cloud-persistence/checkpoint-317-upload-diagnostic.json'
if sha(sidecar)!='0b3d4df5e4a0ffd5ea679ad58fe709a0b2126230cdffa4b7f3c77719c6400511' or sha(payload+'/artifact_upload.py')!='a055094b5ab39c35e10b32acb4938ff56537946a6e55ed8785612360b6aaf146' or sha(payload+'/cloud_contract.py')!='2b345761d5e663e1bf9021856161f362f539e351e414005c39cf47f3952f256a':fail('active payload source differs')
if sha(binding)!='c31671d5e3895a394ee569b0d0537ebe81d636daa2dbbaad4a11fa6559d76dc5' or not os.path.isdir(checkpoint) or os.path.exists(diagnostic):fail('binding/checkpoint/fresh diagnostic precondition differs')
envraw=open(base+'/environ','rb').read().split(b'\0');matches=[x.split(b'=',1)[1].decode() for x in envraw if x.startswith(b'HF_TOKEN=')]
if len(matches)!=1 or not matches[0].startswith('hf_'):fail('single sidecar token absent')
token=matches[0];python=os.path.realpath(base+'/exe')
os.kill(pid,signal.SIGTERM)
for _ in range(100):
 try:
  current=open(base+'/stat').read().split();alive=int(current[21])==start and current[2]!='Z'
 except (OSError,ValueError):alive=False
 if not alive:break
 time.sleep(.1)
if alive:fail('exact sidecar did not stop after SIGTERM')
child=r"""import json,os,sys,traceback\nfrom pathlib import Path\npayload,binding,run,checkpoint,diagnostic=sys.argv[1:];sys.path.insert(0,payload)\nimport artifact_upload as a\ntry:\n value=a.upload_checkpoint(Path(run),json.loads(Path(binding).read_text()),Path(checkpoint));result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v1','status':'success','step':317,'checkpoint_revision':value.get('revision'),'receipt_revision':value.get('receipt_revision'),'credential_persisted':False}\nexcept Exception as e:\n result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v1','status':'failed','step':317,'error_type':type(e).__name__,'error_message':a.safe_message(e),'stack':a.safe_stack(e),'credential_persisted':False}\nPath(diagnostic).write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n');print(json.dumps(result,sort_keys=True),flush=True);raise SystemExit(0 if result['status']=='success' else 1)"""
env={'HF_TOKEN':token,'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':os.environ.get('HOME','/home/ray'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HOME':root+'/hf-cache'}
got=subprocess.run([python,'-I','-c',child,payload,binding,root,checkpoint,diagnostic],env=env,capture_output=True,text=True,timeout=900)
token=None;matches=None;envraw=None
try:result=json.loads(got.stdout.strip().splitlines()[-1])
except Exception:result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v1','status':'wrapper_failed','error_type':'ChildOutputError','credential_persisted':False}
print(json.dumps({'run_id':run,'run_directory_match_count':1,'sidecar_stopped':True,'sidecar_pid':pid,'sidecar_start_tick':start,'upload_result':result,'child_exit_code':got.returncode},sort_keys=True))'''
def auth(path):
 x=json.loads(path.read_text());required={'schema','authorized','job_id','run_id','sidecar_pid','sidecar_start_tick','action'}
 if set(x)!=required or x!={'schema':'sepalith.cloud-cpt.persistence-repair-root-authorization.v1','authorized':True,'job_id':JOB,'run_id':RUN,'sidecar_pid':PID,'sidecar_start_tick':START,'action':ACTION}:raise ValueError('root authorization identity/action differs')
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--authorization',type=Path,required=True);a=p.parse_args();auth(a.authorization)
 client=o.load_sdk();provider=o.provider(client);assert provider['job_id']==JOB and provider['provider_run_status']=='RUNNING'
 o.REMOTE=REMOTE
 def runner(argv,**kwargs):
  if argv and argv[0]=='ssh':kwargs['timeout']=960
  return subprocess.run(argv,**kwargs)
 result=o.ssh_read(client,provider['cluster_id'],runner=runner)
 if result.get('run_id')!=RUN or result.get('sidecar_pid')!=PID or result.get('sidecar_start_tick')!=START:raise RuntimeError('remote result identity differs')
 print(json.dumps({'provider':provider,'remote':result},sort_keys=True))
if __name__=='__main__':main()
