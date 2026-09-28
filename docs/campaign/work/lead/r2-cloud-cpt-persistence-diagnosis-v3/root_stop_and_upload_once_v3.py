#!/usr/bin/env python3
"""Root-authorized exact-sidecar stop and one checkpoint-317 upload diagnostic v2."""
import argparse,base64,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'r2-cloud-cpt-live-observer-v1';sys.path.insert(0,str(BASE));import live_observer as o
JOB=o.JOB_ID;RUN=o.RUN_ID;PID=5498;START=18486
ACTION='stop_exact_sidecar_then_upload_checkpoint317_once_v2'
CHILD='''import json,os,re,sys,traceback
from pathlib import Path
payload,binding,run,checkpoint,diagnostic=sys.argv[1:]
sys.path.insert(0,payload)
import artifact_upload as uploader
SECRET=re.compile(r'hf_[A-Za-z0-9]+')
def safe(value):
 text=SECRET.sub('[REDACTED]',str(value))
 text=re.sub(r'(?i)(bearer\\s+)[A-Za-z0-9._~+/=-]+',r'\\1[REDACTED]',text)
 text=re.sub(r'(?i)(https?://[^?\\s"\\\']+)\\?[^\\s"\\\']+',r'\\1?[REDACTED_QUERY]',text)
 text=re.sub(r'(?i)([?&](?:token|sig|signature|credential|key|x-amz-[^=]+)=)[^&\\s]+',r'\\1[REDACTED]',text)
 text=re.sub(r'(://)[^/@\\s:]+:[^/@\\s]+@',r'\\1[REDACTED]@',text)
 return text[-1024:]
def stack(error):
 return [{'file':Path(row.filename).name,'line':row.lineno,'function':row.name} for row in traceback.extract_tb(error.__traceback__)[-6:]]
try:
 value=uploader.upload_checkpoint(Path(run),json.loads(Path(binding).read_text()),Path(checkpoint))
 result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v2','status':'success','step':317,'checkpoint_revision':value.get('revision'),'receipt_revision':value.get('receipt_revision'),'credential_persisted':False}
except Exception as error:
 result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v2','status':'failed','step':317,'error_type':type(error).__name__,'error_message':safe(error),'stack':stack(error),'credential_persisted':False}
Path(diagnostic).parent.mkdir(parents=True,exist_ok=True)
Path(diagnostic).write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n')
print(json.dumps(result,sort_keys=True),flush=True)
raise SystemExit(0 if result['status']=='success' else 1)
'''
REMOTE_TEMPLATE=r'''import hashlib,json,os,re,signal,subprocess,sys,time,base64
run=sys.argv[1];pid=5498;start=18486
CHILD=base64.b64decode('__CHILD_B64__').decode('utf-8')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for block in iter(lambda:stream.read(4194304),b''):h.update(block)
 return h.hexdigest()
def safe(value):
 text=re.sub(r'hf_[A-Za-z0-9]+','[REDACTED]',str(value));text=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',r'\1[REDACTED]',text);text=re.sub(r'(?i)(https?://[^?\s"\']+)\?[^\s"\']+',r'\1?[REDACTED_QUERY]',text);text=re.sub(r'(?i)([?&](?:token|sig|signature|credential|key|x-amz-[^=]+)=)[^&\s]+',r'\1[REDACTED]',text);return text[-1024:]
def abort(reason,error_type='PreconditionError'):
 print(json.dumps({'run_id':run,'run_directory_match_count':1,'sidecar_stopped':False,'upload_result':{'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v2','status':'precondition_failed','error_type':error_type,'error_message':safe(reason),'credential_persisted':False}}));raise SystemExit(0)
base='/proc/'+str(pid)
try:raw=open(base+'/cmdline','rb').read();parts=[x.decode() for x in raw.split(b'\0') if x];stat=open(base+'/stat').read().split()
except Exception as error:abort(error,type(error).__name__)
if int(stat[21])!=start or int(stat[3])!=4588:abort('exact sidecar process identity differs')
if len(parts)!=6 or not os.path.isabs(parts[0]) or os.path.basename(parts[1])!='checkpoint_sidecar.py' or parts[4]!='--stop-file' or run not in parts[3] or run not in parts[5]:abort('exact sidecar argv differs')
python=parts[0];sidecar=os.path.realpath(parts[1]);binding=os.path.realpath(parts[2]);root=os.path.realpath(parts[3]);payload=os.path.dirname(sidecar);checkpoint=root+'/artifacts/archive/full/checkpoint-317';diagnostic=root+'/artifacts/cloud-persistence/checkpoint-317-upload-diagnostic-v2.json'
if not os.path.isfile(python):abort('managed interpreter argv path absent')
if sha(sidecar)!='0b3d4df5e4a0ffd5ea679ad58fe709a0b2126230cdffa4b7f3c77719c6400511' or sha(payload+'/artifact_upload.py')!='a055094b5ab39c35e10b32acb4938ff56537946a6e55ed8785612360b6aaf146' or sha(payload+'/cloud_contract.py')!='2b345761d5e663e1bf9021856161f362f539e351e414005c39cf47f3952f256a':abort('active payload source differs')
if sha(binding)!='c31671d5e3895a394ee569b0d0537ebe81d636daa2dbbaad4a11fa6559d76dc5' or not os.path.isdir(checkpoint) or os.path.exists(diagnostic):abort('binding/checkpoint/fresh diagnostic precondition differs')
envraw=open(base+'/environ','rb').read().split(b'\0');matches=[x.split(b'=',1)[1].decode() for x in envraw if x.startswith(b'HF_TOKEN=')]
if len(matches)!=1 or not matches[0].startswith('hf_'):abort('single sidecar token absent')
token=matches[0]
child_env={'HF_TOKEN':token,'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':os.environ.get('HOME','/home/ray'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HOME':root+'/hf-cache'}
probe_code="import json,sys;sys.path.insert(0,sys.argv[1]);import artifact_upload,huggingface_hub;print(json.dumps({'status':'managed_import_pass'}))"
try:probe=subprocess.run([python,'-I','-c',probe_code,payload],env={k:v for k,v in child_env.items() if k!='HF_TOKEN'},capture_output=True,text=True,timeout=30)
except Exception as error:abort(error,type(error).__name__)
if probe.returncode!=0 or 'managed_import_pass' not in probe.stdout:abort(probe.stderr or probe.stdout or 'managed import probe failed','ManagedImportProbeError')
os.kill(pid,signal.SIGTERM)
for _ in range(100):
 try:current=open(base+'/stat').read().split();alive=int(current[21])==start and current[2]!='Z'
 except (OSError,ValueError):alive=False
 if not alive:break
 time.sleep(.1)
if alive:abort('exact sidecar did not stop after SIGTERM','SidecarStopError')
try:got=subprocess.run([python,'-I','-c',CHILD,payload,binding,root,checkpoint,diagnostic],env=child_env,capture_output=True,text=True,timeout=900)
except Exception as error:
 result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v2','status':'wrapper_failed','error_type':type(error).__name__,'error_message':safe(error),'credential_persisted':False};print(json.dumps({'run_id':run,'run_directory_match_count':1,'sidecar_stopped':True,'sidecar_pid':pid,'sidecar_start_tick':start,'upload_result':result}));raise SystemExit(0)
token=None;matches=None;envraw=None;child_env=None
try:result=json.loads(got.stdout.strip().splitlines()[-1])
except Exception:result={'schema':'sepalith.cloud-cpt.checkpoint-upload-diagnostic.v2','status':'wrapper_failed','error_type':'ChildOutputError','error_message':safe(got.stderr),'credential_persisted':False}
print(json.dumps({'run_id':run,'run_directory_match_count':1,'sidecar_stopped':True,'sidecar_pid':pid,'sidecar_start_tick':start,'managed_import_probe':'pass','upload_result':result,'child_exit_code':got.returncode},sort_keys=True))'''
REMOTE=REMOTE_TEMPLATE.replace('__CHILD_B64__',base64.b64encode(CHILD.encode()).decode())
def auth(path):
 value=json.loads(path.read_text());expected={'schema':'sepalith.cloud-cpt.persistence-repair-root-authorization.v2','authorized':True,'job_id':JOB,'run_id':RUN,'sidecar_pid':PID,'sidecar_start_tick':START,'action':ACTION}
 if value!=expected:raise ValueError('root authorization identity/action differs')
 return value
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',type=Path,required=True);args=parser.parse_args();auth(args.authorization)
 client=o.load_sdk();provider=o.provider(client)
 if provider['job_id']!=JOB or provider['provider_run_status']!='RUNNING':raise ValueError('owned provider runtime differs')
 o.REMOTE=REMOTE
 def runner(argv,**kwargs):
  if argv and argv[0]=='ssh':kwargs['timeout']=960
  return subprocess.run(argv,**kwargs)
 result=o.ssh_read(client,provider['cluster_id'],runner=runner)
 if result.get('run_id')!=RUN:raise RuntimeError('remote run identity differs')
 print(json.dumps({'provider':provider,'remote':result},sort_keys=True))
if __name__=='__main__':main()
