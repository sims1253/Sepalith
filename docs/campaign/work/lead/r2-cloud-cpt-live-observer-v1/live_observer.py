#!/usr/bin/env python3
"""Read-only, allowlisted observation of the owned cloud CPT job."""
import argparse,base64,datetime,hashlib,json,os,re,shlex,subprocess,sys,tempfile,time
from pathlib import Path

JOB_ID='prodjob_neqwkumat4i3rii5yl8af241ks'
RUN_ID='4f899bbc6e9d46c0a88985d64e6d40e2'
JOB_NAME='sepalith-cpt-'+RUN_ID
ARTIFACT_REPO='scholzmx/sepalith-lora'
ARTIFACT_PREFIX='r2-cpt/'+RUN_ID
SDK='/home/m0hawk/.local/share/uv/tools/anyscale/lib/python3.14/site-packages'
EVENT_SCHEMA='sepalith.cloud-cpt.training-event.v3'
ID=re.compile(r'^(prodjob|job|ses)_[a-z0-9]+$')
REMOTE=r'''import glob,json,os,sys
run=sys.argv[1]
paths=glob.glob('/mnt/local_storage/sepalith-cpt-'+run+'-*')
if len(paths)!=1: raise SystemExit('owned run directory count differs')
p=paths[0];out={'run_id':run,'run_directory_match_count':1}
s=os.path.join(p,'artifacts','staging-receipt.json')
if os.path.isfile(s):
 x=json.load(open(s));out['staging']={'status':x.get('status'),'revision':x.get('revision'),'files_count':len(x.get('files',[])),'credential_persisted':x.get('credential_persisted'),'receipt_sha256':__import__('hashlib').sha256(open(s,'rb').read()).hexdigest()}
t=os.path.join(p,'artifacts','training','telemetry.jsonl');records=[]
if os.path.isfile(t):
 for line in open(t):
  try:x=json.loads(line)
  except Exception:continue
  if x.get('event') not in ('train_begin','optimizer_step','trainer_metrics','train_end'):continue
  y={k:x[k] for k in ('event','at','step','seconds','remaining_seconds','stop_reason') if k in x}
  if isinstance(x.get('metrics'),dict):y['metrics']={k:x['metrics'][k] for k in ('loss','learning_rate','grad_norm','epoch','train_runtime','train_loss') if k in x['metrics'] and isinstance(x['metrics'][k],(int,float,str,bool,type(None)))}
  if isinstance(x.get('resources'),dict):y['resources']={k:x['resources'][k] for k in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_attempt_peak_allocated_bytes','cuda_attempt_peak_reserved_bytes','host_floor_gib','process_peak_rss_bytes') if k in x['resources'] and isinstance(x['resources'][k],(int,float))}
  records.append(y)
optimizers=[x for x in records if x['event']=='optimizer_step'];metrics=[x for x in records if x['event']=='trainer_metrics']
out['telemetry']={'exists':os.path.isfile(t),'records':len(records),'optimizer_steps':len(optimizers),'first_optimizer_record':optimizers[0] if optimizers else None,'first_trainer_metrics':metrics[0] if metrics else None,'last_records':records[-8:]}
l=os.path.join(p,'training.log');out['training_log']={'exists':os.path.isfile(l),'bytes':os.stat(l).st_size if os.path.isfile(l) else None,'mtime_ns':os.stat(l).st_mtime_ns if os.path.isfile(l) else None}
print(json.dumps(out,sort_keys=True))'''

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(value,message):
 if not value:raise ValueError(message)
def scalar(value):return value if isinstance(value,(str,int,float,bool,type(None))) else str(value)
def load_sdk():
 if SDK not in sys.path:sys.path.insert(0,SDK)
 import anyscale
 return anyscale.Anyscale()._anyscale_client
def provider(client):
 job=client.get_job(job_id=JOB_ID,name=None,cloud=None,project=None)
 require(job and job.id==JOB_ID and job.name==JOB_NAME,'owned provider job identity differs')
 state=job.state;cluster=state.cluster_id;require(ID.fullmatch(cluster or ''),'owned cluster identity absent')
 runs=client.get_job_runs(JOB_ID,max_runs_per_job=10);require(len(runs)==1,'owned job run count differs')
 run=runs[0];require(run.cluster_id==cluster and ID.fullmatch(run.id or ''),'owned run/cluster identity differs')
 raw=client.logs_for_job_run(run.id,max_lines=5000,parse_json=False);events=[]
 for line in raw.splitlines():
  try:start=line.find('{');x=json.loads(line[start:]) if start>=0 else None
  except (ValueError,TypeError):continue
  if not isinstance(x,dict) or x.get('schema')!=EVENT_SCHEMA:continue
  events.append({k:scalar(x[k]) for k in ('at','phase','status','exit_code','error_type','memory_floor_bytes','revision') if k in x})
 return {'job_id':JOB_ID,'job_name':JOB_NAME,'job_state':str(state.current_state),'job_status_updated_at':scalar(getattr(job,'status_updated_at',None)),'provider_run_id':run.id,'provider_run_status':str(run.status),'cluster_id':cluster,'provider_log_lines':len(raw.splitlines()),'phase_event_count':len(events),'last_phase_events':events[-12:]}
def remote_command():
 encoded=base64.b64encode(REMOTE.encode()).decode()
 return 'python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+encoded+"'))")+' '+shlex.quote(RUN_ID)
def ssh_failure(stderr):
 text=(stderr or '').lower()
 for phrase,label in (('permission denied','authentication_rejected'),('connection timed out','connection_timeout'),('operation timed out','connection_timeout'),('no route to host','no_route'),('connection refused','connection_refused'),('could not resolve','name_resolution_failed'),('host key verification failed','host_key_failed')):
  if phrase in text:return label
 return 'ssh_exit_nonzero'
def ssh_read(client,cluster,runner=subprocess.run):
 host=client.get_cluster_head_node_ip(cluster);key=client.get_cluster_ssh_key(cluster).private_key
 require(isinstance(host,str) and host and isinstance(key,str) and 'PRIVATE KEY' in key,'owned SSH material absent')
 with tempfile.TemporaryDirectory(prefix='cpt-observer-agent-') as folder:
  socket=str(Path(folder)/'agent.sock');agent=subprocess.Popen(['ssh-agent','-D','-a',socket],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   for _ in range(100):
    if Path(socket).exists():break
    time.sleep(.05)
   require(Path(socket).exists(),'transient SSH agent unavailable');env=dict(os.environ,SSH_AUTH_SOCK=socket)
   added=runner(['ssh-add','-'],input=key.encode(),env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
   require(added.returncode==0,'transient SSH key rejected')
   proxy='ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=12 -W %h:%p ubuntu@'+host
   argv=['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=no','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','LogLevel=ERROR','-o','ConnectTimeout=12','-o','ProxyCommand='+proxy,'-p','5020','ray@0.0.0.0',remote_command()]
   got=runner(argv,env=env,capture_output=True,text=True,timeout=40)
   if got.returncode!=0:return {'status':'unavailable','reason':ssh_failure(got.stderr),'exit_code':got.returncode,'credential_stored':False}
   value=json.loads(got.stdout);require(value.get('run_id')==RUN_ID and value.get('run_directory_match_count')==1,'remote owned-run identity differs')
   return value
  finally:
   agent.terminate()
   try:agent.wait(timeout=3)
   except subprocess.TimeoutExpired:agent.kill()
def private_artifacts():
 token=os.environ.get('HF_TOKEN');require(bool(token),'private read token absent')
 from huggingface_hub import HfApi
 api=HfApi(token=token);info=api.repo_info(ARTIFACT_REPO,repo_type='model');require(info.private is True,'artifact repository is not private')
 rows=[]
 for x in api.list_repo_tree(repo_id=ARTIFACT_REPO,repo_type='model',path_in_repo=ARTIFACT_PREFIX,recursive=True,expand=True):
  if not hasattr(x,'size'):continue
  lfs=getattr(x,'lfs',None);digest=getattr(lfs,'sha256',None) if not isinstance(lfs,dict) else lfs.get('sha256')
  rows.append({'path':x.path,'bytes':x.size,'sha256':digest,'git_blob_sha1':getattr(x,'blob_id',None)})
 return {'repo':ARTIFACT_REPO,'prefix':ARTIFACT_PREFIX,'private':True,'head_revision':info.sha,'files':sorted(rows,key=lambda x:x['path'])}
def observe(use_ssh=True,use_hf=True,client=None):
 client=client or load_sdk();result={'schema':'sepalith.cloud-cpt.live-observation.v1','observed_at':now(),'run_id':RUN_ID,'provider':provider(client)}
 result['private_artifacts']=private_artifacts() if use_hf else {'status':'not_requested'}
 result['remote']=ssh_read(client,result['provider']['cluster_id']) if use_ssh else {'status':'not_requested'}
 telemetry=result['remote'].get('telemetry',{})
 result['first_optimizer_update_observed']=bool(telemetry.get('optimizer_steps',0)>0)
 result['interpretation']='optimizer evidence present' if result['first_optimizer_update_observed'] else 'provider RUNNING/training START does not prove an optimizer update'
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--no-ssh',action='store_true');p.add_argument('--no-hf',action='store_true');a=p.parse_args()
 require(not a.output.exists(),'fresh output required');value=observe(not a.no_ssh,not a.no_hf);a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(a.output),'first_optimizer_update_observed':value['first_optimizer_update_observed'],'provider_status':value['provider']['provider_run_status']}))
if __name__=='__main__':main()
