#!/usr/bin/env python3
"""Root-only controller for one opaque upload or the bounded monitor."""
import argparse,base64,hashlib,json,os,shlex,subprocess,sys,tempfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OBSERVER=HERE.parent/'r2-cloud-cpt-live-observer-v1';sys.path.insert(0,str(OBSERVER));import live_observer as o
JOB='prodjob_neqwkumat4i3rii5yl8af241ks';RUN='4f899bbc6e9d46c0a88985d64e6d40e2';PARENT_PID=4588;PARENT_START=7757;WORKING_DIR='s3_dc963855898b4fa1797b256f71b8711bd61b633c'
BINDING_SHA='c31671d5e3895a394ee569b0d0537ebe81d636daa2dbbaad4a11fa6559d76dc5'
PAYLOAD_HASHES={'cloud_entry.py':'dc0b4d0cc10afc98d8d7a08dbb2729c062f757c99ca5ef4dc0a7fe624c7bd12f','artifact_upload.py':'a055094b5ab39c35e10b32acb4938ff56537946a6e55ed8785612360b6aaf146','checkpoint_sidecar.py':'0b3d4df5e4a0ffd5ea679ad58fe709a0b2126230cdffa4b7f3c77719c6400511'}
REMOTE=r'''import base64,hashlib,json,os,subprocess,sys
run,mode,pid,parent_start,working_dir,binding_sha,source_b64,hashes_json=sys.argv[1:]
pid=int(pid);parent_start=int(parent_start);hashes=json.loads(hashes_json)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def abort(m):print(json.dumps({'status':'precondition_failed','error_message':str(m)[-512:],'credential_persisted':False}));raise SystemExit(0)
paths=__import__('glob').glob('/mnt/local_storage/sepalith-cpt-'+run+'-*')
if len(paths)!=1:abort('owned run directory count differs')
root=os.path.realpath(paths[0]);base='/proc/'+str(pid)
try:parts=[x.decode() for x in open(base+'/cmdline','rb').read().split(b'\0') if x];stat=open(base+'/stat').read().split()
except Exception as e:abort(type(e).__name__)
cwd=os.path.realpath(base+'/cwd');expected_entry=cwd+'/payload/cloud_entry.py'
if stat[2]=='Z' or int(stat[21])!=parent_start or os.path.basename(cwd)!=working_dir or parts!=['python','-B',expected_entry,'binding.root.json']:abort('owned cloud entry process identity differs')
payload=root+'/payload';binding=root+'/binding.root.json';submission_payload=cwd+'/payload';submission_binding=cwd+'/binding.root.json'
if sha(binding)!=binding_sha or sha(submission_binding)!=binding_sha:abort('active binding differs')
for name,digest in hashes.items():
 if sha(payload+'/'+name)!=digest or sha(submission_payload+'/'+name)!=digest:abort('active payload source differs: '+name)
python=root+'/venv/bin/python';
if not os.path.isfile(python):abort('managed venv interpreter absent')
probe=subprocess.run([python,'-I','-c','import huggingface_hub,tarfile;print("ok")'],capture_output=True,text=True,timeout=30)
if probe.returncode or probe.stdout.strip()!='ok':abort('managed persistence import probe failed')
token=sys.stdin.readline().strip()
if not token.startswith('hf_'):abort('forwarded private token absent')
scratch=root+'/persistence-staging';os.makedirs(scratch,exist_ok=True);source=base64.b64decode(source_b64)
source_path=scratch+'/tar_persistence_v2.py'
if os.path.exists(source_path):
 if hashlib.sha256(open(source_path,'rb').read()).digest()!=hashlib.sha256(source).digest():abort('existing tar persistence source differs')
else:
 fd=os.open(source_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 with os.fdopen(fd,'wb') as f:f.write(source);f.flush();os.fsync(f.fileno())
env={'HF_TOKEN':token,'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':os.environ.get('HOME','/home/ray'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HOME':root+'/hf-cache'}
argv=[python,'-I',source_path,mode,root]+(['--step','317'] if mode=='once' else [])
if mode=='once':
 got=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=1800);line=(got.stdout.strip().splitlines() or ['{}'])[-1]
 try:result=json.loads(line)
 except Exception:result={'status':'wrapper_failed','error_type':'ChildOutputError','error_message':got.stderr[-512:]}
 print(json.dumps({'status':'remote_once_finished','run_id':run,'parent_pid':pid,'parent_start_tick':int(stat[21]),'working_dir_basename':working_dir,'child_exit_code':got.returncode,'result':result,'credential_persisted':False},sort_keys=True))
else:
 marker=scratch+'/monitor-start.json'
 if os.path.exists(marker):abort('monitor start marker already exists')
 log=open(scratch+'/monitor.log','ab',buffering=0);child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=int(open('/proc/'+str(child.pid)+'/stat').read().split()[21])
 value={'schema':'sepalith.cloud-cpt.tar-persistence-monitor-start.v1','run_id':run,'pid':child.pid,'start_tick':start,'source_sha256':hashlib.sha256(source).hexdigest(),'deadline':'2026-09-14T18:40:37.166121+00:00','credential_persisted':False}
 fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
 with os.fdopen(fd,'wb') as f:f.write((json.dumps(value,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'monitor_started','run_id':run,**value},sort_keys=True))
'''
def auth(path,mode):
 value=json.loads(path.read_text());expected={'schema':'sepalith.cloud-cpt.tar-persistence-monitor-root-authorization.v2','authorized':True,'job_id':JOB,'run_id':RUN,'parent_pid':PARENT_PID,'parent_start_tick':PARENT_START,'action':'launch_bounded_tar_monitor_v2'}
 if value!=expected:raise ValueError('root authorization identity/action differs')
def remote_command(mode):
 source=base64.b64encode((HERE/'tar_persistence_v2.py').read_bytes()).decode();code=base64.b64encode(REMOTE.encode()).decode()
 return 'python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+code+"'))")+' '+ ' '.join(shlex.quote(x) for x in (RUN,mode,str(PARENT_PID),str(PARENT_START),WORKING_DIR,BINDING_SHA,source,json.dumps(PAYLOAD_HASHES,sort_keys=True)))
def ssh(client,cluster,mode,token):
 host=client.get_cluster_head_node_ip(cluster);key=client.get_cluster_ssh_key(cluster).private_key
 if not (isinstance(host,str) and host and isinstance(key,str) and 'PRIVATE KEY' in key):raise ValueError('owned SSH material absent')
 with tempfile.TemporaryDirectory(prefix='cpt-tar-agent-') as folder:
  socket=str(Path(folder)/'agent.sock');agent=subprocess.Popen(['ssh-agent','-D','-a',socket],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  try:
   for _ in range(100):
    if Path(socket).exists():break
    time.sleep(.05)
   env=dict(os.environ,SSH_AUTH_SOCK=socket);added=subprocess.run(['ssh-add','-'],input=key.encode(),env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
   if added.returncode:raise ValueError('transient SSH key rejected')
   proxy='ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=12 -W %h:%p ubuntu@'+host
   argv=['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=no','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','LogLevel=ERROR','-o','ConnectTimeout=12','-o','ProxyCommand='+proxy,'-p','5020','ray@0.0.0.0',remote_command(mode)]
   got=subprocess.run(argv,input=token+'\n',env=env,capture_output=True,text=True,timeout=1860 if mode=='once' else 90)
   if got.returncode:raise RuntimeError('remote controller exit '+str(got.returncode)+': '+o.ssh_failure(got.stderr))
   return json.loads(got.stdout.strip().splitlines()[-1])
  finally:
   agent.terminate()
   try:agent.wait(timeout=3)
   except subprocess.TimeoutExpired:agent.kill()
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['monitor']);p.add_argument('--authorization',type=Path,required=True);a=p.parse_args();auth(a.authorization,a.mode)
 token=os.environ.get('HF_TOKEN');
 if not token or not token.startswith('hf_'):raise ValueError('local private token absent')
 client=o.load_sdk();provider=o.provider(client)
 if provider['job_id']!=JOB or provider['provider_run_status']!='RUNNING':raise ValueError('owned provider runtime differs')
 result=ssh(client,provider['cluster_id'],a.mode,token);token=None
 print(json.dumps({'provider':provider,'remote':result},sort_keys=True))
if __name__=='__main__':main()
