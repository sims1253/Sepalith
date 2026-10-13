"""Private sentinel, durable full-checkpoint, and terminal artifact upload."""
import argparse, hashlib, json, logging, os, re, secrets, traceback
from pathlib import Path
from cloud_contract import require,sha,stamp,write
logging.disable(logging.CRITICAL)
SECRET=re.compile(r'hf_[A-Za-z0-9]+')
def safe_message(error):
 text=SECRET.sub('[REDACTED]',str(error));text=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',r'\1[REDACTED]',text);return text[-1024:]
def safe_stack(error):
 return [{'file':Path(row.filename).name,'line':row.lineno,'function':row.name} for row in traceback.extract_tb(error.__traceback__)[-6:]]

def hashes(p):
 p=Path(p);size=p.stat().st_size;git=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):git.update(block)
 return {'bytes':size,'sha256':sha(p),'git_blob_sha1':git.hexdigest()}
def inventory(folder,exclude=()):
 out={}
 for p in sorted(Path(folder).rglob('*')):
  require(not p.is_symlink(),'artifact symlink forbidden')
  name=str(p.relative_to(folder))
  if any(name.startswith(prefix) for prefix in exclude):continue
  if p.is_file():out[name]=hashes(p)
 return out
def validate_remote(api,repo,prefix,revision,local):
 remote={}
 for row in api.list_repo_tree(repo_id=repo,repo_type='model',path_in_repo=prefix,revision=revision,recursive=True,expand=True):
  if not hasattr(row,'size'):continue
  name=row.path.removeprefix(prefix+'/');lfs=getattr(row,'lfs',None);remote[name]={'bytes':row.size,'sha256':getattr(lfs,'sha256',None) if not isinstance(lfs,dict) else lfs.get('sha256'),'git_blob_sha1':getattr(row,'blob_id',None)}
 for name,expected in local.items():
  require(name in remote and remote[name]['bytes']==expected['bytes'],'remote artifact size differs')
  require(remote[name].get('sha256')==expected['sha256'] if remote[name].get('sha256') else remote[name].get('git_blob_sha1')==expected['git_blob_sha1'],'remote artifact hash differs')

def checkpoint_complete(folder):
 p=Path(folder);m=p/'campaign-manifest.json'
 if not m.is_file():return None
 try:spec=json.loads(m.read_text())
 except (OSError,ValueError):return None
 files=spec.get('files')
 required={'adapter_config.json','adapter_model.safetensors','campaign-state.json','optimizer.pt','rng_state.pth','scheduler.pt','trainer_state.json'}
 if spec.get('full') is not True or type(spec.get('step')) is not int or not isinstance(files,dict) or not files or not required.issubset(files):return None
 for name,row in files.items():
  q=p/name
  if not q.is_file() or q.stat().st_size!=row['bytes'] or sha(q)!=row['sha256']:return None
 return spec

def publish_once(path,value,label):
 path=Path(path)
 if path.exists():
  try:current=json.loads(path.read_text())
  except (OSError,ValueError) as error:raise ValueError(f'existing {label} is unreadable') from error
  require(current==value,f'existing {label} differs')
 else:write(path,value)
 return value

def client():
 token=os.environ.get('HF_TOKEN');require(bool(token),'private upload token absent')
 from huggingface_hub import HfApi,hf_hub_download
 return HfApi(token=token),hf_hub_download,token

def upload_checkpoint(run,b,folder):
 spec=checkpoint_complete(folder);require(spec is not None,'checkpoint is incomplete')
 step=spec['step'];require(step%317==0 and 0<step<=1902,'checkpoint step differs')
 api,download,token=client();repo=b['artifact_repo'];prefix=b['artifact_prefix']+f'/checkpoints/checkpoint-{step}'
 files=inventory(folder);local=run/f'artifacts/cloud-persistence/checkpoint-{step}.json'
 if local.exists():
  receipt=json.loads(local.read_text());require(receipt.get('schema')=='sepalith.cloud-cpt.remote-checkpoint.v1' and receipt.get('step')==step and receipt.get('prefix')==prefix,'existing checkpoint receipt identity differs')
  require(receipt.get('files')==files and receipt.get('campaign_manifest_sha256')==sha(Path(folder)/'campaign-manifest.json'),'existing checkpoint receipt inventory differs')
  require(receipt.get('schedule_sha256')==json.loads((Path(folder)/'campaign-state.json').read_text())['sampler']['schedule_sha256'] and receipt.get('consumed_draws')==step*16,'existing checkpoint receipt state differs')
  revision=receipt.get('revision');require(isinstance(revision,str) and len(revision)>=40,'existing checkpoint revision missing')
  validate_remote(api,repo,prefix,revision,files)
 else:
  commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=folder,path_in_repo=prefix,commit_message=f'CPT full checkpoint {step}');revision=commit.oid;validate_remote(api,repo,prefix,revision,files)
  receipt={'schema':'sepalith.cloud-cpt.remote-checkpoint.v1','step':step,'revision':revision,'prefix':prefix,'files':files,'campaign_manifest_sha256':sha(Path(folder)/'campaign-manifest.json'),'schedule_sha256':json.loads((Path(folder)/'campaign-state.json').read_text())['sampler']['schedule_sha256'],'consumed_draws':step*16}
  publish_once(local,receipt,'checkpoint receipt')
 remote=download(repo,filename=prefix+'/campaign-manifest.json',revision=revision,repo_type='model',token=token,local_dir=run/'readback')
 require(sha(remote)==sha(Path(folder)/'campaign-manifest.json'),'remote checkpoint manifest differs')
 inventory_path=b['artifact_prefix']+f'/checkpoint-receipts/checkpoint-{step}.json'
 if api.file_exists(repo,inventory_path,repo_type='model'):
  receipt_revision=api.repo_info(repo,repo_type='model').sha
  back=download(repo,filename=inventory_path,revision=receipt_revision,repo_type='model',token=token,local_dir=run/'receipt-readback')
  require(sha(back)==sha(local),'existing remote checkpoint receipt differs')
 else:
  final=api.upload_file(repo_id=repo,repo_type='model',path_or_fileobj=local,path_in_repo=inventory_path,commit_message=f'CPT checkpoint {step} receipt');receipt_revision=final.oid
  back=download(repo,filename=inventory_path,revision=receipt_revision,repo_type='model',token=token,local_dir=run/'receipt-readback')
  require(sha(back)==sha(local),'remote checkpoint receipt differs')
 commit_record={'step':step,'checkpoint_revision':revision,'receipt_revision':receipt_revision,'inventory_path':inventory_path}
 publish_once(run/f'artifacts/cloud-persistence/checkpoint-{step}-commit.json',commit_record,'checkpoint commit record')
 return dict(receipt,receipt_revision=receipt_revision)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['sentinel','checkpoint','final']);p.add_argument('binding',type=Path);p.add_argument('run',type=Path);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
 b=json.loads(a.binding.read_text());api,download,token=client();require(api.repo_info(b['artifact_repo'],repo_type='model').private is True,'artifact repository is not private')
 if a.mode=='sentinel':
  remote=b['artifact_prefix']+'/sentinel.json';require(not api.file_exists(b['artifact_repo'],remote,repo_type='model'),'artifact prefix already used')
  q=a.run/'sentinel.json';write(q,{'run_id':b['run_id'],'nonce':secrets.token_hex(32)})
  c=api.upload_file(repo_id=b['artifact_repo'],repo_type='model',path_or_fileobj=q,path_in_repo=remote,commit_message='CPT private persistence probe')
  back=download(b['artifact_repo'],filename=remote,revision=c.oid,repo_type='model',token=token,local_dir=a.run/'readback');require(sha(back)==sha(q),'sentinel readback differs');return
 if a.mode=='checkpoint':upload_checkpoint(a.run,b,a.checkpoint);return
 folder=a.run/'artifacts';files=inventory(folder,exclude=('archive/','training/trainer/checkpoint-'));manifest=folder/'cloud-artifact-manifest.json';write(manifest,{'schema':1,'files':files,'credential_persisted':False});files['cloud-artifact-manifest.json']=hashes(manifest)
 terminal_prefix=b['artifact_prefix']+'/artifacts';c=api.upload_folder(repo_id=b['artifact_repo'],repo_type='model',folder_path=folder,path_in_repo=terminal_prefix,ignore_patterns=['archive/**','training/trainer/checkpoint-*/**'],commit_message='CPT terminal artifacts');validate_remote(api,b['artifact_repo'],terminal_prefix,c.oid,files)
 receipt={'schema':'sepalith.cloud-cpt.persistence.v1','at':stamp(),'status':'terminal_artifacts_uploaded','revision':c.oid,'prefix':b['artifact_prefix'],'local_files':files,'independent_root_readback_required':True}
 q=a.run/'persistence-receipt.json';write(q,receipt);final=api.upload_file(repo_id=b['artifact_repo'],repo_type='model',path_or_fileobj=q,path_in_repo=b['artifact_prefix']+'/persistence-receipt.json',commit_message='CPT persistence receipt')
 back=download(b['artifact_repo'],filename=b['artifact_prefix']+'/persistence-receipt.json',revision=final.oid,repo_type='model',token=token,local_dir=a.run/'readback');require(sha(back)==sha(q),'persistence receipt readback differs')
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'status':'artifact_operation_failed','error_type':type(e).__name__,'error_message':safe_message(e),'stack':safe_stack(e)}),flush=True);raise SystemExit(1)
