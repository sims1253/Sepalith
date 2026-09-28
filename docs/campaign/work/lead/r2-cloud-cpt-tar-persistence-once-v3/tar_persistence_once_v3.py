#!/usr/bin/env python3
"""Opaque, bounded tar persistence for full CPT checkpoints."""
import argparse,datetime,hashlib,json,os,posixpath,re,tarfile,time,traceback
from pathlib import Path,PurePosixPath

RUN_ID='4f899bbc6e9d46c0a88985d64e6d40e2'
REPO='scholzmx/sepalith-lora'
PREFIX='r2-cpt/'+RUN_ID
STEPS=(317,634,951,1268,1585,1902)
DEADLINE='2026-09-14T18:40:37.166121+00:00'
REQUIRED={'adapter_config.json','adapter_model.safetensors','campaign-state.json','optimizer.pt','rng_state.pth','scheduler.pt','trainer_state.json'}
SECRET=re.compile(r'hf_[A-Za-z0-9]+')

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(v,m):
 if not v: raise ValueError(m)
def safe(error):
 s=SECRET.sub('[REDACTED]',str(error));s=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+',r'\1[REDACTED]',s)
 s=re.sub(r'(?i)(https?://[^?\s"\']+)\?[^\s"\']+',r'\1?[REDACTED_QUERY]',s)
 s=re.sub(r'(?i)([?&](?:token|sig|signature|credential|key|x-amz-[^=]+)=)[^&\s]+',r'\1[REDACTED]',s)
 return s[-1024:]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def git_blob_sha(path):
 path=Path(path);h=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 return path
def event(scratch,status,**fields):
 row={'schema':'sepalith.cloud-cpt.tar-persistence-event.v1','at':now(),'status':status,**fields}
 p=Path(scratch)/'events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 return row
def safe_name(name):
 p=PurePosixPath(name)
 return bool(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and posixpath.normpath(name)==name)
def file_inventory(folder):
 folder=Path(folder);out={}
 for p in sorted(folder.rglob('*')):
  require(not p.is_symlink(),'checkpoint symlink forbidden')
  if p.is_dir():continue
  require(p.is_file(),'checkpoint non-regular entry forbidden')
  name=p.relative_to(folder).as_posix();require(safe_name(name),'unsafe checkpoint path')
  out[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 return out
def validate_checkpoint(folder,step):
 folder=Path(folder);require(step in STEPS and folder.name==f'checkpoint-{step}','checkpoint step/path differs')
 m=folder/'campaign-manifest.json';require(m.is_file(),'campaign manifest absent');spec=json.loads(m.read_text())
 require(spec.get('full') is True and spec.get('step')==step,'campaign manifest full/step differs')
 files=spec.get('files');require(isinstance(files,dict) and files and REQUIRED.issubset(files),'campaign manifest file set incomplete')
 inventory=file_inventory(folder);require(set(inventory)==set(files)|{'campaign-manifest.json'},'checkpoint inventory differs from manifest')
 for name,row in files.items():require(inventory[name]=={'bytes':row['bytes'],'sha256':row['sha256']},'checkpoint file hash differs')
 state=json.loads((folder/'campaign-state.json').read_text());schedule=state['sampler']['schedule_sha256']
 require(isinstance(schedule,str) and len(schedule)==64,'schedule identity absent')
 return spec,inventory,schedule
def build_tar(folder,step,scratch):
 folder=Path(folder);scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
 spec,inventory,schedule=validate_checkpoint(folder,step);target=scratch/f'checkpoint-{step}.tar'
 if not target.exists():
  tmp=target.with_suffix('.tar.tmp')
  with tarfile.open(tmp,'w',format=tarfile.PAX_FORMAT) as tf:
   for name in inventory:tf.add(folder/name,arcname=f'checkpoint-{step}/{name}',recursive=False)
  os.replace(tmp,target)
 expected={f'checkpoint-{step}/{name}':row for name,row in inventory.items()};seen={}
 with tarfile.open(target,'r') as tf:
  for member in tf.getmembers():
   require(member.isfile() and safe_name(member.name),'unsafe/non-file tar member')
   require(member.name in expected and member.name not in seen,'unexpected/duplicate tar member')
   f=tf.extractfile(member);require(f is not None,'tar member unreadable');h=hashlib.sha256()
   for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
   seen[member.name]={'bytes':member.size,'sha256':h.hexdigest()}
 require(seen==expected,'tar member inventory differs')
 meta={'step':step,'tar_path':str(target),'tar_bytes':target.stat().st_size,'tar_sha256':sha(target),'files':inventory,'campaign_manifest_sha256':sha(folder/'campaign-manifest.json'),'schedule_sha256':schedule,'consumed_draws':step*16}
 write_json(scratch/f'checkpoint-{step}-local.json',meta)
 return meta
def remote_file(api,path,revision):
 parent=PurePosixPath(path).parent.as_posix()
 rows=[x for x in api.list_repo_tree(repo_id=REPO,repo_type='model',path_in_repo=parent,revision=revision,recursive=False,expand=True) if getattr(x,'path',None)==path]
 require(len(rows)==1,'remote file count differs');return rows[0]
def validate_remote_tar(api,path,revision,meta):
 row=remote_file(api,path,revision);lfs=getattr(row,'lfs',None);digest=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
 require(row.size==meta['tar_bytes'] and digest==meta['tar_sha256'],'remote tar bytes/SHA differs')
def upload_step(run,step,api=None):
 run=Path(run);folder=run/'artifacts/archive/full'/f'checkpoint-{step}';scratch=run/'persistence-staging';event(scratch,'STEP_START',step=step)
 meta=build_tar(folder,step,scratch);token=os.environ.get('HF_TOKEN');require(bool(token),'private upload token absent')
 if api is None:
  from huggingface_hub import HfApi
  api=HfApi(token=token)
 require(api.repo_info(REPO,repo_type='model').private is True,'artifact repository is not private')
 tar_path=Path(meta['tar_path']);remote=f'{PREFIX}/checkpoint-tars/checkpoint-{step}.tar';complete=scratch/f'checkpoint-{step}-complete.json';receipt_path=f'{PREFIX}/checkpoint-tar-receipts/checkpoint-{step}.json'
 if complete.exists():
  result=json.loads(complete.read_text());require(result.get('schema')=='sepalith.cloud-cpt.remote-checkpoint-tar.v1' and result.get('step')==step,'existing completion identity differs')
  require(result.get('tar_sha256')==meta['tar_sha256'] and result.get('files')==meta['files'],'existing completion inventory differs')
  validate_remote_tar(api,remote,result['revision'],meta);q=scratch/f'checkpoint-{step}-receipt.json';rr=remote_file(api,receipt_path,result['receipt_revision'])
  require(rr.size==q.stat().st_size and getattr(rr,'blob_id',None)==git_blob_sha(q),'existing remote receipt differs')
  event(scratch,'STEP_ALREADY_COMPLETE',step=step,revision=result['revision']);return result
 tar_upload=scratch/f'checkpoint-{step}-tar-upload.json'
 if tar_upload.exists():
  uploaded=json.loads(tar_upload.read_text());require(uploaded.get('step')==step and uploaded.get('remote_path')==remote and uploaded.get('tar_sha256')==meta['tar_sha256'],'existing tar upload record differs')
  revision=uploaded['revision'];validate_remote_tar(api,remote,revision,meta);event(scratch,'TAR_ALREADY_UPLOADED',step=step,revision=revision)
 else:
  commit=api.upload_file(repo_id=REPO,repo_type='model',path_or_fileobj=tar_path,path_in_repo=remote,commit_message=f'CPT opaque checkpoint tar {step}')
  revision=commit.oid;validate_remote_tar(api,remote,revision,meta);write_json(tar_upload,{'step':step,'revision':revision,'remote_path':remote,'tar_sha256':meta['tar_sha256'],'tar_bytes':meta['tar_bytes']})
 receipt={'schema':'sepalith.cloud-cpt.remote-checkpoint-tar.v1','run_id':RUN_ID,'step':step,'revision':revision,'remote_path':remote,'tar_sha256':meta['tar_sha256'],'tar_bytes':meta['tar_bytes'],'files':meta['files'],'campaign_manifest_sha256':meta['campaign_manifest_sha256'],'schedule_sha256':meta['schedule_sha256'],'consumed_draws':meta['consumed_draws'],'original_manifest_preserved':True,'credential_persisted':False}
 q=write_json(scratch/f'checkpoint-{step}-receipt.json',receipt)
 rc=api.upload_file(repo_id=REPO,repo_type='model',path_or_fileobj=q,path_in_repo=receipt_path,commit_message=f'CPT opaque checkpoint tar {step} receipt')
 rr=remote_file(api,receipt_path,rc.oid);require(rr.size==q.stat().st_size and getattr(rr,'blob_id',None)==git_blob_sha(q),'remote receipt bytes/hash differs')
 result={**receipt,'receipt_revision':rc.oid,'receipt_path':receipt_path};write_json(complete,result)
 event(scratch,'STEP_END',step=step,revision=revision,receipt_revision=rc.oid,tar_sha256=meta['tar_sha256'])
 return result
def attempt_step(run,step,max_retries=3,backoffs=(30,120),uploader=upload_step,sleeper=time.sleep,reporter=None):
 failures=[]
 for attempt in range(1,max_retries+1):
  try:return uploader(run,step),failures
  except Exception as error:
   row={'step':step,'attempt':attempt,'error_type':type(error).__name__,'error_message':safe(error)};failures.append(row)
   if reporter is not None:reporter(row)
   if attempt<max_retries:sleeper(backoffs[min(attempt-1,len(backoffs)-1)])
 return None,failures
def monitor(run,poll=30,max_retries=3,backoffs=(30,120)):
 run=Path(run);scratch=run/'persistence-staging';done=set();abandoned=set();failures=[];event(scratch,'MONITOR_START',steps=list(STEPS),max_retries=max_retries,deadline=DEADLINE)
 while datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(DEADLINE):
  progress=False
  for step in STEPS:
   if step in done or step in abandoned:continue
   folder=run/'artifacts/archive/full'/f'checkpoint-{step}'
   try:validate_checkpoint(folder,step)
   except Exception:continue
   def report(row):failures.append(row);event(scratch,'STEP_ERROR',**row)
   result,failed=attempt_step(run,step,max_retries,backoffs,reporter=report)
   if result is not None:done.add(step);progress=True
   if step not in done:abandoned.add(step);event(scratch,'STEP_ABANDONED',step=step,max_retries=max_retries)
  if set(STEPS)==done|abandoned:break
  if (run/'stop-sidecar').exists() and not progress:break
  time.sleep(poll)
 result={'schema':'sepalith.cloud-cpt.tar-persistence-monitor.v1','status':'complete' if set(STEPS)==done else 'bounded_exit','uploaded_steps':sorted(done),'abandoned_steps':sorted(abandoned),'failures':failures,'deadline':DEADLINE,'credential_persisted':False}
 write_json(scratch/'monitor-terminal.json',result);event(scratch,'MONITOR_END',result_status=result['status'],uploaded_steps=result['uploaded_steps']);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['once','monitor']);p.add_argument('run',type=Path);p.add_argument('--step',type=int,default=317);a=p.parse_args()
 try:r=upload_step(a.run,a.step) if a.mode=='once' else monitor(a.run);print(json.dumps({'status':'success','mode':a.mode,'result':r},sort_keys=True))
 except Exception as e:
  print(json.dumps({'status':'failed','mode':a.mode,'error_type':type(e).__name__,'error_message':safe(e),'stack':[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)[-6:]],'credential_persisted':False},sort_keys=True));raise SystemExit(1)
if __name__=='__main__':main()
