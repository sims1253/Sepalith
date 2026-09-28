#!/usr/bin/env python3
"""Root-only, explicitly armed upload of the frozen private CPT inputs."""
import argparse,hashlib,json,logging,os,re
from pathlib import Path
logging.disable(logging.CRITICAL)
HERE=Path(__file__).resolve().parent;REPO='scholzmx/sepalith-lora'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def git_blob(p):
 p=Path(p);h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0')
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def remote_files(api,prefix,revision=None):
 # list_repo_tree is lazy: force iteration inside the handler so an absent
 # immutable prefix is treated as an empty resumable upload, not a failure.
 try:rows=list(api.list_repo_tree(REPO,repo_type='model',path_in_repo=prefix,revision=revision,recursive=True,expand=True))
 except Exception as e:
  if type(e).__name__ in ('EntryNotFoundError','RemoteEntryNotFoundError'):return {}
  raise
 return {r.path:r for r in rows if hasattr(r,'size')}
def remote_exact(remote,row,local):
 r=remote.get(row['remote_path'])
 if r is None:return False
 lfs=getattr(r,'lfs',None);lfs_sha=getattr(lfs,'sha256',None) if not isinstance(lfs,dict) else lfs.get('sha256')
 return r.size==row['bytes'] and (lfs_sha==row['sha256'] if lfs_sha else getattr(r,'blob_id',None)==git_blob(local))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root-admission',type=Path,required=True);p.add_argument('--expected-root-admission-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 require(a.execute,'private upload requires explicit --execute after root admission')
 require(re.fullmatch('[0-9a-f]{64}',a.expected_root_admission_sha256) and sha(a.root_admission)==a.expected_root_admission_sha256,'root admission hash differs')
 admission=json.loads(a.root_admission.read_text());status=str(admission.get('status','')).lower();require(admission.get('admitted') is True or any(x in status for x in ('accept','admit','pass')),'root admission is not affirmative')
 manifest=HERE/'transport-manifest.json';spec=json.loads(manifest.read_text());require(spec['repo_id']==REPO,'input repo differs')
 token=os.environ.get('HF_TOKEN');require(bool(token),'private upload token absent')
 from huggingface_hub import HfApi
 api=HfApi(token=token);require(api.repo_info(REPO,repo_type='model').private is True,'repository is not private')
 require(not api.file_exists(REPO,spec['prefix']+'/transport-manifest.json',repo_type='model'),'complete input prefix already exists')
 existing=remote_files(api,spec['prefix']+'/files')
 commits=[]
 for row in spec['files']:
  local=Path(row['original_path']);require(local.is_file() and local.stat().st_size==row['bytes'] and sha(local)==row['sha256'],'local frozen input differs')
  if row['remote_path'] in existing:
   require(remote_exact(existing,row,local),'partial remote input differs');continue
  c=api.upload_file(repo_id=REPO,repo_type='model',path_or_fileobj=local,path_in_repo=row['remote_path'],commit_message='CPT frozen private input');commits.append(c.oid)
 final=api.upload_file(repo_id=REPO,repo_type='model',path_or_fileobj=manifest,path_in_repo=spec['prefix']+'/transport-manifest.json',commit_message='CPT transport manifest committed last')
 remote=remote_files(api,spec['prefix']+'/files',final.oid)
 for row in spec['files']:
  require(remote_exact(remote,row,Path(row['original_path'])),'remote input inventory differs')
 receipt={'schema':'sepalith.cloud-cpt.input-upload.v1','status':'uploaded_and_remote_sizes_verified','repo':REPO,'prefix':spec['prefix'],'revision':final.oid,'files':len(spec['files']),'bytes':spec['bytes'],'transport_manifest_sha256':sha(manifest),'root_admission_sha256':a.expected_root_admission_sha256,'credential_persisted':False,'independent_hash_readback_required':True}
 destination=HERE/'private-input-upload-receipt.json';destination.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':receipt['status'],'revision':final.oid}))
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'status':'upload_rejected_or_failed','error_type':type(e).__name__}));raise SystemExit(1)
