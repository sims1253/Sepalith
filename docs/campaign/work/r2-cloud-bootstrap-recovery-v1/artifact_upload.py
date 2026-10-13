"""Private HF artifact persistence. Token exists only in this process environment."""
import hashlib,json,logging,os,secrets,sys
from pathlib import Path
from cloud_entry import require,sha,write,stamp
logging.disable(logging.CRITICAL)
def hashes(path):
 size=path.stat().st_size;h=hashlib.sha256();git=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block);git.update(block)
 return {'bytes':size,'sha256':h.hexdigest(),'git_blob_sha1':git.hexdigest()}
def validate_remote(local,remote):
 for name,expected in local.items():
  require(name in remote,'remote artifact missing');row=remote[name]
  require(row['bytes']==expected['bytes'],'remote size differs')
  require(row.get('sha256')==expected['sha256'] if row.get('sha256') else row.get('git_blob_sha1')==expected['git_blob_sha1'],'remote object hash differs')
def main():
 mode,run=sys.argv[1],Path(sys.argv[2]);b=json.loads((run/'artifacts/binding.json').read_text())
 token=os.environ.get('HF_TOKEN');require(bool(token),'private upload token absent')
 from huggingface_hub import HfApi,hf_hub_download
 api=HfApi(token=token);repo=b['artifact_repo'];prefix=b['artifact_prefix']
 require(api.repo_info(repo_id=repo,repo_type='model').private is True,'artifact repository is not private')
 if mode=='sentinel':
  require(not api.file_exists(repo_id=repo,filename=prefix+'/sentinel.json',repo_type='model'),'artifact prefix already used')
  path=run/'sentinel.json';write(path,{'run_id':b['run_id'],'nonce':secrets.token_hex(32)})
  commit=api.upload_file(path_or_fileobj=path,path_in_repo=prefix+'/sentinel.json',repo_id=repo,repo_type='model',commit_message='R2 private persistence probe')
  downloaded=hf_hub_download(repo,filename=prefix+'/sentinel.json',revision=commit.oid,token=token,local_dir=run/'sentinel-readback')
  require(sha(downloaded)==sha(path),'sentinel readback mismatch')
  write(run/'artifacts/sentinel-receipt.json',{'at':stamp(),'commit':commit.oid,'sha256':sha(path),'scope':'Uploader-client readback only; independent root readback remains a provider admission/release gate.'})
  return
 require(mode=='final','unknown upload stage')
 folder=run/'artifacts';local={}
 for path in sorted(folder.rglob('*')):
  require(not path.is_symlink(),'artifact symlink forbidden')
  if path.is_file():local[str(path.relative_to(folder))]=hashes(path)
 write(folder/'upload-manifest.json',{'schema':1,'files':local,'token_persisted':False})
 local['upload-manifest.json']=hashes(folder/'upload-manifest.json')
 commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=folder,path_in_repo=prefix+'/artifacts',commit_message='R2 bounded control artifacts')
 remote={}
 for row in api.list_repo_tree(repo_id=repo,repo_type='model',path_in_repo=prefix+'/artifacts',revision=commit.oid,recursive=True,expand=True):
  if not hasattr(row,'size'):continue
  name=row.path.removeprefix(prefix+'/artifacts/');lfs=getattr(row,'lfs',None)
  remote[name]={'bytes':row.size,'sha256':getattr(lfs,'sha256',None) if not isinstance(lfs,dict) else lfs.get('sha256'),'git_blob_sha1':getattr(row,'blob_id',None)}
 validate_remote(local,remote)
 receipt={'schema':1,'at':stamp(),'status':'uploaded_commit_metadata_verified','repo':repo,'prefix':prefix,'commit':commit.oid,'files':local,'remote_files_verified':len(local),'independent_root_readback_after_provider_exit_required':True}
 receipt_path=run/'persistence-receipt.json';write(receipt_path,receipt)
 final=api.upload_file(repo_id=repo,repo_type='model',path_or_fileobj=receipt_path,path_in_repo=prefix+'/persistence-receipt.json',commit_message='R2 persistence receipt')
 downloaded=hf_hub_download(repo,filename=prefix+'/persistence-receipt.json',revision=final.oid,token=token,local_dir=run/'receipt-readback')
 require(sha(downloaded)==sha(receipt_path),'persistence receipt readback differs')
 receipt['receipt_commit']=final.oid;write(receipt_path,receipt)
if __name__=='__main__':
 try:main()
 except Exception as error:
  print(json.dumps({'status':'artifact_operation_failed','error_type':type(error).__name__}),flush=True);raise SystemExit(1)
