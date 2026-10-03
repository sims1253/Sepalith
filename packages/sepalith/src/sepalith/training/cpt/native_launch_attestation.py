#!/usr/bin/env python3
"""Verify reusable base inputs plus the selected resume checkpoint, then run."""
import argparse,fcntl,hashlib,json,os,subprocess
from pathlib import Path
from sepalith.training.cpt.native_stage import verify_and_lock,safe_relative,fingerprint,open_regular,BLOCK

CUDA_LOCK=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')

def require(v,m):
 if not v:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(BLOCK),b''):h.update(b)
 return h.hexdigest()

def inherited_cuda_lock_fd(lock_path=CUDA_LOCK):
 value=os.environ.get('SEPALITH_CUDA_LOCK_FD')
 if value is None:return None
 require(value.isdigit(),'CUDA lock fd differs');fd=int(value);held=os.fstat(fd);expected=Path(lock_path).stat()
 require((held.st_dev,held.st_ino)==(expected.st_dev,expected.st_ino),'CUDA lock fd refers to another file')
 probe=os.open(lock_path,os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
 try:
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:
   fcntl.flock(probe,fcntl.LOCK_UN);raise ValueError('CUDA lock fd does not hold the lease')
 finally:os.close(probe)
 return fd

def run_child(command,env,all_fds,lock_path=CUDA_LOCK):
 lock_fd=inherited_cuda_lock_fd(lock_path);passed=list(all_fds)
 if lock_fd is not None and lock_fd not in passed:passed.append(lock_fd)
 return subprocess.run(command,env=env,pass_fds=tuple(passed),check=False).returncode

def base_records(receipt,held):
 records=[];index=0
 for obj in receipt['objects']:
  root=Path(obj['staged_path'])
  for relative,expected in obj['files'].items():
   records.append({'path':str((root/safe_relative(relative)).resolve()),'fd':held[index],'bytes':expected['bytes'],'sha256':expected['sha256'],'fingerprint':expected['staged_fingerprint']});index+=1
 return records

def verify_resume(checkpoint,manifest_sha256):
 root=Path(checkpoint);mp=root/'campaign-manifest.json';require(root.is_dir() and not root.is_symlink() and sha(mp)==manifest_sha256,'resume checkpoint manifest differs')
 manifest=json.loads(mp.read_text());require(manifest.get('full')is True and manifest.get('checkpoint_kind')=='full_weights','resume is not full state')
 required={'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','campaign-state.json','tokenizer.json'};require(required<=set(manifest.get('files',{})),'resume state inventory incomplete')
 expected={**manifest['files'],'campaign-manifest.json':{'bytes':mp.stat().st_size,'sha256':manifest_sha256}}
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()};require(actual==set(expected),'resume checkpoint file set differs')
 held=[];records=[]
 try:
  for relative,item in sorted(expected.items()):
   path=root/safe_relative(relative);fd,before=open_regular(path);fcntl.flock(fd,fcntl.LOCK_SH);h=hashlib.sha256();size=0
   while True:
    block=os.read(fd,BLOCK)
    if not block:break
    h.update(block);size+=len(block)
   after=os.fstat(fd);require(fingerprint(before)==fingerprint(after),'resume changed during verification');require(size==item['bytes'] and h.hexdigest()==item['sha256'],f'resume payload differs:{relative}')
   held.append(fd);records.append({'path':str(path.resolve()),'fd':fd,'bytes':size,'sha256':item['sha256'],'fingerprint':fingerprint(before)})
  return records,held,sum(item['bytes'] for item in manifest['files'].values())+mp.stat().st_size
 except Exception:
  for fd in held:os.close(fd)
  raise

def run(base_receipt,base_sha,resume,resume_manifest_sha,command):
 receipt,base_fds=verify_and_lock(base_receipt,base_sha);resume_records,resume_fds,resume_bytes=verify_resume(resume,resume_manifest_sha);all_fds=base_fds+resume_fds
 try:
  records=base_records(receipt,base_fds)+resume_records;env=dict(os.environ);env['SEPALITH_NATIVE_STAGE_BUNDLE_ID']=receipt['bundle_id'];env['SEPALITH_NATIVE_STAGE_ATTESTATION']=json.dumps({'receipt_sha256':base_sha,'bundle_id':receipt['bundle_id'],'files':records},sort_keys=True,separators=(',',':'));env['SEPALITH_NATIVE_RESUME_PATH']=str(Path(resume).resolve());env['SEPALITH_NATIVE_RESUME_BYTES']=str(resume_bytes if str(Path(resume).resolve()).startswith('/home/') else 0)
  return run_child(command,env,all_fds)
 finally:
  for fd in all_fds:os.close(fd)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base-receipt',required=True);p.add_argument('--base-receipt-sha256',required=True);p.add_argument('--resume',required=True);p.add_argument('--resume-manifest-sha256',required=True);p.add_argument('child',nargs=argparse.REMAINDER);a=p.parse_args();child=a.child[1:]if a.child and a.child[0]=='--'else a.child;require(child,'child command empty');return run(a.base_receipt,a.base_receipt_sha256,a.resume,a.resume_manifest_sha256,child)
if __name__=='__main__':raise SystemExit(main())
