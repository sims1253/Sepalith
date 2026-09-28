"""Relocate the two recorded intermediate F16 exports after CUDA release."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,shutil,subprocess,time
W=pathlib.Path(__file__).resolve().parent
N=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args();plan=json.loads((W/'plan.json').read_text())
 if not args.execute:print(json.dumps(plan,indent=2));return
 with (N/'resource-locks/cuda0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
  for row in plan['candidates']:
   source=pathlib.Path(row['source']);dest=pathlib.Path(row['destination']);receipt=W/(source.parent.name+'-relocation.json')
   assert source.is_file() and not source.is_symlink();assert not dest.exists() and not receipt.exists()
   pin=next(x for x in json.loads(pathlib.Path(row['metadata']).read_text())['files'] if x['path']==str(source));assert source.stat().st_size==pin['bytes']
   dest.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(dest.parent).free>pin['bytes']+1024**3
   temp=dest.with_suffix(dest.suffix+'.partial');start=time.monotonic();h=hashlib.sha256();copied=0
   with source.open('rb') as src,temp.open('xb') as out:
    for block in iter(lambda:src.read(4*1024*1024),b''):
     out.write(block);h.update(block);copied+=len(block);time.sleep(max(0,copied/(40*1024*1024)-(time.monotonic()-start)))
    out.flush();os.fsync(out.fileno())
   assert copied==pin['bytes'] and h.hexdigest()==pin['sha256'];assert digest(temp)==pin['sha256'];temp.rename(dest)
   r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(source),'destination':str(dest),'bytes':copied,'sha256':pin['sha256'],'destination_readback_verified':True,'status':'copy_verified_local_present'}
   receipt.write_text(json.dumps(r,indent=2)+'\n')
   # The verified NAS copy preserves this intermediate artifact. All other
   # model files and all checkpoints remain at their original paths.
   source.unlink();r['status']='relocated_local_duplicate_removed';receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
