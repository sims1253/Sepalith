import sys,tarfile,io,pathlib,os,json,hashlib,subprocess,time,datetime
root=pathlib.Path('/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/r2-notebook-r-parse-a');root.mkdir(parents=True,exist_ok=False)
buf=sys.stdin.buffer.read();assert len(buf)<20000000
with tarfile.open(fileobj=io.BytesIO(buf),mode='r:gz') as tf:
 for m in tf.getmembers():
  assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(m).read())
rows=json.loads((root/'manifest.json').read_text());assert len(rows)==411
for r in rows:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''});start=time.monotonic();result=subprocess.run(['Rscript','--vanilla','check.R'],cwd=root,env=env,capture_output=True,text=True,timeout=150)
receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'DAT-10/SFT-11','pid':os.getpid(),'hostname':os.uname().nodename,'kernel':os.uname().release,'exit':result.returncode,'seconds':time.monotonic()-start,'stdout':result.stdout,'stderr':result.stderr,'rows':len(rows),'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'operation':'Native R parse only. No generated R code executed.','result_csv':(root/'r-parse-results.csv').read_text() if (root/'r-parse-results.csv').exists() else None}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
