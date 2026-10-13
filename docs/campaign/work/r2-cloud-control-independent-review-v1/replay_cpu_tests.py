"""Replay worker CPU tests with all temporary writes relocated into this review scope."""
import hashlib,os,sys,unittest
from pathlib import Path
H=Path(__file__).resolve().parent;W=H.parent/'r2-cloud-control-entry-v1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});sys.path.insert(0,str(W))
text=(W/'test_cloud_entry.py').read_text().replace('dir=c.HERE','dir=REVIEW_ROOT')
module={'__name__':'worker_cpu_tests','__file__':str(W/'test_cloud_entry.py'),'REVIEW_ROOT':H}
exec(compile(text,str(W/'test_cloud_entry.py'),'exec'),module)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(module['CloudTests'])
r=unittest.TextTestRunner(verbosity=2).run(suite)
assert r.wasSuccessful()
import torch
assert not torch.cuda.is_initialized()
print({'worker_tests':r.testsRun,'CUDA_initialized':False,'temporary_write_root':str(H),'test_source_sha256':hashlib.sha256((W/'test_cloud_entry.py').read_bytes()).hexdigest()})
