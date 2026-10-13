import pathlib,sys,json,fcntl,subprocess,datetime
W=pathlib.Path(__file__).resolve().parent
P=W.parents[2]
lock=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path.insert(0,str(P/'work/r2-draft-cloud-profile-v2'))
from native_cuda_smoke import load_upstream_smoke
m=load_upstream_smoke(W/'upstream_dspark_smoke.py',P/'work/lead/r2-draft-cloud-runtime-preparation/vendor/DeepSpec')
r=m.run_smoke(device='cuda:0');(W/'cuda-result.json').write_text(json.dumps(r,indent=2)+'\n');print('CUDA probe returned')
