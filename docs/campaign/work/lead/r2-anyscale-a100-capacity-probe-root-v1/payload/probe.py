import csv,hashlib,json,os,shutil,subprocess,tempfile,time
from pathlib import Path
r={'schema':'sepalith.a100.capacity_probe.v1','at':time.time(),'training_performed':False}
x=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
assert x.returncode==0,'nvidia-smi failed'
gpus=[{'name':a.strip(),'memory_mib':int(b.strip()),'driver':c.strip()} for a,b,c in csv.reader(x.stdout.splitlines())]
assert len(gpus)==8 and all('A100' in g['name'] and g['memory_mib']>=39000 for g in gpus),'expected eight A10040GB devices'
r['gpus']=gpus
base=Path('/mnt/local_storage');base.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='sepalith-a100-probe-',dir=base) as tmp:
 p=Path(tmp)/'roundtrip';data=b'sepalith-storage-proof'*1024;p.write_bytes(data);assert p.read_bytes()==data
r['local_storage']={'roundtrip':True,'free_bytes':shutil.disk_usage(base).free}
code="import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device_count':torch.cuda.device_count(),'bf16_roundtrip':[bool(torch.isfinite(torch.ones((16,16),device='cuda:'+str(i),dtype=torch.bfloat16)@torch.ones((16,16),device='cuda:'+str(i),dtype=torch.bfloat16)).all()) for i in range(torch.cuda.device_count())]}))"
x=subprocess.run(['python','-c',code],capture_output=True,text=True,timeout=90)
r['base_torch_probe']={'exit_code':x.returncode,'stdout':x.stdout[-4000:],'stderr':x.stderr[-1000:]}
r['status']='hardware_and_local_storage_verified'
print('SEPALITH_A100_RECEIPT '+json.dumps(r,sort_keys=True),flush=True)
Path('/tmp/sepalith-a100-probe.json').write_text(json.dumps(r,indent=2)+'\n')
