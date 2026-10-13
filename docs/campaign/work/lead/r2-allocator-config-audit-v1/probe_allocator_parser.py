#!/usr/bin/env python3
"""CPU-only parser/import-order evidence. Run with CUDA_VISIBLE_DEVICES empty."""
import json,os,subprocess,sys
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
code_parser=r'''
import json,torch
before=torch.cuda.is_initialized();results={}
for value in ('expandable_segments:True','definitely_invalid_key:True','roundup_power2_divisions:[32:256,64:128,256:64,>:32]','backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8'):
 try: torch.cuda.memory._set_allocator_settings(value);results[value]='accepted'
 except Exception as e:results[value]=type(e).__name__+':'+str(e)
 assert not torch.cuda.is_initialized()
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'before_initialized':before,'after_initialized':torch.cuda.is_initialized(),'results':results,'backend':torch.cuda.memory.get_allocator_backend()}))
'''
code_unsloth=r'''
import json,os
before={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
try: import unsloth_zoo;status='ok'
except BaseException as e:status=type(e).__name__+':'+str(e).splitlines()[0]
import torch
print(json.dumps({'before':before,'after':{k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},'import_status':status,'cuda_initialized':torch.cuda.is_initialized()}))
'''
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
def run(code):
 p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=30);assert p.returncode==0,p.stderr;return json.loads(p.stdout.strip().splitlines()[-1])
out={'schema':'sepalith.sft11.allocator_config_cpu_probe.v1','parser':run(code_parser),'unsloth_import':run(code_unsloth),'cuda_visible_devices':'empty','gpu_workload':False}
assert out['parser']['before_initialized'] is False and out['parser']['after_initialized'] is False
assert out['parser']['results']['expandable_segments:True']=='accepted'
assert out['parser']['results']['backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8']=='accepted'
assert out['parser']['results']['definitely_invalid_key:True'].startswith('ValueError:Unrecognized key')
assert out['unsloth_import']['after']['PYTORCH_ALLOC_CONF'] is None
assert out['unsloth_import']['cuda_initialized'] is False
print(json.dumps(out,indent=2,sort_keys=True))
