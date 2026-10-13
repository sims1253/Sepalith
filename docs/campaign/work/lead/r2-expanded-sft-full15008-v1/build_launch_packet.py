#!/usr/bin/env python3
"""Build root-owned launch metadata for the fresh E750/full15006 stage."""
import hashlib,json
from copy import deepcopy
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
WORK=PLAN/'docs/campaign/work/lead/r2-expanded-sft-full15008-v1'
SOURCE=WORK/'source'
SOURCE_ID='28e224b22497ea9881aba29fdea1c9dd5725681da19099a54bfb800ae2f4bbde'
RUNNER_ROOT=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1')
OUTPUT=Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-full15006-a')
ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a')
SUPERVISION=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-full15006-a-host-supervision')
PYTHON='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
RUNNER='/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rec(p): return {'path':str(p),'sha256':sha(p)}
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def build():
 recipe=json.load(open(WORK/'recipe.json'))
 base=json.load(open(PLAN/'docs/campaign/work/lead/r2-expanded-sft-d/runner-recipe.json'))
 runner=deepcopy(base)
 runner.update({'id':recipe['id'],'snapshot':SOURCE_ID,'description':'Fresh LoRA on verified merged E750; reset optimizer; one complete mixed full15006 coverage pass through step1160','depends_on':[]})
 runner['provenance']={'task':'SFT-11-expanded-full15006','owner':'lead','status':'prepared_root_review_snapshot_enqueue_launch_required','parent':'merged E750 weights '+recipe['identity']['parent']['weights_sha256'],'optimizer':'fresh reset at step0; no E checkpoint resume','coverage':'all15006 eligible rows after exactly two named contradiction exclusions, including all3503 finish additions, first complete at terminal1160','later_continuation':'requires evidence and a separate recipe; this packet declares one pass'}
 extras=[PLAN/'docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py',PLAN/'docs/campaign/work/lead/host-memory-guard-v4/host_memory_policy.py',PLAN/'docs/campaign/work/lead/host-memory-guard-v4/post_load_cache.py',PLAN/'docs/campaign/work/lead/host-memory-guard-v4/root-policy-tests.json',PLAN/'docs/campaign/work/lead/r2-cpt-broad-f/prepare_guard_command.py',WORK/'recipe.json',WORK/'source-manifest.json']
 runner['inputs']=sorted({r['path']:r for r in [*recipe['inputs'],*(rec(p) for p in extras)]}.values(),key=lambda r:r['path'])
 runner['steps'][0]['argv'][4]=str(WORK/'recipe.json')
 runner['steps'][1]['argv']=[PYTHON,str(PLAN/'docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py'),'--command-json','{run}/guard-command.json','--output',str(SUPERVISION),'--seconds','15000','--minimum-free-mib','6144','--admission-free-mib','14336','--release-cache-file',str(Path(recipe['model_path'])/'model.safetensors')]
 runner['steps'][1]['id']='guarded-expanded-full15006-one-pass'
 write(WORK/'runner-recipe.json',runner)
 pins={str(WORK/'recipe.json'):sha(WORK/'recipe.json'),str(WORK/'runner-recipe.json'):sha(WORK/'runner-recipe.json'),str(WORK/'source-manifest.json'):sha(WORK/'source-manifest.json')}
 source='''"""Run one reviewed full15006 queue item and retain supervisor identity."""\nimport datetime,hashlib,json,os,sys,time\nfrom pathlib import Path\nsys.path.insert(0,"/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")\nfrom sepalith.runner import Runner\nRUNNER_ROOT={runner_root!r}\nPINS={pins!r}\nfor name,expected in PINS.items():\n actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()\n if actual!=expected:raise RuntimeError(f"{{name}} differs from reviewed full15006 packet")\nfor path in map(Path,{fresh!r}):\n if path.exists():raise RuntimeError(f"full15006 output is not fresh: {{path}}")\nwork=Path(__file__).resolve().parent;started=time.monotonic();launch={{"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"start_stat":Path('/proc/self/stat').read_text(),"owner":"root","task":"SFT-11-expanded-full15006","pins":PINS}}\nwith (work/'supervisor-launch.json').open('x') as stream:json.dump(launch,stream,indent=2)\nterminal={{}}\ntry:\n runner=Runner(RUNNER_ROOT);runner.resume();result=runner.run_next();terminal={{"result":result,"seconds":time.monotonic()-started}}\nexcept BaseException as error:\n terminal={{"failure":repr(error),"seconds":time.monotonic()-started}};raise\nfinally:\n terminal['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()\n with (work/'supervisor-terminal.json').open('x') as stream:json.dump(terminal,stream,indent=2)\n'''.format(runner_root=str(RUNNER_ROOT),pins=pins,fresh=[str(OUTPUT),str(ARCHIVE),str(SUPERVISION)])
 (WORK/'run_owned.py').write_text(source)
 commands={'schema':'sepalith.sft11.expanded-full15006.commands.v1','status':'prepared_not_executed_root_owned','pins':{'runner_snapshot':SOURCE_ID,'recipe_sha256':sha(WORK/'recipe.json'),'runner_recipe_sha256':sha(WORK/'runner-recipe.json'),'source_manifest_sha256':sha(WORK/'source-manifest.json'),'run_owned_sha256':sha(WORK/'run_owned.py')},'metadata_compatibility_audit':['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+str(SOURCE/'experiments/training'),PYTHON,'-B',str(WORK/'tests/test_full15008_packet.py')],'launch_path_cpu_audit_without_weight_read':['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+str(SOURCE/'experiments/training'),PYTHON,'-B',str(WORK/'launch_path_audit.py')],'root_cpu_preflight_no_cuda':['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+str(SOURCE/'packages/sepalith/src')+':'+str(SOURCE/'experiments/training'),PYTHON,'-B',str(SOURCE/'experiments/training/campaign_expanded_sft.py'),str(WORK/'recipe.json'),'--preflight-only'],'snapshot_replay':[PYTHON,'-B',RUNNER,'--state',str(RUNNER_ROOT),'snapshot','--repo',str(SOURCE),'--include','experiments','--include','packages'],'enqueue_after_exact_snapshot_and_root_admission':[PYTHON,'-B',RUNNER,'--state',str(RUNNER_ROOT),'enqueue',str(WORK/'runner-recipe.json')],'launch_after_root_review_and_cuda_lease':[PYTHON,'-B',str(WORK/'run_owned.py')]}
 write(WORK/'commands.json',commands)
 print(json.dumps({'runner_recipe':sha(WORK/'runner-recipe.json'),'run_owned':sha(WORK/'run_owned.py'),'commands':sha(WORK/'commands.json')}))
if __name__=='__main__':build()
