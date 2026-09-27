"""Run one root-reviewed SFT-11 E queue item and retain supervisor identity."""
import datetime,hashlib,json,os
from pathlib import Path
import sys,time
sys.path.insert(0,"/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")
from sepalith.runner import Runner
RUNNER_ROOT='/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-e-v1'
PINS={'/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-e/recipe.json': '2e8038f81c8fee2112b7164e4e0c92c11474459dd1e810f8a86c2bf224ca2316', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-e/runner-recipe.json': '93284074b4ae2185bdbac540c2102280bed5a9093c7dc7a2ece2eba71891136a', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-d/source-manifest.json': '8e3ddae3cdd318b20d83a2068bef417e1073e5b9ae61c7df60ec9e6380ccf289', '/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d/full/checkpoint-500/campaign-manifest.json': '6a4fe603566af95a09c3fd3bdadea37dac2717db55b08063a0bc9a1d93a1223b'}
for name,expected in PINS.items():
 actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError(f"{name} differs from reviewed E packet")
for path in map(Path,['/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e', '/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-e', '/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-e-host-supervision']):
 if path.exists():raise RuntimeError(f"E output is not fresh: {path}")
work=Path(__file__).resolve().parent;started=time.monotonic();launch={"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"start_stat":Path('/proc/self/stat').read_text(),"owner":"root","task":"SFT-11-expanded-E-resume500-to1000","pins":PINS}
with (work/'supervisor-launch.json').open('x') as stream:json.dump(launch,stream,indent=2)
terminal={}
try:
 runner=Runner(RUNNER_ROOT);runner.resume();result=runner.run_next();terminal={"result":result,"seconds":time.monotonic()-started}
except BaseException as error:
 terminal={"failure":repr(error),"seconds":time.monotonic()-started};raise
finally:
 terminal['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (work/'supervisor-terminal.json').open('x') as stream:json.dump(terminal,stream,indent=2)
