"""Run one reviewed full15006 queue item and retain supervisor identity."""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,"/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")
from sepalith.runner import Runner
RUNNER_ROOT='/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-full15006-v1'
PINS={'/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/recipe.json': 'a839121502dd78d15a9a825c5c6f29d24276806a5bfc4214825917f3f0daeaef', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/runner-recipe.json': '23c6d298b2dfb95803d919ecc510527c834831cc397d1b6b790599bbbaaa3c42', '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-expanded-sft-full15008-v1/source-manifest.json': '8389c8266f1f831fd3b2166011ec02094ee591074ded2df0b28b61d3fa905105'}
for name,expected in PINS.items():
 actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError(f"{name} differs from reviewed full15006 packet")
for path in map(Path,['/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-full15006-a', '/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-full15006-a', '/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-full15006-a-host-supervision']):
 if path.exists():raise RuntimeError(f"full15006 output is not fresh: {path}")
work=Path(__file__).resolve().parent;started=time.monotonic();launch={"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"start_stat":Path('/proc/self/stat').read_text(),"owner":"root","task":"SFT-11-expanded-full15006","pins":PINS}
with (work/'supervisor-launch.json').open('x') as stream:json.dump(launch,stream,indent=2)
terminal={}
try:
 runner=Runner(RUNNER_ROOT);runner.resume();result=runner.run_next();terminal={"result":result,"seconds":time.monotonic()-started}
except BaseException as error:
 terminal={"failure":repr(error),"seconds":time.monotonic()-started};raise
finally:
 terminal['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (work/'supervisor-terminal.json').open('x') as stream:json.dump(terminal,stream,indent=2)
