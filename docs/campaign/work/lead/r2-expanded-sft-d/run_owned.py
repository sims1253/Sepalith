"""Run one root-reviewed SFT-11 D queue item and retain supervisor identity."""
import datetime, hashlib, json, os
from pathlib import Path
import sys, time
sys.path.insert(0, "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")
from sepalith.runner import Runner
WORK=Path(__file__).resolve().parent
RUNNER_ROOT="/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-d-v1"
PINS={"recipe.json":"7ff8bc587b50a078bd7e5d7bd6f4538c985b5781e2fdc40273e2739c24e35550","runner-recipe.json":"ceafc5913923aa5e86656e6e774173b297ea80213b382c310c5e038236ee3b4b","source-manifest.json":"8e3ddae3cdd318b20d83a2068bef417e1073e5b9ae61c7df60ec9e6380ccf289"}
for name, expected in PINS.items():
    actual=hashlib.sha256((WORK/name).read_bytes()).hexdigest()
    if actual != expected: raise RuntimeError(f"{name} differs from reviewed D packet")
for path in (Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-d'),Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d'),Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-expanded-postsft500-d-host-supervision')):
    if path.exists(): raise RuntimeError(f"D output is not fresh: {path}")
started=time.monotonic(); launch={"at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"start_stat":Path('/proc/self/stat').read_text(),"owner":"root","task":"SFT-11-expanded-D-resume250-to500","pins":PINS}
with (WORK/'supervisor-launch.json').open('x') as stream: json.dump(launch,stream,indent=2)
terminal={}
try:
    runner=Runner(RUNNER_ROOT); runner.resume(); result=runner.run_next(); terminal={"result":result,"seconds":time.monotonic()-started}
except BaseException as error:
    terminal={"failure":repr(error),"seconds":time.monotonic()-started}; raise
finally:
    terminal['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (WORK/'supervisor-terminal.json').open('x') as stream: json.dump(terminal,stream,indent=2)
