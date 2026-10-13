"""Run the root-admitted expanded SFT queue and retain supervisor identity."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, '/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src')
from sepalith.runner import Runner

W = Path(__file__).resolve().parent
ROOT = '/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-v1'
started = time.monotonic()
launch = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              pid=os.getpid(), start_stat=Path('/proc/self/stat').read_text(),
              owner='lead', task='SFT-11')
with (W/'supervisor-launch.json').open('x') as f:
    json.dump(launch, f)
try:
    result = Runner(ROOT).run_next()
    terminal = dict(result=result, seconds=time.monotonic()-started)
except BaseException as exc:
    terminal = dict(failure=repr(exc), seconds=time.monotonic()-started)
    raise
finally:
    terminal['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (W/'supervisor-terminal.json').open('x') as f:
        json.dump(terminal, f, indent=2)
