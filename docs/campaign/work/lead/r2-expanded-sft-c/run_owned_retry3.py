"""Run one root-admitted SFT-11 C queue item and retain supervisor identity."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, "/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src")
from sepalith.runner import Runner


WORK = Path(__file__).resolve().parent
RUNNER_ROOT = "/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-c-v1"
started = time.monotonic()
launch = {
    "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pid": os.getpid(),
    "start_stat": Path("/proc/self/stat").read_text(),
    "owner": "root",
    "task": "SFT-11-expanded-C-resume150",
}
with (WORK / "supervisor-launch-retry3.json").open("x") as stream:
    json.dump(launch, stream)
try:
    runner = Runner(RUNNER_ROOT)
    runner.resume()
    result = runner.run_next()
    terminal = {"result": result, "seconds": time.monotonic() - started}
except BaseException as error:
    terminal = {"failure": repr(error), "seconds": time.monotonic() - started}
    raise
finally:
    terminal["at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (WORK / "supervisor-terminal-retry3.json").open("x") as stream:
        json.dump(terminal, stream, indent=2)
