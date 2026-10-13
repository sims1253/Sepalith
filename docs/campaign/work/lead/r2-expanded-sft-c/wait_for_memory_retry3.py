"""One root-authorized retry after two healthy memory observations; never promote."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

W = Path(__file__).resolve().parent
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
PYTHON = '/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
RUNNER = '/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py'
ROOT = '/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-expanded-sft-c-v1'
LOCK = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
SCRIPT = 'Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes,PagesInputPersec,PagesOutputPersec | ConvertTo-Json'
CONFIG = json.loads((W/'retry3-wait-admission.json').read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def record(name, value):
    with (W/name).open('a') as f:
        f.write(json.dumps({'at': now(), **value})+'\n'); f.flush(); os.fsync(f.fileno())
def check_inputs():
    for path, expected in CONFIG['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for path in CONFIG['fresh_paths']:
        assert not Path(path).exists(), path
record('retry3-memory-wait.jsonl', {'status':'waiting', 'pid':os.getpid(), 'admission_free_mib':18432, 'required_samples':2})
healthy = 0
for sample in range(720):
    try:
        p = subprocess.run([PS,'-NoProfile','-NonInteractive','-Command',SCRIPT],capture_output=True,text=True,timeout=10,check=True)
        value = json.loads(p.stdout)
        good = value['AvailableMBytes'] >= 18432 and value['PagesOutputPersec'] == 0 and value['PagesInputPersec'] <= 64
        healthy = healthy+1 if good else 0
        record('retry3-memory-wait.jsonl', {'status':'observation','memory':value,'consecutive_healthy':healthy})
        if healthy >= 2:
            check_inputs()
            with LOCK.open('r+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            queued = subprocess.run([PYTHON,RUNNER,'--state',ROOT,'enqueue',str(W/'root-runner-recipe-retry3.json')],capture_output=True,text=True,check=True)
            record('retry3-memory-wait.jsonl', {'status':'enqueued','result':queued.stdout})
            with (W/'supervisor-retry3.log').open('xb') as log:
                child = subprocess.Popen([PYTHON,str(W/'run_owned_retry3.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record('retry3-memory-wait.jsonl', {'status':'supervisor_started','pid':child.pid})
            code = child.wait()
            record('retry3-memory-wait.jsonl', {'status':'supervisor_finished','pid':child.pid,'exit_code':code,'experiment_acceptance':'requires root artifact review'})
            break
    except BlockingIOError:
        healthy = 0
        record('retry3-memory-wait.jsonl', {'status':'cuda_owned_elsewhere'})
    except (subprocess.SubprocessError,ValueError,KeyError) as error:
        healthy = 0
        record('retry3-memory-wait.jsonl', {'status':'observation_or_launch_error','error':str(error)})
        if isinstance(error, subprocess.CalledProcessError) and 'enqueue' in error.cmd:
            raise
    time.sleep(30)
else:
    record('retry3-memory-wait.jsonl', {'status':'wait_window_expired_no_launch','next':'root rechecks resources; this is not a data exclusion or training budget limit'})
