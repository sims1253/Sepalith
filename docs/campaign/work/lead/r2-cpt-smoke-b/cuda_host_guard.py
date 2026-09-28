#!/usr/bin/env python3
"""Supervise one owned process group with Windows memory and GPU-reset checks."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import fcntl
import subprocess
import time
from host_memory_policy import HostMemoryPolicy

p = argparse.ArgumentParser()
p.add_argument('--command-json', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=int, required=True)
p.add_argument('--minimum-free-mib', type=int, default=12288)
p.add_argument('--release-cache-file', type=Path)
a = p.parse_args()
assert 1 <= a.seconds <= 93600
if a.release_cache_file is not None:
    if not a.release_cache_file.resolve().is_relative_to(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models')):
        raise ValueError('cache-release model path is outside owned campaign models')
    if a.release_cache_file.name != 'model.safetensors' or not a.release_cache_file.is_file():
        raise ValueError('cache-release path must be an existing model.safetensors')
    from post_load_cache import release_after_load
cache_attempts = set()
command = json.loads(a.command_json.read_text())
assert isinstance(command, list) and command and all(isinstance(v, str) for v in command)
assert a.output.is_absolute() and not a.output.exists()
a.output.mkdir(parents=True)
# The same lock inode stays alive in this guard, timeout and trainer.
# pass_fds keeps the CUDA claim valid even if an outer runner exits.
CUDA_LOCK = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
cuda_lock_fd = os.open(CUDA_LOCK, os.O_RDWR | os.O_NOFOLLOW)
fcntl.flock(cuda_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.set_inheritable(cuda_lock_fd, True)
os.environ['SEPALITH_CUDA_LOCK_FD'] = str(cuda_lock_fd)
_lock_stat = os.fstat(cuda_lock_fd)
(a.output / 'cuda-lease.json').write_text(json.dumps({
    'path':str(CUDA_LOCK), 'device':_lock_stat.st_dev, 'inode':_lock_stat.st_ino,
    'guard_pid':os.getpid(), 'fd':cuda_lock_fd, 'child_inherits_fd':True,
}) + '\n')
started_at = dt.datetime.now(dt.timezone.utc).isoformat()
ps = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
script = '''$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction Stop;
$since=[datetime]::Parse('START_ISO').ToUniversalTime();
$events=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';StartTime=$since.ToLocalTime()} -ErrorAction SilentlyContinue | Where-Object {$_.TimeCreated.ToUniversalTime() -ge $since});
[pscustomobject]@{At=(Get-Date).ToUniversalTime().ToString('o');AvailableMBytes=$m.AvailableMBytes;CommittedBytes=$m.CommittedBytes;CommitLimit=$m.CommitLimit;PageReadsPersec=$m.PageReadsPersec;PagesInputPersec=$m.PagesInputPersec;PagesOutputPersec=$m.PagesOutputPersec;DriverEvents=@($events | Select-Object Id,@{n='At';e={$_.TimeCreated.ToUniversalTime().ToString('o')}})} | ConvertTo-Json -Depth 4
'''.replace('START_ISO', started_at)

def write(name, value):
    with (a.output / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

def observation():
    result = subprocess.run([ps, '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, 'Windows memory query failed'
    value = json.loads(result.stdout)
    assert isinstance(value.get('AvailableMBytes'), (int, float))
    assert value['CommitLimit'] > 0 and value['CommittedBytes'] >= 0
    events = value.get('DriverEvents')
    assert isinstance(events, list), 'driver event query did not return an array'
    start_dt = dt.datetime.fromisoformat(started_at)
    value['DriverEvents'] = [event for event in events
                            if dt.datetime.fromisoformat(event['At'].replace('Z', '+00:00')) >= start_dt]
    value['IgnoredHistoricalDriverEvents'] = len(events) - len(value['DriverEvents'])
    with (a.output / 'host-memory.jsonl').open('a') as f:
        f.write(json.dumps(value) + '\n'); f.flush(); os.fsync(f.fileno())
    return value

policy = HostMemoryPolicy(soft_mib=a.minimum_free_mib, hard_mib=4096, samples=2)

def reject(value, *, preflight=False):
    return policy.reason(value, preflight=preflight)

initial = observation()
reason = reject(initial, preflight=True)
write('preflight.json', {'at': started_at, 'minimum_free_mib': a.minimum_free_mib,
                         'initial': initial, 'reason': reason, 'command': command,
                         'memory_policy': {'admission_MiB': a.minimum_free_mib, 'soft_MiB': a.minimum_free_mib, 'consecutive_low_samples': 2, 'hard_MiB': 4096, 'sample_interval_seconds': 15},
                         'scope': 'Guard owns its launched process group and optional one-shot clean model file cache release after load; no file contents or persistent settings changed. One-shot VM compaction can affect unrelated WSL process latency.'})
if reason:
    write('terminal.json', {'status': 'not_launched', 'reason': reason})
    raise SystemExit(2)
start = time.monotonic()
interrupted = [False]
signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
with (a.output / 'process.log').open('xb') as log:
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(cuda_lock_fd,))
    write('launch.json', {'at': started_at, 'guard_pid': os.getpid(), 'child_pid': proc.pid,
                          'max_seconds': a.seconds, 'command': command})
    print(json.dumps({'guard_pid': os.getpid(), 'child_pid': proc.pid, 'output': str(a.output)}), flush=True)
    failures = 0
    try:
        while proc.poll() is None:
            if interrupted[0]: reason = 'guard_terminated'; break
            if time.monotonic() - start >= a.seconds: reason = 'wall_deadline'; break
            try:
                proc.wait(timeout=min(15, max(0.01, a.seconds - (time.monotonic() - start))))
                break
            except subprocess.TimeoutExpired: pass
            try:
                if a.release_cache_file is not None and not cache_attempts:
                    released = release_after_load(a.release_cache_file, a.output / 'process.log', proc.pid, attempts=cache_attempts)
                    if released is not None:
                        write('post-load-cache-release.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), **released})
                value = observation(); failures = 0
                if value['AvailableMBytes'] < a.minimum_free_mib:
                    detail = {'at': value['At'], 'linux_memory': Path('/proc/meminfo').read_text()}
                    with (a.output / 'low-memory-details.jsonl').open('a') as f:
                        f.write(json.dumps(detail) + '\n'); f.flush(); os.fsync(f.fileno())
                reason = reject(value)
                if reason: break
            except Exception as error:
                failures += 1
                with (a.output / 'query-errors.jsonl').open('a') as f:
                    f.write(json.dumps({'at':dt.datetime.now(dt.timezone.utc).isoformat(),
                                        'type':type(error).__name__, 'message':str(error)})+'\n')
                if failures >= 2: reason = 'host_monitor_unavailable'; break
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        write('terminal.json', {'at':dt.datetime.now(dt.timezone.utc).isoformat(),
              'status':'completed' if proc.returncode == 0 and reason is None else 'stopped_or_failed',
              'reason':reason,'child_exit_code':proc.returncode,'seconds':time.monotonic()-start})
print(json.dumps({'child_exit_code':proc.returncode,'reason':reason}),flush=True)
raise SystemExit(0 if proc.returncode == 0 and reason is None else 1)
