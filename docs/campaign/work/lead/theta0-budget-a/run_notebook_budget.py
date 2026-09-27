#!/usr/bin/env python3
"""Lead-owned one-arm quality replay, bounded remote server and local client."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--candidate', choices=['Q8_0', 'F16', 'Q4_K_M', 'Q6_K'], required=True)
p.add_argument('--run-name', required=True)
p.add_argument('--budget', type=int, choices=[3000,1500], required=True)
p.add_argument('--context', type=int, choices=[4096,8192], default=4096)
a = p.parse_args()
assert all(c.isalnum() or c in '-_' for c in a.run_name)
plan = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
native = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
remote = '/home/m0hawk/.local/share/sepalith-campaign-20260915'
host = 'm0hawk@192.168.178.40'
client = plan / 'docs/campaign/work/lead/theta0-budget-a/replay-budget.ts'
server = plan / 'docs/campaign/work/lead/theta0-notebook-long-a/notebook_long_server.py'
py = '/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

assert sha(client) == '17f58d60ac66afb5da9d3910b51ea5a1bce169ca06956c41e4b67e5a2e16915f'
run = native / 'training' / a.run_name
run.mkdir(parents=True, exist_ok=False)

def save(name, value):
    with (run / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 18413))
remote_cmd = ['timeout', '--foreground', '--signal=TERM', '--kill-after=15s', '180s', 'python3',
              remote + '/prepared/notebook_long_server_a.py', '--backend', 'vulkan',
              '--build-receipt', remote + '/runs/backend-build-d/vulkan-terminal.json',
              '--build-receipt-sha256', '6f4b5c3064e74cd25013e142417aa6cc71aa66a7591d442a978d74f5a75ba7ed',
              '--candidate', a.candidate, '--run-name', a.run_name, '--context', str(a.context)]
argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=3', '-L', '127.0.0.1:18413:127.0.0.1:18403',
        host, shlex.join(remote_cmd)]
started = time.monotonic()
client_rc = None
error = None
with (run / 'ssh-stderr.log').open('xb') as err:
    ssh = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                           text=True, start_new_session=True)
    save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'task': 'RUN-05',
         'owner': 'lead', 'supervisor_pid': os.getpid(), 'ssh_pid': ssh.pid,
         'argv': argv, 'client_sha256': sha(client), 'server_supervisor_sha256': sha(server),
         'supervisor_sha256': sha(__file__), 'candidate': a.candidate,
         'source_budget_utf16_units': a.budget, 'scope': 'Six synthetic long-source events, selectedtheta0Q8. NotebookVulkan; local canonical client through SSH. No quality or editor-visible claim.'})
    print(json.dumps({'status': 'launched', 'candidate': a.candidate, 'run': str(run),
                      'supervisor_pid': os.getpid(), 'ssh_pid': ssh.pid}), flush=True)
    try:
        ready, _, _ = select.select([ssh.stdout], [], [], 100)
        assert ready, 'remote readiness timed out'
        line = ssh.stdout.readline()
        remote_ready = json.loads(line)
        assert remote_ready['status'] == 'ready' and remote_ready['candidate'] == a.candidate
        assert remote_ready['context'] == a.context
        assert remote_ready['supervisor_sha256'] == sha(server)
        save('remote-ready.json', remote_ready)
        print(json.dumps(remote_ready), flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                   HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '65s',
                   'node', '--no-warnings=MODULE_TYPELESS_PACKAGE_JSON', '--experimental-strip-types', str(client),
                   '--budget', str(a.budget), '--server', 'http://127.0.0.1:18413', '--context', str(a.context), '--out', str(run / 'trace.json')]
        with (run / 'client.log').open('xb') as log:
            client_rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if ssh.poll() is None:
            try: ssh.stdin.write('STOP\n'); ssh.stdin.flush(); ssh.stdin.close()
            except (BrokenPipeError, OSError): pass
            try: ssh.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ssh.terminate()
                try: ssh.wait(timeout=10)
                except subprocess.TimeoutExpired: ssh.kill(); ssh.wait()
        for name in ['launch.json', 'live-device-audit.json', 'props.json', 'terminal.json', 'server.log']:
            result = subprocess.run(['scp', '-q', '-o', 'ConnectTimeout=8',
                       f'{host}:{remote}/runs/{a.run_name}/{name}', str(run / ('remote-' + name))],
                       capture_output=True, timeout=30)
            if result.returncode and error is None: error = 'remote receipt retrieval failed: ' + name
        save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
             'seconds': time.monotonic() - started, 'client_exit_code': client_rc,
             'ssh_exit_code': ssh.returncode, 'error': error,
             'artifacts': {f.name: sha(f) for f in run.iterdir() if f.is_file()}})
print(json.dumps({'candidate': a.candidate, 'client_exit_code': client_rc,
                  'ssh_exit_code': ssh.returncode, 'error': error}), flush=True)
raise SystemExit(0 if client_rc == 0 and ssh.returncode == 0 and error is None else 1)
