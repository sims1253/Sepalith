#!/usr/bin/env python3
"""Root-owned CPU comparison with independent server/client deadlines."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

plan = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
run = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-08-b4-dev-v3')
run.mkdir(parents=True, exist_ok=False)
client = plan / 'docs/campaign/work/b4-dev/run03_b4_dev_client.py'
assert hashlib.sha256(client.read_bytes()).hexdigest() == '77e1ef0d7a1a45c355266c89cd4bb7f65f5055257b7802f7d7218e117ba93820'
server = '/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server'
model = '/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf'
with socket.socket() as sock: sock.bind(('127.0.0.1', 18099))
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', SEPALITH_B4_DEV_HARD_DEADLINE_S='2400', SEPALITH_B4_DEV_CHECKPOINT_RESERVE_S='60')
argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '2500s', server,
        '-m', model, '--alias', 'sepalith', '--temp', '0', '--host', '127.0.0.1',
        '--port', '18099', '-c', '8192', '--parallel', '1', '-t', '2', '-tb', '2', '-ngl', '0']
def save(name, obj):
    with (run / name).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
started = time.monotonic()
with (run / 'server.log').open('xb') as log:
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save('launch.json', {'owner':'lead', 'task':'SFT-08/RUN-03', 'at':dt.datetime.now(dt.timezone.utc).isoformat(),
         'supervisor_pid':os.getpid(), 'server_timeout_pid':proc.pid, 'argv':argv,
         'model_sha256':'e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d',
         'client_sha256':hashlib.sha256(client.read_bytes()).hexdigest(), 'cpu_only':True})
    try:
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, 'CPU server exited during load'
            try:
                with urllib.request.urlopen('http://127.0.0.1:18099/health', timeout=1) as r:
                    if r.status == 200: break
            except Exception:
                if time.monotonic() >= deadline: raise
                time.sleep(.5)
        with (run / 'client.log').open('xb') as out:
            result = subprocess.run(['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '2400s',
                '/usr/bin/python3', str(client), '--url', 'http://127.0.0.1:18099',
                '--resume-successful-jsonl', '/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-08-b4-dev-v2/per_request.jsonl', '--resume-summary', '/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT-08-b4-dev-v2/summary.json', '--soft-deadline-s', '2340', '--output', str(run / 'per_request.jsonl'), '--summary', str(run / 'summary.json')],
                env=env, stdout=out, stderr=subprocess.STDOUT)
        save('client-terminal.json', {'exit_code':result.returncode, 'seconds':time.monotonic()-started})
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        save('server-terminal.json', {'exit_code':proc.returncode, 'seconds':time.monotonic()-started,
             'at':dt.datetime.now(dt.timezone.utc).isoformat()})
