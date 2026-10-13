#!/usr/bin/env python3
"""One root-owned, bounded CPU server; no model or port discovery."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=['baseline', 'ngram-mod'], required=True)
p.add_argument('--run-name', required=True)
a = p.parse_args()
root = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
binary = root / 'build-b10453-avx2/bin/llama-server'
model = root / 'models/SFT-primary-step500-runtime/model-Q8_0.gguf'
assert hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() == 'e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6'
assert hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() == 'f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'
with socket.socket() as s:
    s.bind(('127.0.0.1', 18401))
run = root / 'runs' / a.run_name
run.mkdir(parents=True, exist_ok=False)
argv = [str(binary), '-m', str(model), '--host', '127.0.0.1', '--port', '18401',
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '4096', '-b', '256', '-ub', '256', '-ngl', '0']
if a.arm == 'ngram-mod':
    argv += ['--spec-type', 'ngram-mod', '--spec-draft-n-max', '64',
             '--spec-ngram-mod-n-match', '24', '--spec-ngram-mod-n-min', '48',
             '--spec-ngram-mod-n-max', '64']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
def save(name, value):
    with (run / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
started = time.monotonic()
with (run / 'server.log').open('xb') as log:
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    save('launch.json', dict(owner='lead', task='RUN-01/RUN-06', pid=proc.pid,
         supervisor_pid=os.getpid(), at=dt.datetime.now(dt.timezone.utc).isoformat(),
         argv=argv, max_seconds=1200, model_sha256='f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'))
    signal.signal(signal.SIGTERM, lambda *_: proc.terminate() if proc.poll() is None else None)
    try:
        code = proc.wait(timeout=1200)
        reason = 'server_exit'
    except subprocess.TimeoutExpired:
        reason = 'hard_deadline'; proc.terminate()
        try: code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); code = proc.wait(timeout=10)
    save('terminal.json', dict(exit_code=code, reason=reason,
         at=dt.datetime.now(dt.timezone.utc).isoformat(), seconds=time.monotonic()-started))
