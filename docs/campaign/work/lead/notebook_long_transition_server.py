"""Lead-owned Q8 Vulkan endpoint for one bounded synthetic transition panel."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

ROOT = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
RUN = ROOT/'runs/transition-long-v1-q8-a'
BINARY = ROOT/'build-b10453-vulkan-avx2/bin/llama-server'
MODEL = ROOT/'models/SFT-primary-step500-runtime/model-Q8_0.gguf'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

assert sha(BINARY) == '92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f'
assert sha(MODEL) == 'f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 18403))
RUN.mkdir(exist_ok=False)

def save(name, value):
    with (RUN/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

interrupted = False
def on_signal(_sig, _frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
argv = [str(BINARY), '-m', str(MODEL), '--host', '127.0.0.1', '--port', '18403',
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '4096', '-b', '256', '-ub', '256', '-ngl', '99', '-lv', '4']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
for key in ['GGML_BACKEND_PATH', 'GGML_CUDA_GRAPH_OPT']:
    env.pop(key, None)
started = time.monotonic()
reason = 'server_exit'
with (RUN/'server.log').open('xb') as log:
    child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'task': 'RUN-04', 'owner': 'lead', 'supervisor_pid': os.getpid(),
        'server_pid': child.pid, 'argv': argv, 'max_seconds': 480,
        'model_sha256': sha(MODEL), 'binary_sha256': sha(BINARY),
        'client_scope': 'Canonical six-event long synthetic panel via a lead-owned LAN SSH forward; no editor application.'})
    try:
        while child.poll() is None:
            if interrupted or (RUN/'stop').exists():
                reason = 'lead_stop'; break
            if time.monotonic()-started > 480:
                reason = 'hard_deadline'; break
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0]) < 1024*1024:
                reason = 'memory_floor_1GiB'; break
            time.sleep(.5)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'reason': reason, 'exit_code': child.returncode,
            'seconds': time.monotonic()-started,
            'server_pid_exists': Path(f'/proc/{child.pid}').exists(),
            'acceptance': 'Root must verify client denominators and actual Vulkan offload; exit alone is not acceptance.'})
