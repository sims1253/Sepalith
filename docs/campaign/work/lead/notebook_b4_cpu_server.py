"""Lead-owned protected b4 CPU endpoint for a bounded synthetic baseline."""
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
RUN = ROOT/'runs/b4-cpu-cycle-a'
BINARY = ROOT/'build-b10453-avx2/bin/llama-server'
MODEL = ROOT/'models/b4/packaging_b4-Q8_0.gguf'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

DEPENDENCIES = [{'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libggml-base.so.0.20.0', 'bytes': 952528, 'sha256': '4ffa2aefa2657ac24e9d7625dfa290c0aeb716a5884f4e88c00152eda9e32641'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libggml-cpu.so.0.20.0', 'bytes': 1052944, 'sha256': '848685c35953b489406dfefff51d118ac7520b4180742cf52f73bff03a1f48e9'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libggml.so.0.20.0', 'bytes': 55416, 'sha256': 'b1a9f396314229b99c598d55299fe4816f43c709f7a2e15af0cdf109ae79ef1a'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libllama-cli-impl.so', 'bytes': 446192, 'sha256': '825e455fa07fcdd5c1dc08bf8bcb592881767e1d838fc3bbd77cdef3678ca208'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libllama-common.so.0.1.0', 'bytes': 6439832, 'sha256': 'd159d9012e0b77c810979a9975c27ce0e83bc92179c7278a4c365b80d9ca04d8'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libllama-server-impl.so', 'bytes': 4522640, 'sha256': '4b8e02007d166bd7d0af1613a11ef750406a3654ea8aaf20d1ea45973aba6ad4'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libllama.so.0.1.0', 'bytes': 4426208, 'sha256': 'f8cffe232d4acc2407c51a13354b71ebe7801ffde530c290a8f054fa3a7c96cf'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/libmtmd.so.0.1.0', 'bytes': 2020216, 'sha256': '49dcf64dc2884ecfdc6ff9f9b4c7f3bc2d1e5ed789f52e500fe69909e01ab1e0'}, {'path': '/home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server', 'bytes': 16000, 'sha256': 'e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6'}]
for artifact in DEPENDENCIES:
    item=Path(artifact["path"])
    assert item.stat().st_size == artifact["bytes"] and sha(item)==artifact["sha256"]
assert sha(MODEL) == 'e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d'
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
signal.signal(signal.SIGHUP, on_signal)
argv = [str(BINARY), '-m', str(MODEL), '--host', '127.0.0.1', '--port', '18403',
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '8192', '-b', '256', '-ub', '256', '-ngl', '0', '-lv', '4']
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
        'task': 'RUN-03', 'owner': 'lead', 'supervisor_pid': os.getpid(),
        'server_pid': child.pid, 'argv': argv, 'max_seconds': 360,
        'model_sha256': sha(MODEL), 'binary_sha256': sha(BINARY), 'runtime_dependencies': DEPENDENCIES, 'host_loadavg': Path('/proc/loadavg').read_text(), 'host_meminfo': Path('/proc/meminfo').read_text(),
        'client_scope': 'Protected b4 legacy render_zeta2 synthetic 5-second request cycles and explicit diagnostics only; no editor application or quality claim.'})
    try:
        while child.poll() is None:
            if interrupted or (RUN/'stop').exists():
                reason = 'lead_stop'; break
            if time.monotonic()-started > 360:
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
            'acceptance': 'Root must verify legacy parser, full-cycle denominator, CPU backend and artifact identity; no editor or quality claim.'})
