#!/usr/bin/env python3
"""Root-owned isolated backend screen, with durable identities and deadlines."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--graph-opt', choices=['0','1'], required=True)
p.add_argument('--run-name', required=True)
a = p.parse_args()
root = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
plan = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
binroot = Path('/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
for pid in [3273988,3274675,3274676]:
    assert not Path(f'/proc/{pid}').exists(), 'SFT C is still active'
# This hardcoded previous-owner guard is supplementary. Root must inspect all
# actual CUDA workloads and record a fresh exclusive lease before each launch.
binary = binroot / 'llama-server'
assert sha(binary) == 'e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee'
assert sha(binroot / 'libggml-cuda.so.0.20.0') == '3c96a25c15a70fbd568ce77599fe628fdc50c7ce27bc9ebcac297692537e9a78'
model = root / 'models/SFT-primary-step500-runtime-gguf/model-Q8_0.gguf'
assert sha(model) == 'f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'
probe = plan / 'docs/campaign/work/serving-readiness/runtime_native_probe.py'
assert sha(probe) == 'f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7'
fixture = probe.parent / 'native-probe-train-fixture.jsonl'
manifest = probe.parent / 'native-probe-train-fixture.manifest.json'
assert sha(fixture) == '4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008'
assert sha(manifest) == '0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3'
port = 18404
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
run = root / 'training' / a.run_name
run.mkdir(parents=True, exist_ok=False)
def save(name, value):
    with (run / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='6',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', GGML_CUDA_GRAPH_OPT=a.graph_opt)
for key in ['GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_FORCE_MMQ', 'GGML_CUDA_FORCE_CUBLAS', 'GGML_BACKEND_PATH']:
    env.pop(key, None)
argv = ['timeout', '--signal=TERM', '--kill-after=10s', '140s', str(binary),
        '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '4096', '-b', '256', '-ub', '256', '-lv', '4', '-ngl', '99']
started = time.monotonic()
results = []
peak_rss_kib = [0]
stop = threading.Event()
def memory_watch(parent):
    while not stop.wait(1):
        try:
            children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
            for child in children:
                for line in Path(f'/proc/{child}/status').read_text().splitlines():
                    if line.startswith(('VmHWM:', 'VmRSS:')):
                        peak_rss_kib[0] = max(peak_rss_kib[0], int(line.split()[1]))
        except (OSError, ValueError):
            pass
with (run / 'server.log').open('xb') as log:
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    signal.signal(signal.SIGTERM, lambda *_: proc.terminate() if proc.poll() is None else None)
    thread = threading.Thread(target=memory_watch, args=(proc.pid,), daemon=True); thread.start()
    save('launch.json', {'owner':'lead','task':'RUN-05','at':dt.datetime.now(dt.timezone.utc).isoformat(),
         'supervisor_pid':os.getpid(),'server_timeout_pid':proc.pid,'argv':argv,'backend':'cuda',
         'graph_opt':a.graph_opt,'cuda_backend_sha256':sha(binroot / 'libggml-cuda.so.0.20.0'),'supervisor_sha256':sha(Path(__file__)),'probe_sha256':sha(probe),'model_sha256':sha(model),
         'threads':6,'batch_threads':6,'openblas_threads':env['OPENBLAS_NUM_THREADS'],
         'measurement_scope':'Four TRAIN mechanical fixtures, cold and identical cached replay; no development quality or final promotion.',
         'known_probe_v2_metadata_issue':'deadline_scope text hardcodes 5000ms; numeric user_deadline_ms and actual CLI are authoritative.'})
    try:
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, 'server exited during load'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    if response.status == 200: break
            except Exception:
                if time.monotonic() >= deadline: raise
                time.sleep(.5)
        for ms in (5000,):
            out = run / f'probe-{ms}ms.json'
            command = ['timeout','--signal=TERM','--kill-after=10s','65s','python3',str(probe),
                       '--fixture',str(fixture),'--manifest',str(manifest),'--url',f'http://127.0.0.1:{port}',
                       '--arm','baseline','--user-deadline-ms',str(ms),'--diagnostic-timeout-ms','5000','--out',str(out)]
            with (run / f'client-{ms}ms.log').open('xb') as clientlog:
                result = subprocess.run(command, env=env, stdout=clientlog, stderr=subprocess.STDOUT)
            results.append({'deadline_ms':ms,'exit_code':result.returncode,'output':str(out),
                            'sha256':sha(out) if out.exists() else None})
            assert proc.poll() is None, 'server exited during probe'
            time.sleep(2)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        stop.set(); thread.join(timeout=2)
        save('terminal.json', {'at':dt.datetime.now(dt.timezone.utc).isoformat(),'server_exit_code':proc.returncode,
             'seconds':time.monotonic()-started,'peak_server_rss_kib':peak_rss_kib[0],'probes':results})
