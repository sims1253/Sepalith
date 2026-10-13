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
p.add_argument('--backend', choices=['openblas', 'vulkan'], required=True)
p.add_argument('--build-receipt', type=Path, required=True)
p.add_argument('--build-receipt-sha256', required=True)
p.add_argument('--run-name', required=True)
p.add_argument('--batch', type=int, choices=[64,128,256], required=True)
a = p.parse_args()
root = Path.home() / '.local/share/sepalith-campaign-20260915'
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
assert sha(a.build_receipt) == a.build_receipt_sha256
build = json.loads(a.build_receipt.read_text())
assert build['backend'] == a.backend and build['exit_code'] == 0 and build['source_unchanged']
for artifact in build['artifacts']:
    assert sha(artifact['path']) == artifact['sha256']
binary = root / ('build-b10453-' + a.backend + '-avx2/bin/llama-server')
assert any(x['path'] == str(binary) for x in build['artifacts'])
model = root / 'models/SFT-primary-step500-runtime/model-Q8_0.gguf'
assert sha(model) == 'f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'
probe = root / 'probe-v2/runtime_native_probe.py'
assert sha(probe) == 'f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7'
fixture = probe.parent / 'native-probe-train-fixture.jsonl'
manifest = probe.parent / 'native-probe-train-fixture.manifest.json'
assert sha(fixture) == '4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008'
assert sha(manifest) == '0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3'
transition = probe.parent / 'runtime_transition_probe.py'
assert sha(transition) == 'f7cf0a0b8d0340c95f7735e602f87f5107516d9c95f859a2033bfd66eb8e4306'
transition_manifest = probe.parent / 'runtime-transition-probe.manifest.json'
assert sha(transition_manifest) == '9e147141c65935e887280711d26fb6289097596ecfa05f761d0b5fbd951699b7'
baseline = root / 'runs/primary500-vulkan-a/probe-60000ms.json'
assert sha(baseline) == '2204fae5b02530de509a14e249032354824378053d3d0f07a8ba267c1d21bb71'
port = 18402 if a.backend == 'openblas' else 18403
with socket.socket() as sock:
    sock.bind(('127.0.0.1', port))
run = root / 'runs' / a.run_name
run.mkdir(parents=True, exist_ok=False)
def save(name, value):
    with (run / name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6',
           OPENBLAS_NUM_THREADS='6' if a.backend == 'openblas' else '1', MKL_NUM_THREADS='1')
argv = ['timeout', '--signal=TERM', '--kill-after=10s', '740s', str(binary),
        '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '4096', '-b', str(a.batch), '-ub', str(a.batch), '-lv', '4', '-ngl', '0' if a.backend == 'openblas' else '99']
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
    save('launch.json', {'owner':'lead','task':'RUN-01/RUN-06','at':dt.datetime.now(dt.timezone.utc).isoformat(),
         'supervisor_pid':os.getpid(),'server_timeout_pid':proc.pid,'argv':argv,'backend':a.backend,
         'build_receipt_sha256':a.build_receipt_sha256,'supervisor_sha256':sha(Path(__file__)),'probe_sha256':sha(probe),'model_sha256':sha(model),
         'threads':6,'batch_threads':6,'openblas_threads':env['OPENBLAS_NUM_THREADS'],
         'batch':a.batch,'transition_probe_sha256':sha(transition),'transition_manifest_sha256':sha(transition_manifest),'baseline_sha256':sha(baseline),'measurement_scope':'Four TRAIN mechanical fixtures, bounded cancellation and exact retry; no real editor transition or quality promotion.',
         'known_probe_v2_metadata_issue':'deadline_scope text hardcodes 5000ms; numeric user_deadline_ms and actual CLI are authoritative.'})
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, 'server exited during load'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    if response.status == 200: break
            except Exception:
                if time.monotonic() >= deadline: raise
                time.sleep(.5)
        children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
        device_audit = []
        for child in children:
            cp = Path('/proc') / child
            maps = [line for line in (cp / 'maps').read_text().splitlines()
                    if any(s in line for s in ['libggml-vulkan', 'libvulkan', 'vulkan_radeon'])]
            fds = []
            for fd in (cp / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if '/dev/dri/' in target: fds.append(target)
                except OSError: pass
            device_audit.append({'pid':int(child),'mapped_backend_libraries':maps,'dri_fds':fds})
        save('live-device-audit.json', {'at':dt.datetime.now(dt.timezone.utc).isoformat(),
             'server':device_audit,'batch':a.batch})
        if a.backend == 'vulkan':
            assert any(x['dri_fds'] and any('libggml-vulkan' in s for s in x['mapped_backend_libraries'])
                       for x in device_audit), 'No live Vulkan backend and DRI device proof'
        out = run / 'transition.json'
        command = ['timeout','--signal=TERM','--kill-after=10s','650s','python3',str(transition),
                   '--url',f'http://127.0.0.1:{port}','--batch',str(a.batch),
                   '--baseline-results',str(baseline),'--out',str(out)]
        with (run / 'client.log').open('xb') as clientlog:
            result = subprocess.run(command, env=env, stdout=clientlog, stderr=subprocess.STDOUT)
        results.append({'batch':a.batch,'exit_code':result.returncode,'output':str(out),
                        'sha256':sha(out) if out.exists() else None})
        assert result.returncode == 0 and out.exists(), 'Transition client failed; preserve partials'
        assert proc.poll() is None, 'server exited during probe'
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        stop.set(); thread.join(timeout=2)
        save('terminal.json', {'at':dt.datetime.now(dt.timezone.utc).isoformat(),'server_exit_code':proc.returncode,
             'seconds':time.monotonic()-started,'peak_server_rss_kib':peak_rss_kib[0],'probes':results})
