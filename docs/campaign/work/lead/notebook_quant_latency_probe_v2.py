#!/usr/bin/env python3
"""Root-owned Q8 Vulkan A/B screen for the server backend sampler.

This preparation keeps the native four-row probe as the client contract.  The
target arm changes one server startup option, ``--backend-sampling``; it does
not change the stored token-ID request body.  Root owns the actual launch.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request


p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['vulkan'], default='vulkan')
p.add_argument('--build-receipt', type=Path, required=True)
p.add_argument('--build-receipt-sha256', required=True)
p.add_argument('--run-name', required=True)
p.add_argument('--candidate', choices=['Q8_0', 'Q6_K'], default='Q8_0')
sampling_group = p.add_mutually_exclusive_group()
sampling_group.add_argument('--backend-sampling', dest='backend_sampling',
                            action='store_true',
                            help='run the target arm with the server --backend-sampling flag')
sampling_group.add_argument('--no-backend-sampling', dest='backend_sampling',
                            action='store_false',
                            help='run the baseline arm; omit the server sampling flag')
p.set_defaults(backend_sampling=False)
p.add_argument('--port', type=int, default=18403)
p.add_argument('--deadline-ms', type=int, choices=[5000, 60000], required=True)
a = p.parse_args()
assert not a.backend_sampling, 'This quant comparison keeps backend sampling disabled'


root = Path.home() / '.local/share/sepalith-campaign-20260915'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


assert 1024 < a.port < 65536
assert sha(a.build_receipt) == a.build_receipt_sha256
build = json.loads(a.build_receipt.read_text())
assert build['backend'] == a.backend and build['exit_code'] == 0
assert build['source_unchanged']
for artifact in build['artifacts']:
    assert sha(artifact['path']) == artifact['sha256']

binary = root / 'build-b10453-vulkan-avx2/bin/llama-server'
assert any(x['path'] == str(binary) for x in build['artifacts'])
models = {'Q8_0': ('models/SFT-primary-step500-runtime/model-Q8_0.gguf', 'f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256'), 'Q6_K': ('models/SFT-primary-step500-quant-candidates/model-Q6_K.gguf', 'b11ffcc093b78261af1c5eb450feefcbca6ee35c0cecc22133236506410e143e')}
model = root / models[a.candidate][0]
assert sha(model) == models[a.candidate][1]
probe = root / 'probe-v2/runtime_native_probe.py'
assert sha(probe) == 'f4f556a046f801eb233106c6d773accef317eb83eae5e261c0e0102388ca51a7'
fixture = probe.parent / 'native-probe-train-fixture.jsonl'
manifest = probe.parent / 'native-probe-train-fixture.manifest.json'
assert sha(fixture) == '4081472e1e19457ab4b9e186858e837fbf37f9011d7f8d576a6298f3e6ddb008'
assert sha(manifest) == '0b2195b87b5f6eabc892164124c25b876833a00ba6add0d79af6826413f5d0b3'

source_root = root / 'llama.cpp-b10453'
source_points = {
    'backend_sampling_cli': (source_root / 'common/arg.cpp', 2296,
                             '--backend-sampling'),
    'backend_sampling_context': (source_root / 'tools/server/server-context.cpp', 1732,
                                 'task.params.sampling.backend_sampling'),
    'backend_sampling_request_schema': (source_root / 'tools/server/server-schema.cpp', 187,
                                        'backend_sampling'),
    'backend_sampling_incompatibility_gate': (source_root / 'common/sampling.cpp', 415,
                                              'not compatible with grammar'),
}
source_pins = {'common/arg.cpp': '566a8122afc02bbfcca572b7ac8d7d7d985b2b087b780434be75f588184f458d', 'tools/server/server-context.cpp': '26f130b76c27be72e4674943754575cf5efa14b6a6325591be07df57f651e681', 'tools/server/server-schema.cpp': 'ca6db259fadf6a8bfc182407d539398c803e4db4e288cabec9891990a875b946', 'common/sampling.cpp': 'f23f8d663932bfd73abac1bd5e99055fc6efb07e21ef27dd7e25af2f2da40bff'}
source_proof = {}
for name, (path, line, needle) in source_points.items():
    assert path.is_file(), f'pinned source file is missing: {path}'
    assert sha(path) == source_pins[str(path.relative_to(source_root))], f'pinned source changed: {path}'
    source_text = path.read_text(errors='replace')
    assert needle in source_text, f'pinned source marker is missing: {path}:{line}'
    source_proof[name] = {
        'path': str(path), 'sha256': sha(path), 'line': line, 'marker': needle,
    }

target_arm = a.candidate
sampling_flag = ['--backend-sampling'] if a.backend_sampling else []
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', a.port))

run = root / 'runs' / a.run_name
run.mkdir(parents=True, exist_ok=False)


def save(name, value):
    destination = run / name
    with destination.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
for key in list(env):
    if key.startswith(('GGML_VK_', 'GGML_CUDA_')) or key == 'GGML_BACKEND_PATH':
        del env[key]

argv = ['timeout', '--signal=TERM', '--kill-after=10s', '1100s', str(binary),
        '-m', str(model), '--host', '127.0.0.1', '--port', str(a.port),
        '-t', '6', '-tb', '6', '--threads-http', '2', '--parallel', '1',
        '-c', '4096', '-b', '256', '-ub', '256', '-lv', '4', '-ngl', '99',
        '-fa', 'on', *sampling_flag]

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


def sampling_log_proof(log_path):
    """Summarize safe markers; retain the log hash for root's final correlation."""
    text = log_path.read_text(errors='replace')
    lines = text.splitlines()
    true_pattern = re.compile(r'backend_sampling\s*["\']?\s*[:=]\s*(true|1)\b', re.I)
    false_pattern = re.compile(r'backend_sampling\s*["\']?\s*[:=]\s*(false|0)\b', re.I)
    disabled_pattern = re.compile(r'backend sampling.*(disable|unsupported)|'
                                  r'(disable|unsupported).*backend sampling', re.I)
    true_lines = [i + 1 for i, line in enumerate(lines) if true_pattern.search(line)]
    false_lines = [i + 1 for i, line in enumerate(lines) if false_pattern.search(line)]
    disabled_lines = [i + 1 for i, line in enumerate(lines) if disabled_pattern.search(line)]
    if a.backend_sampling:
        if true_lines and not disabled_lines:
            status = 'observed_enabled_marker'
        elif disabled_lines:
            status = 'runtime_disabled_or_incompatible'
        else:
            status = 'unproven_requires_log_review'
    else:
        status = 'baseline_flag_absent_requires_log_review'
    return {
        'target_arm': target_arm,
        'requested': a.backend_sampling,
        'server_log_sha256': sha(log_path),
        'backend_sampling_true_marker_lines': true_lines,
        'backend_sampling_false_marker_lines': false_lines,
        'backend_sampling_disable_marker_lines': disabled_lines,
        'effective_status': status,
        'source_gate': {
            'pre_sampling_logits': 'must_be_false',
            'grammar': 'must_be_absent',
            'reasoning_budget': 'must_be_absent',
            'why': 'The server disables backend sampling when pre-sampling logits, grammar, or reasoning budget is active.',
        },
        'raw_log_retained_for_root_review': True,
    }


def offload_log_proof(log_path):
    """Record 43/43 offload evidence without retaining server-log text."""
    lines = log_path.read_text(errors='replace').splitlines()
    pattern = re.compile(r'(?:43\s*/\s*43|43/43).*(?:offload|vulkan)|'
                         r'(?:offload|vulkan).*(?:43\s*/\s*43|43/43)', re.I)
    marker_lines = [index + 1 for index, line in enumerate(lines) if pattern.search(line)]
    return {
        'requested_ngl': 99,
        'expected_offloaded_layers': 43,
        'marker_lines': marker_lines,
        'status': 'observed_43_of_43' if marker_lines else 'unproven_requires_log_review',
        'server_log_sha256': sha(log_path),
        'raw_log_retained_for_root_review': True,
    }


server_log = run / 'server.log'
proc = None
thread = None
try:
    with server_log.open('xb') as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        signal.signal(signal.SIGTERM,
                      lambda *_: proc.terminate() if proc.poll() is None else None)
        thread = threading.Thread(target=memory_watch, args=(proc.pid,), daemon=True)
        thread.start()
        save('launch.json', {
            'owner': 'root',
            'task': 'RUN-05 backend-sampling preparation',
            'at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'supervisor_pid': os.getpid(),
            'server_timeout_pid': proc.pid,
            'argv': argv,
            'backend': a.backend,
            'target_arm': target_arm,
            'backend_sampling_requested': a.backend_sampling,
            'backend_sampling_cli_scope': 'server startup only; native probe request body is unchanged',
            'build_receipt_sha256': a.build_receipt_sha256,
            'supervisor_sha256': sha(Path(__file__)),
            'probe_sha256': sha(probe),
            'model_sha256': sha(model),
            'candidate': a.candidate,
            'speculative': False,
            'threads': 6,
            'batch_threads': 6,
            'threads_http': 2,
            'context': 4096,
            'batch': 256,
            'ubatch': 256,
            'parallel': 1,
            'flash_attention': True,
            'ngl': 99,
            'probe_client_arm': 'baseline',
            'source_proof': source_proof,
            'fixture_sha256': sha(fixture),
            'manifest_sha256': sha(manifest),
            'offload_expected': {'layers': 43, 'requested_ngl': 99},
            'measurement_scope': 'Four TRAIN native fixtures, step500 Q8/Q6 Vulkan, eight requests (one cold and one warm per row). Actual5s and diagnostic60s are separate fresh-server runs; no final promotion.',
            'exact_output_gate': {
                'required_before_latency_comparison': True,
                'comparison_tool': str(probe),
                'comparison_argv_template': [
                    'python3', str(probe), '--compare',
                    '<baseline-probe.json>', '<backend-sampling-probe.json>',
                    '--out', '<exact-output-comparison.json>'
                ],
                'comparison_keys': 'row_id, phase, repetition, returned token IDs, and raw text',
            },
        })
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, 'server exited during load'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.5)

        process_evidence = []
        for child in Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split():
            actual = Path('/proc') / child
            maps = (actual / 'maps').read_text().splitlines()
            libraries = sorted({line.split()[-1] for line in maps
                                if 'libggml' in line or 'libvulkan' in line})
            devices = []
            for fd in (actual / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if '/dev/dri/' in target:
                        devices.append(target)
                except OSError:
                    pass
            process_evidence.append({
                'pid': int(child),
                'libraries': libraries,
                'render_devices': sorted(set(devices)),
            })
        save('actual-backend.json', {
            'processes': process_evidence,
            'required_backend': 'vulkan',
            'proof_rule': 'At least one child must map a Vulkan library and hold a /dev/dri render device.',
            'verified': any(
                any('vulkan' in lib.lower() for lib in item['libraries']) and item['render_devices']
                for item in process_evidence
            ),
        })
        assert any(any('vulkan' in lib.lower() for lib in item['libraries']) and item['render_devices']
                   for item in process_evidence), 'Vulkan process proof missing'

        out = run / f'probe-{a.deadline_ms}ms.json'
        command = [
            'timeout', '--signal=TERM', '--kill-after=10s', '650s', 'python3', str(probe),
            '--fixture', str(fixture), '--manifest', str(manifest),
            '--url', f'http://127.0.0.1:{a.port}', '--arm', 'baseline',
            '--cap', '192', '--context', '4096', '--reps', '1',
            '--user-deadline-ms', str(a.deadline_ms), '--diagnostic-timeout-ms', str(a.deadline_ms),
            '--out', str(out),
        ]
        with (run / f'client-{a.deadline_ms}ms.log').open('xb') as clientlog:
            result = subprocess.run(command, env=env, stdout=clientlog,
                                    stderr=subprocess.STDOUT)
        results.append({
            'requests_expected': 8,
            'deadline_ms': a.deadline_ms,
            'diagnostic_timeout_ms': a.deadline_ms,
            'exit_code': result.returncode,
            'output': str(out),
            'sha256': sha(out) if out.exists() else None,
        })
        assert proc.poll() is None, 'server exited during probe'
        save('sampling-runtime-proof.json', sampling_log_proof(server_log))
        save('offload-runtime-proof.json', offload_log_proof(server_log))
finally:
    if proc is not None and proc.poll() is None:
        children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
        owned = [int(pid) for pid in children if Path(f'/proc/{pid}/exe').resolve() == binary.resolve()]
        assert len(owned) == 1, 'Cannot identify sole owned server for graceful termination'
        save('shutdown-request.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'server_pid': owned[0], 'signal': 'SIGTERM', 'scope': 'one signal directly to verified server child; wait on timeout parent'})
        os.kill(owned[0], signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    stop.set()
    if thread is not None:
        thread.join(timeout=2)
    if proc is not None:
        save('terminal.json', {
            'at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'server_exit_code': proc.returncode,
            'seconds': time.monotonic() - started,
            'peak_server_rss_kib': peak_rss_kib[0],
            'probes': results,
            'target_arm': target_arm,
        })
