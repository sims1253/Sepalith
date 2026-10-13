"""Sequential CPU export under the campaign's separate host-memory guard."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PREP = PLAN / 'docs/campaign/receipts/RUN-01-theta0-export-preparation.json'
spec = json.loads(PREP.read_text())
parent = Path(spec['parent'])
output = Path(spec['output_new_only'])
rl_guard = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c-host-supervision')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    with (output / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU environment required')
own_guard = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-01-theta0-export-a-host-supervision')
guard_preflight = json.loads((own_guard / 'preflight.json').read_text())
require(guard_preflight['reason'] is None and
        guard_preflight['initial']['AvailableMBytes'] >= 16384,
        'Export requires at least16GiB Windows admission with active12GiB soft guard')
require(b'host_memory_guard_v3.py' in Path(f'/proc/{os.getppid()}/cmdline').read_bytes(),
        'Export must run directly under its host-memory guard')
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
require(int(mem['MemAvailable'].split()[0]) >= 12 * 1024 * 1024,
        'At least12GiB available inside WSL required')
terminal = json.loads((rl_guard / 'terminal.json').read_text())
require(terminal.get('status') == 'completed' and terminal.get('child_exit_code') == 0,
        'RL terminal acceptance is required before export')
launch = json.loads((rl_guard / 'launch.json').read_text())
for key in ('guard_pid', 'child_pid'):
    require(not Path(f'/proc/{launch[key]}').exists(), 'Previous RL owner still exists')
require(not output.exists(), 'Fresh output directory required')
require(shutil.disk_usage(parent).free >= (100 + 12) * (1 << 30), 'Storage headroom gate')
pins = {
    parent / 'parent-manifest.json': spec['parent_manifest_sha256'],
    parent / 'model.safetensors': spec['weights_sha256_expected'],
    parent / 'tokenizer.json': '3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
    parent / 'tokenizer_config.json': 'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
    Path(spec['converter']['path']): spec['converter']['sha256'],
    Path(spec['quantizer']['path']): spec['quantizer']['sha256'],
}
for path, expected in pins.items():
    require(sha(path) == expected, f'Input hash mismatch: {path.name}')
output.mkdir()
started = time.monotonic()
save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'pid': os.getpid(), 'task': 'RUN-01/RUN-09',
                    'preparation_sha256': sha(PREP),
                    'input_pins': {str(k): v for k, v in pins.items()},
                    'command_seconds_each': 240, 'outer_host_guard_required': True})
results = []
status = 'failed'
error = None
try:
    for index, command in enumerate(spec['commands']):
        command_started = time.monotonic()
        with (output / f'command-{index}.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save(f'command-{index}-launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                               'pid': child.pid, 'command': command})
            try:
                code = child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                raise RuntimeError(f'Command {index} deadline exceeded')
        results.append({'index': index, 'exit_code': code,
                        'seconds': time.monotonic() - command_started,
                        'cumulative_child_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
        save(f'command-{index}-terminal.json', results[-1])
        require(code == 0, f'Command {index} failed')
    artifacts = []
    for name in ('model-F16.gguf', 'model-Q8_0.gguf', 'model-Q6_K.gguf'):
        path = output / name
        require(path.is_file() and path.stat().st_size > 1_000_000_000,
                f'Expected complete artifact: {name}')
        with path.open('rb') as stream:
            require(stream.read(4) == b'GGUF', f'GGUF magic: {name}')
            os.fsync(stream.fileno())
        artifacts.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    save('artifacts.json', {'artifacts': artifacts,
                           'acceptance': 'Export integrity only. Tensor/tokenizer checks and DEV/native quality remain.'})
    status = 'exported_pending_root_review'
except Exception as exc:
    error = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                           'status': status, 'seconds': time.monotonic() - started,
                           'commands': results, 'error': error})
