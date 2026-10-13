"""Quantize verified theta0 F16, releasing completed file reads between stages."""
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
BASE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
ADMISSION = PLAN / 'docs/campaign/receipts/RUN-01-theta0-quant-c-admission.json'
spec = json.loads(ADMISSION.read_text())
output = Path(spec['output'])
source = Path(spec['source']['path'])
guard = BASE / 'training/RUN-01-theta0-quant-c-host-supervision'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def hash_and_release(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        offset = 0
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
        after = os.fstat(stream.fileno())
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'File changed during hashing')
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


require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
require(b'host_memory_guard_v3.py' in Path(f'/proc/{os.getppid()}/cmdline').read_bytes(),
        'Direct host guard required')
preflight = json.loads((guard / 'preflight.json').read_text())
require(preflight['reason'] is None and preflight['initial']['AvailableMBytes'] >= 18432,
        'At least18GiB Windows memory at admission required')
require(shutil.disk_usage(BASE).free >= (100 + 6) * (1 << 30), 'Storage headroom gate')
require(not output.exists(), 'Fresh quant output required')
require(hash_and_release(source) == spec['source']['sha256'], 'F16 hash mismatch')
require(hash_and_release(Path(spec['quantizer']['path'])) == spec['quantizer']['sha256'],
        'Quantizer hash mismatch')
output.mkdir()
started = time.monotonic()
save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'pid': os.getpid(), 'admission': str(ADMISSION),
                    'source': spec['source'], 'commands': spec['commands'],
                    'cache_policy': 'Release closed input after each quant; fsync/hash output with consumed-range advisory.'})
results = []
status = 'failed'
error = None
try:
    for index, command in enumerate(spec['commands']):
        target = Path(spec['outputs'][index])
        require(target.parent == output and not target.exists(), 'Fresh owned target required')
        command_started = time.monotonic()
        with (output / f'command-{index}.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            save(f'command-{index}-launch.json', {'pid': child.pid, 'command': command})
            try:
                code = child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                raise RuntimeError(f'Quant {index} deadline exceeded')
        command_result = {'index': index, 'exit_code': code,
                          'seconds': time.monotonic() - command_started,
                          'cumulative_child_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        save(f'command-{index}-terminal.json', command_result)
        require(code == 0, f'Quant {index} failed')
        with source.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        with target.open('rb') as stream:
            require(stream.read(4) == b'GGUF', 'GGUF magic required')
            os.fsync(stream.fileno())
        artifact = {'path': str(target), 'bytes': target.stat().st_size,
                    'sha256': hash_and_release(target)}
        require(artifact['bytes'] > 1_000_000_000, 'Implausible artifact size')
        results.append({'command': command_result, 'artifact': artifact})
        save(f'quant-{index}-artifact.json', artifact)
    status = 'exported_pending_root_metadata_and_quality_review'
except Exception as exc:
    error = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
                           'status': status, 'seconds': time.monotonic() - started,
                           'results': results, 'error': error})
