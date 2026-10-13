"""Bounded, separate baseline/candidate Q6 builds. No model/server launch."""
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
SOURCE = ROOT / 'llama.cpp-b10453'
RUN = ROOT / 'runs/q6-kernel-build-a'
PACKET = ROOT / 'q6-kernel-build-a-packet'
COMMIT = '3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70'
PATCH_SHA = 'f9f57597ae119e433095c52e3de63b83a0cdbf472f982a123389256a01763310'
SOURCE_SHA = 'c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f'
SHADER_SHA = 'bc785a2457aa5b04d416e18f5ae2304da267e35cea7b54ac67b5d20987e450af'
CPP = 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
Q6 = 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save(name, data):
    with (RUN / name).open('x') as f:
        json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
assert sha(SOURCE / CPP) == SOURCE_SHA and sha(SOURCE / Q6) == SHADER_SHA
assert sha(PACKET / 'candidate.patch') == PATCH_SHA
for port in (18401, 18402, 18403):
    with socket.socket() as s: s.bind(('127.0.0.1', port))
RUN.mkdir(exist_ok=False)
os.sched_setaffinity(0, {0, 1})
start = time.monotonic()
interrupted = False
def interrupt(_sig, _frame):
    global interrupted
    interrupted = True
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, interrupt)
save('launch.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'owner': 'lead',
    'task': 'RUN-05', 'pid': os.getpid(), 'cpu_affinity': [0, 1], 'wall_seconds': 1800,
    'memory_floor_MiB': 2048, 'model_or_gpu_launch': False, 'source_commit': COMMIT,
    'source_clean': True, 'source_cpp_sha256': SOURCE_SHA, 'patch_sha256': PATCH_SHA})

def run(command, name, ceiling):
    begun = time.monotonic()
    reason = None
    with (RUN / (name + '.log')).open('xb') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(name + '-launch.json', {'argv': command, 'pid': proc.pid,
            'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'ceiling_seconds': ceiling})
        try:
            while proc.poll() is None:
                mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
                if interrupted or (RUN / 'stop').exists(): reason = 'lead_stop'; break
                if int(mem['MemAvailable'].split()[0]) < 2048 * 1024: reason = 'memory_floor'; break
                if time.monotonic() - start > 1800 or time.monotonic() - begun > ceiling:
                    reason = 'deadline'; break
                time.sleep(1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
    result = {'exit_code': proc.returncode, 'reason': reason, 'seconds': time.monotonic() - begun}
    save(name + '-terminal.json', result)
    if proc.returncode or reason: raise RuntimeError(name + ' failed: ' + str(result))

results = []
error = None
try:
    archive = RUN / 'source.tar'
    subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=tar', '-o', str(archive), COMMIT], check=True, timeout=30)
    settings = {}
    for line in (ROOT / 'build-b10453-vulkan-avx2/CMakeCache.txt').read_text().splitlines():
        if ':BOOL=' in line and line.startswith(('GGML_', 'LLAMA_')):
            k, v = line.split(':BOOL=', 1); settings[k] = v
    settings.update(CMAKE_BUILD_TYPE='Release', LLAMA_BUILD_NUMBER='10453',
        LLAMA_BUILD_COMMIT='3cb7ffb1a', GGML_VULKAN_DEBUG='ON',
        Vulkan_INCLUDE_DIR=str(ROOT / 'deps/vulkan-headers-1.4.357/usr/include'),
        Vulkan_GLSLC_EXECUTABLE='/usr/bin/glslc',
        **{'SPIRV-Headers_DIR': str(ROOT / 'deps/spirv-headers-1.4.357/usr/share/cmake/SPIRV-Headers')})
    for arm in ('baseline', 'candidate'):
        src = RUN / (arm + '-source'); src.mkdir()
        subprocess.run(['tar', '-xf', str(archive), '-C', str(src)], check=True, timeout=30)
        assert sha(src / CPP) == SOURCE_SHA and sha(src / Q6) == SHADER_SHA
        # Both arms use identical optional tracing. Without this env variable,
        # no debug stream formatting is performed; the default release binary
        # is untouched. Trace measurements remain separate from timing runs.
        cpp = src / CPP
        old = '#define VK_LOG_DEBUG(msg) std::cerr << msg << std::endl'
        new = '#define VK_LOG_DEBUG(msg) do { static const bool sepalith_trace = std::getenv("SEPALITH_VK_TRACE") != nullptr; if (sepalith_trace) { std::cerr << msg << std::endl; } } while (false)'
        text = cpp.read_text(); assert text.count(old) == 1
        cpp.write_text(text.replace(old, new).replace('#include <algorithm>', '#include <cstdlib>\n#include <algorithm>', 1))
        if arm == 'candidate':
            with (PACKET / 'candidate.patch').open('rb') as inp:
                subprocess.run(['patch', '-p1', '-d', str(src)], stdin=inp, check=True, timeout=10)
            shader = src / Q6
            text = shader.read_text(); assert text.startswith('#version 450\n')
            shader.write_text(text.replace('#version 450\n', '#version 450\n#define GGML_VK_Q6K_EXACT_DIV 1\n', 1))
        save(arm + '-source-identity.json', {'archive_sha256': sha(archive),
            'cpp_sha256': sha(cpp), 'q6_shader_sha256': sha(src / Q6), 'settings': settings,
            'changes': ['identical default-off runtime debug switch'] + (['reviewed Q6 patch and compile define'] if arm == 'candidate' else [])})
        build = RUN / (arm + '-build')
        run(['cmake', '-S', str(src), '-B', str(build), '-G', 'Ninja',
             *[f'-D{k}={v}' for k, v in sorted(settings.items())]], arm + '-configure', 120)
        run(['cmake', '--build', str(build), '--target', 'llama-server', '--parallel', '2'], arm + '-build', 780)
        artifacts = [{'path': str(f), 'bytes': f.stat().st_size, 'sha256': sha(f)}
            for f in sorted((build / 'bin').iterdir()) if f.is_file() and not f.is_symlink()]
        result = {'arm': arm, 'status': 'compiled_only', 'artifacts': artifacts,
            'cache_sha256': sha(build / 'CMakeCache.txt')}
        save(arm + '-artifacts.json', result); results.append(result)
except Exception as exc:
    error = type(exc).__name__ + ': ' + str(exc)
finally:
    save('terminal.json', {'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'seconds': time.monotonic() - start, 'status': 'compiled_only' if error is None else 'failed',
        'error': error, 'completed_arms': [x['arm'] for x in results],
        'acceptance': 'No dispatch, token parity, numerical correctness or latency claim until root checks.',
        'protected_source_unchanged': sha(SOURCE / CPP) == SOURCE_SHA and sha(SOURCE / Q6) == SHADER_SHA})
if error: raise SystemExit(1)
