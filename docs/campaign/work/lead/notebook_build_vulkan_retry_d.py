#!/usr/bin/env python3
"""Build two root-admitted backends sequentially, without launching a model."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

root = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
source = root / 'llama.cpp-b10453'
manifest = root / 'llama-b10453-source-files.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == 'b0e6fbb71b2528924480c6da34f9bd90c87f7b847c504b7e3ee8132fb446bc7e'
files = json.loads(manifest.read_text())['files']
def verify_source():
    for row in files:
        p = source / row['path']
        if 'symlink' in row:
            assert p.is_symlink() and str(p.readlink()) == row['symlink'], row['path']
        else:
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], row['path']
verify_source()
for port in [18401, 18402, 18403]:
    with socket.socket() as s: s.bind(('127.0.0.1', port))
run = root / 'runs/backend-build-d'
run.mkdir(parents=True, exist_ok=False)
def save(name, obj):
    with (run / name).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
save('launch.json', {'owner':'lead', 'task':'RUN-01/RUN-06', 'pid':os.getpid(),
     'at':dt.datetime.now(dt.timezone.utc).isoformat(), 'source_files_verified':len(files),
     'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
     'cpu_affinity':list(os.sched_getaffinity(0)), 'models_launched':False})
cache = (root / 'build-b10453-avx2/CMakeCache.txt').read_text().splitlines()
common = {}
for line in cache:
    if ':BOOL=' in line and line.startswith(('GGML_', 'LLAMA_')):
        key, value = line.split(':BOOL=', 1); common[key] = value
common.update(CMAKE_BUILD_TYPE='Release', LLAMA_BUILD_NUMBER='10453',
              LLAMA_BUILD_COMMIT='3cb7ffb1a', GGML_LLAMAFILE='ON')
archive = Path('/var/cache/pacman/pkg/vulkan-headers-1:1.4.357.0-1-any.pkg.tar.zst')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '2f6c34cc829c4b63c0cf08cf147841c8ded023b746324540fb02322d9c415c07'
deps = root / 'deps/vulkan-headers-1.4.357'
assert deps.is_dir()
assert (deps / 'usr/include/vulkan/vulkan.h').is_file()
assert (deps / 'usr/include/vulkan/vulkan.h').is_file()
common['Vulkan_INCLUDE_DIR'] = str(deps / 'usr/include')
spirv_archive = Path('/var/cache/pacman/pkg/spirv-headers-1:1.4.357.0-1-any.pkg.tar.zst')
assert hashlib.sha256(spirv_archive.read_bytes()).hexdigest() == 'f27479489eae94ad391f6437dd9207afccd744bb65491aec44a8f021353e295a'
spirv_deps = root / 'deps/spirv-headers-1.4.357'
assert spirv_deps.is_dir()
# ggml-vulkan includes SPIRV headers directly without adding its imported include root.
# Use the same combined include prefix layout as a normal system package installation.
assert not (deps / 'usr/include/spirv').exists()
subprocess.run(['tar', '-xf', str(spirv_archive), '-C', str(deps), 'usr/include'], check=True, timeout=60)
assert (deps / 'usr/include/spirv/unified1/spirv.hpp').is_file()
common['SPIRV-Headers_DIR'] = str(spirv_deps / 'usr/share/cmake/SPIRV-Headers')
results = []
for backend, seconds in [('vulkan', 1500)]:
    build = root / f'build-b10453-{backend}-avx2'
    previous = json.loads((root / 'runs/backend-build-a/vulkan-terminal.json').read_text())
    assert previous['exit_code'] == 1 and not (build / 'bin/llama-server').exists()
    settings = dict(common, GGML_BLAS='ON' if backend == 'openblas' else 'OFF',
                    GGML_BLAS_VENDOR='OpenBLAS', GGML_VULKAN='ON' if backend == 'vulkan' else 'OFF')
    argv = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja']
    argv += [f'-D{k}={v}' for k,v in sorted(settings.items())]
    started = time.monotonic()
    with (run / f'{backend}.log').open('xb') as log:
        configured = subprocess.run(['timeout', '--kill-after=10s', '120s', *argv],
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        code = configured
        if configured == 0:
            code = subprocess.run(['timeout', '--kill-after=10s', f'{seconds}s', 'cmake', '--build',
                str(build), '--target', 'llama-server', 'llama-cli', '--parallel', '2'],
                stdout=log, stderr=subprocess.STDOUT).returncode
    verify_source()
    artifacts = []
    if code == 0:
        for p in sorted((build / 'bin').iterdir()):
            if p.is_file() and not p.is_symlink():
                artifacts.append({'path':str(p), 'bytes':p.stat().st_size,
                                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result = {'backend':backend, 'exit_code':code, 'seconds':time.monotonic()-started,
              'settings':settings, 'artifacts':artifacts, 'source_unchanged':True,
              'status':'build_verified_only' if code == 0 else 'build_failed'}
    save(f'{backend}-terminal.json', result); results.append(result)
save('terminal.json', {'at':dt.datetime.now(dt.timezone.utc).isoformat(),
     'results':[{'backend':r['backend'], 'exit_code':r['exit_code'], 'seconds':r['seconds']} for r in results]})
