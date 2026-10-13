"""Root-run read-only target preflight. Writes JSON to stdout; never builds."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    plan = json.loads(Path(sys.argv[1]).read_text())
    expected = plan['expected']
    build = Path(expected['protected_build'])
    files = {}
    calls = []
    unresolved = []

    def record_file(path):
        path = Path(path)
        if not path.is_absolute():
            path = build / path
        key = str(path)
        if key in files:
            return files[key]
        if not path.is_file():
            raise RuntimeError('Missing required input: ' + key)
        digest = hashlib.sha256()
        with path.open('rb') as inp:
            for block in iter(lambda: inp.read(1 << 20), b''):
                digest.update(block)
        row = {'path': key, 'resolved_path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        files[key] = row
        return row

    def run(argv):
        result = subprocess.run(argv, cwd=build, text=True, capture_output=True, timeout=30)
        calls.append({'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError('Read-only metadata command failed: ' + repr(argv))
        return result.stdout

    for row in plan['protected_metadata'] + expected['accepted_runtime_files']:
        if record_file(row['path'])['sha256'] != row['sha256']:
            raise RuntimeError('Protected input differs from accepted pin: ' + row['path'])
    source = Path(expected['protected_source']) / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
    if record_file(source)['sha256'] != expected['source_cpp_sha256']:
        raise RuntimeError('Original source hash mismatch')
    manifest = Path(expected['protected_root']) / 'llama-b10453-source-files.json'
    if record_file(manifest)['sha256'] != expected['source_manifest_sha256']:
        raise RuntimeError('Original source manifest mismatch')
    for name in plan['explicit_link_inputs'] + [plan['generated_header'], '/usr/bin/c++']:
        record_file(name)
    deps = run(['ninja', '-C', str(build), '-t', 'deps', expected['compile_target']])
    headers = [line.strip() for line in deps.splitlines() if line.startswith('    ')]
    if not headers or 'VALID' not in deps.splitlines()[0] or 'STALE' in deps.splitlines()[0]:
        raise RuntimeError('Compiler dependency log is missing or stale')
    for name in headers:
        record_file(name)
    for tool in ['cc1plus','collect2','ld','as']:
        value = run(['/usr/bin/c++', '-print-prog-name=' + tool]).strip()
        resolved = Path(value) if Path(value).is_absolute() else Path(shutil.which(value) or '')
        record_file(resolved)
    run(['/usr/bin/c++', '--version'])
    specs = run(['/usr/bin/c++', '-dumpspecs'])
    library_names = ['crtbeginS.o','crtendS.o','crti.o','crtn.o','libstdc++.so','libgcc.a','libgcc_s.so','libgcc_s.so.1','libm.so','libc.so']
    pending = list(library_names) + [str(build / 'bin/libggml-base.so.0.20.0'), '/usr/lib/libvulkan.so']
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        value = name if Path(name).is_absolute() else run(['/usr/bin/c++', '-print-file-name=' + name]).strip()
        target = Path(value)
        if not target.is_absolute() or not target.is_file():
            unresolved.append(name)
            continue
        record_file(target)
        prefix = target.read_bytes()[:4096]
        if prefix.startswith(b'\x7fELF'):
            dynamic = run(['readelf', '-d', str(target)])
            pending.extend(re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic))
        elif target.suffix != '.o' and prefix.startswith((b'/*', b'GROUP', b'INPUT')):
            text = re.sub(r'/\*.*?\*/', '', prefix.decode(errors='replace'), flags=re.S)
            pending.extend(re.findall(r'(?<![\w])(?:/[^\s()]+|lib[\w+.-]+\.(?:so(?:\.\d+)*|a))', text))
    print(json.dumps({'schema':'sepalith.opus16.target-fingerprints.v1',
        'at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'ready_for_root_review' if not unresolved else 'unresolved_link_inputs',
        'build_plan_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
        'files':list(files.values()),'compiler_specs_sha256':hashlib.sha256(specs.encode()).hexdigest(),
        'unresolved':sorted(set(unresolved)),'metadata_commands':calls,
        'scope':'Read-only fingerprints. No compile, link, shader generator, model, Vulkan device, or server execution.'},indent=2))

if __name__ == '__main__':
    main()
