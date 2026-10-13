"""Run only the reviewed, standalone CPU checks in this private directory."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess

root = Path(__file__).resolve().parent
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
rows = []

def run(argv, mode=None, trace=False):
    env = os.environ.copy()
    env.pop('GGML_VK_Q8_0_M2S', None)
    env.pop('GGML_VK_Q8_0_M2S_TRACE', None)
    if mode is not None:
        env['GGML_VK_Q8_0_M2S'] = mode
    if trace:
        env['GGML_VK_Q8_0_M2S_TRACE'] = '0'  # Presence enables tracing, even "0".
    result = subprocess.run(argv, cwd=root, env=env, text=True, capture_output=True, timeout=45)
    row = dict(argv=argv, mode=mode, trace_env_zero_present=trace, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr)
    rows.append(row)
    if result.returncode:
        raise RuntimeError(row)

flags = ['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror']
run(['g++', '--version'])
for name, extra in [('standalone-test', []), ('standalone-test-ubsan', ['-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
    run(flags + extra + ['-I.', 'test_vk_q8_0_m2s.cpp', '-o', name])
    run([str(root / name)])
for suffix, extra in [('32', []), ('64', ['-DVK_EXT_shader_64bit_indexing=1'])]:
    name = 'source-wiring-' + suffix
    run(flags + ['-fsanitize=undefined', '-fno-sanitize-recover=all'] + extra + ['test_source_wiring.cpp', '-o', name])
    for mode in [None, 'small', 'pad', 'bogus']:
        run([str(root / name)], mode=mode)
    run([str(root / name)], trace=True)
(root / 'final-cpu-validation.json').write_text(json.dumps({
    'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'PASS',
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'commands': rows,
    'scope': 'Host-only tests and exact-function projections under reduced carriers. No Vulkan headers, shaders, backend ABI, model, or device tested.',
}, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'commands': len(rows), 'cpu_affinity': sorted(os.sched_getaffinity(0))}))
