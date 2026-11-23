"""Turn one notebook_latency.py result and its notebook logs into a C17 receipt.

    python scripts/packaging/latency_receipt.py --result results/Q6_K-cpu.json \
        --logs logs/ --output receipts/C17-latency-Q6_K-cpu.json

`--logs` holds the files the notebook driver writes for tag `<tier>-<backend>`:
`.server.log` (measured run, default verbosity), `.probe-server.log` (same
command plus `-lv 4`, unmeasured, so device and offload lines are printed),
`.probe.txt` (iGPU sysfs memory before/loaded/after), optional `.gpu.tsv`
(iGPU sysfs samples during the measured run), `.env-before.txt`,
`.env-after.txt`, and `env.json` for the host.

The receipt keeps every raw sample and adds card C17 section 3's checks: 40
samples, finite positive timings, untruncated prompts, stable prompt token
counts, pinned identities, and logs that confirm the intended device. It
applies no latency cutoff.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

EXPECTED_SAMPLES = 40
RUNTIME_COMMIT = '3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70'
PROMPTS_SHA256 = 'c61cabc8cdb8a5d4eaf83cd90dc1633a1e20c858d3fbe85cbfd511ab2d2e36d2'
MODEL_SHA256 = {
    'Q6_K': '6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261',
    'Q4_K_M-mixB-e4': 'e24fbfbc5ea47c88647445ec682196fc67af9134d4fa36e826beeda045f1c71b',
}
EXCERPT = re.compile(r'warning: no usable GPU|device_info|  - Vulkan\d|system_info|using device|offload'
                     r'|model buffer size|KV buffer size|compute buffer size|no changes needed'
                     r'|n_ctx_slot|n_threads =')
MEM = re.compile(r'^(before|loaded|after) gpu_busy=(\d+) vram_used=(\d+) gtt_used=(\d+)$', re.MULTILINE)


def probe_memory(text: str) -> dict[str, dict[str, int]]:
    return {m[0]: {'gpu_busy': int(m[1]), 'vram_used': int(m[2]), 'gtt_used': int(m[3])}
            for m in MEM.findall(text)}


def gpu_samples(text: str | None) -> dict | None:
    if text is None:
        return None
    rows = [line.split('\t') for line in text.strip().splitlines()[1:]]
    busy = [int(r[1]) for r in rows]
    return {'samples': len(rows), 'busy_nonzero': sum(b > 0 for b in busy),
            'busy_max': max(busy, default=0), 'gtt_used_max': max((int(r[3]) for r in rows), default=0)}


def backend_evidence(backend: str, measured: str, probe: str, memory: dict, gpu: dict | None = None) -> dict:
    """Check that the device lines and iGPU memory match the intended backend."""
    offload = [tuple(map(int, m)) for m in re.findall(r'offloaded (\d+)/(\d+) layers to GPU', probe)]
    devices = re.findall(r'^\S+ I cmn  common_param:   - (Vulkan\d+ : .+)$', probe, re.MULTILINE)
    vulkan_buffer = re.findall(r'Vulkan\d+ model buffer size = +([\d.]+) MiB', probe)
    cpu_buffer = re.findall(r'CPU(?:_Mapped|_REPACK)? model buffer size = +([\d.]+) MiB', probe)
    gtt_rise = memory['loaded']['gtt_used'] - memory['before']['gtt_used']
    no_gpu = 'warning: no usable GPU found' in measured
    if backend == 'vulkan':
        # The probe shows the device; the measured run must not have fallen back and must load the iGPU.
        ok = (not no_gpu and bool(devices) and bool(vulkan_buffer) and bool(offload)
              and all(done == total for done, total in offload) and gtt_rise > 0
              and gpu is not None and gpu['busy_nonzero'] > 0)
    else:
        ok = (no_gpu and not devices and not vulkan_buffer and bool(cpu_buffer) and not offload
              and gtt_rise == 0 and (gpu is None or gpu['busy_nonzero'] == 0))
    return {'confirmed': ok, 'vulkan_devices': devices, 'offloaded_layers': offload,
            'vulkan_model_buffer_mib': vulkan_buffer, 'cpu_model_buffer_mib': cpu_buffer,
            'gtt_rise_bytes_at_load': gtt_rise}


def gate(result: dict, evidence: dict, measured: str) -> dict:
    samples = result['samples']
    tokens: dict[int, set[int]] = {}
    for s in samples:
        tokens.setdefault(s['prompt'], set()).add(s['prompt_n'])
    completions = re.findall(r'stop processing: n_tokens = \d+, truncated = (\d)', measured)
    # Per-request decode lines in server order: one warm-up then five measured per prompt.
    decode = re.findall(r'\|\s+eval time = +[\d.]+ ms / +(\d+) tokens \( *[\d.]+ ms per token, +([\d.]+) tokens per second\)',
                        measured)
    logged = [(int(n), float(r)) for i, (n, r) in enumerate(decode) if i % 6]
    ordered = sorted(samples, key=lambda s: (s['prompt'], s['repeat']))
    checks = {
        'samples_40': len(samples) == EXPECTED_SAMPLES,
        'five_per_prompt': sorted(Counter(s['prompt'] for s in samples).values()) == [5] * 8,
        'finite_positive_timings': all(math.isfinite(s[k]) and s[k] > 0 for s in samples
                                       for k in ('ttft_ms', 'tokens_s', 'wall_ms')),
        'predicted_n_at_least_2': all(s['predicted_n'] >= 2 for s in samples),
        'prompt_n_stable': all(len(v) == 1 for v in tokens.values()),
        # notebook_latency.py raises on a truncated final event; the server log agrees.
        'untruncated_prompts': 'truncated = 1' not in measured,
        # 8 warm-ups + 40 measured requests completed in this server log, none truncated.
        'server_log_48_completions': completions == ['0'] * (EXPECTED_SAMPLES + 8),
        # The log belongs to this result: measured decode counts and rates match in order.
        'server_log_matches_samples': len(decode) == EXPECTED_SAMPLES + 8 and len(logged) == len(ordered) and all(
            n == s['predicted_n'] and abs(r - s['tokens_s']) <= 0.006 for (n, r), s in zip(logged, ordered)),
        'runtime_commit': result['runtime_commit'] == RUNTIME_COMMIT,
        'model_sha256': result['model_sha256'] == MODEL_SHA256[result['tier']],
        'prompts_sha256': result['prompts_sha256'] == PROMPTS_SHA256,
        'backend_confirmed': evidence['confirmed'],
    }
    return {'checks': checks, 'outcome': 'pass' if all(checks.values()) else 'fail', 'latency_cutoff': None}


def build(result_path: Path, logs: Path) -> dict:
    result = json.loads(result_path.read_text())
    tag = f"{result['tier']}-{result['backend']}"
    files = {name: logs / f'{tag}.{name}' for name in
             ('server.log', 'probe-server.log', 'probe.txt', 'gpu.tsv', 'env-before.txt', 'env-after.txt')}
    measured = files['server.log'].read_text(errors='replace')
    probe = files['probe-server.log'].read_text(errors='replace')
    memory = probe_memory(files['probe.txt'].read_text())
    gpu = files['gpu.tsv']
    gpu_summary = gpu_samples(gpu.read_text() if gpu.exists() else None)
    evidence = backend_evidence(result['backend'], measured, probe, memory, gpu_summary)
    samples = result['samples']
    return {
        'schema': 'sepalith.c17.notebook-latency-receipt.v1',
        'card': 'C17 section 3',
        'tier': result['tier'], 'backend': result['backend'],
        'gate': gate(result, evidence, measured),
        'summary': result['summary'],
        'percentile_method': 'nearest-rank over all 40 measured requests',
        'prompt_tokens': {str(k): v for k, v in sorted({s['prompt']: s['prompt_n'] for s in samples}.items())},
        'stop_types': dict(Counter(str(s['stop_type']) for s in samples)),
        'predicted_n': sorted({s['predicted_n'] for s in samples}),
        'request': {'n_predict': 1024, 'temperature': 0, 'top_k': 1, 'seed': 3407, 'cache_prompt': False,
                    'stream': True, 'warmups_per_prompt': 1, 'measured_per_prompt': 5, 'manual_bos': 0},
        'identity': {k: result[k] for k in ('runtime_commit', 'model_sha256', 'prompts_sha256',
                                             'server_command', 'host', 'processor')},
        'environment': json.loads((logs / 'env.json').read_text()),
        'load_before': files['env-before.txt'].read_text().strip().splitlines(),
        'load_after': files['env-after.txt'].read_text().strip().splitlines(),
        'backend_evidence': {
            **evidence,
            'probe': 'unmeasured restart of the identical server command plus `-lv 4`; '
                     'verbosity 3 at 3cb7ffb1a prints no device or offload lines',
            'probe_igpu_memory': memory,
            'measured_run_igpu': gpu_summary,
        },
        'logs': {name: {'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for name, p in files.items() if p.exists()},
        'log_excerpt': {
            'measured': [ln for ln in measured.splitlines() if EXCERPT.search(ln)][:20],
            'probe': [ln for ln in probe.splitlines() if EXCERPT.search(ln)][:40],
        },
        'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'samples': samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = build(a.result, a.logs)
    a.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(a.output.name, receipt['gate']['outcome'])


if __name__ == '__main__':
    main()
