#!/usr/bin/env python3
"""Prepare and collect finite GPU/TPU capability probes using durable job state."""
import argparse
import json
from pathlib import Path
import re
import tempfile
import uuid
import kaggle_job as jobs

PROFILES = {
    'gpu': ('NvidiaTeslaT4', 'GPU', 900, 720),
    'tpu': ('Tpu1VmV38', 'TPU', 600, 480),
    'tpu-runtime': ('Tpu1VmV38', 'TPU', 300, 240),
    'tpu-v6': ('TpuV6E8', 'TPU', 180, 120),
}
REQUIRED = {
    'gpu': {'cuda', 'all_devices_compute', 'adapter_training', 'adapter_reload', 'generation'},
    'tpu': {'tpu_backend', 'bf16_matmul', 'training', 'checkpoint_reload'},
    'tpu-runtime': {'tpu_backend', 'bf16_matmul', 'training', 'checkpoint_reload'},
    'tpu-v6': {'tpu_backend', 'bf16_matmul', 'training', 'checkpoint_reload'},
}


def prepare(root, owner, probe):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', owner): raise ValueError('Invalid owner')
    accelerator, resource, timeout, code_timeout = PROFILES[probe]
    # Reuse the existing atomic preparation and provenance record, replacing
    # the payload before any submit is allowed. No remote side effect here.
    record = jobs.prepare(root, owner, ready=False)
    source_root = Path(__file__).parent / 'probes'
    stage = root / 'source'
    job_id = f'sepalith-{probe}-' + uuid.uuid4().hex[:16]
    source = f'JOB_ID = {job_id!r}\nPROBE_NAME = {probe!r}\nCODE_TIMEOUT = {code_timeout}\n'
    source += f'INSTALL_LIBTPU = {probe in ("tpu-runtime", "tpu-v6")!r}\n'
    source += (source_root / (probe.split('-')[0] + '.py')).read_text() + '\n'
    source += (source_root / 'runtime.py').read_text()
    (stage / 'kernel.py').write_text(source)
    metadata = json.loads((stage / 'kernel-metadata.json').read_text())
    metadata.update(id=f'{owner}/{job_id}', title=job_id,
                    enable_gpu=resource == 'GPU', enable_tpu=resource == 'TPU',
                    enable_internet=resource == 'GPU' or probe in ('tpu-runtime', 'tpu-v6'), machine_shape=accelerator)
    (stage / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record.update(status='prepared', job_id=job_id, remote_id=f'{owner}/{job_id}/1', probe=probe,
                  accelerator=accelerator, quota_resource=resource, timeout_seconds=timeout,
                  source_sha256=jobs.digest(stage / 'kernel.py'),
                  metadata_sha256=jobs.digest(stage / 'kernel-metadata.json'),
                  probe_controller_sha256=jobs.digest(Path(__file__)))
    jobs.save(root, record)
    return record


def verify(folder, record):
    receipt = json.loads((folder / 'receipt.json').read_text())
    if receipt['job_id'] != record['job_id'] or receipt['source_sha256'] != record['source_sha256']:
        raise ValueError('Receipt identity mismatch')
    if receipt['operational_status'] != 'succeeded' or receipt['scientific_verdict'] != 'not_applicable':
        raise ValueError('Unexpected receipt status')
    allowed = {'report.json', 'adapter.safetensors'} if record['probe'] == 'gpu' else {'report.json', 'checkpoint.npy'}
    if 'report.json' not in receipt['outputs'] or not set(receipt['outputs']) <= allowed:
        raise ValueError('Unexpected artifact set')
    for name, expected in receipt['outputs'].items():
        path = folder / name
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or jobs.digest(path) != expected['sha256']:
            raise ValueError('Artifact mismatch')
    report = json.loads((folder / 'report.json').read_text())
    if report['probe'] != record['probe'] or report['outcome'] not in ('supported', 'blocked'):
        raise ValueError('Invalid capability report')
    if report['outcome'] == 'supported':
        if set(receipt['outputs']) != allowed or any(report['checks'].get(key) is not True for key in REQUIRED[record['probe']]):
            raise ValueError('Supported capability lacks required evidence')
    return report


def collect(root, record, binary, invoke=jobs.cli):
    jobs.status(root, record, binary, invoke)
    if record['remote_status'] != 'complete': raise ValueError('Job not complete; inspect terminal errors separately')
    if (root / 'artifacts').exists(): report = verify(root / 'artifacts', record)
    else:
        with tempfile.TemporaryDirectory(prefix='download-', dir=root) as temporary:
            stage = Path(temporary)
            invoke(binary, 'kernels', 'output', record['remote_id'], '-p', str(stage), '--force')
            report = verify(stage, record)
            stage.rename(root / 'artifacts')
    record.update(status='verified', capability=report['outcome'], receipt_sha256=jobs.digest(root / 'artifacts/receipt.json'))
    jobs.save(root, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'status', 'collect'])
    parser.add_argument('--probe', choices=PROFILES)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--owner', default='m0hawk')
    parser.add_argument('--kaggle', default='kaggle')
    args = parser.parse_args()
    root = args.state.expanduser().resolve()
    jobs.ensure_external_state(root)
    if args.action == 'prepare':
        if not args.probe: parser.error('prepare needs --probe')
        record = prepare(root, args.owner, args.probe)
    else:
        with jobs.locked(root):
            record = json.loads((root / 'state.json').read_text())
            if args.action == 'collect': record = collect(root, record, args.kaggle)
            elif args.action == 'submit': record = jobs.submit(root, record, args.kaggle)
            else: record = jobs.status(root, record, args.kaggle)
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
