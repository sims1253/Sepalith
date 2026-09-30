#!/usr/bin/env python3
"""Explicit prepare/submit/status/collect for one bounded Kaggle smoke.

State belongs outside Git. A persisted submitting state is uncertain after a
crash, even if the local CLI disappeared. Only submit from prepared. A retry
uses a new directory and ID after investigation; this tool never retries push.
"""
import argparse
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(root, record):
    path = root / 'state.json.tmp'
    with path.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.replace(root / 'state.json')
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(root):
    with (root / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(root, owner):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', owner):
        raise ValueError('Invalid Kaggle owner')
    root.mkdir(parents=True, exist_ok=False)
    stage = root / 'source'
    stage.mkdir()
    job = 'sepalith-smoke-' + uuid.uuid4().hex[:16]
    source = Path(__file__).with_name('kaggle_smoke.py').read_bytes()
    (stage / 'kernel.py').write_bytes(('JOB_ID = ' + repr(job) + '\n').encode() + source)
    metadata = dict(id=f'{owner}/{job}', title=job, code_file='kernel.py',
                    language='python', kernel_type='script', is_private=True,
                    enable_gpu=True, enable_tpu=False, enable_internet=False,
                    dataset_sources=[], competition_sources=[], kernel_sources=[])
    (stage / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    record = dict(job_id=job, remote_id=metadata['id'] + '/1', status='prepared',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  controller_sha256=digest(Path(__file__)),
                  controller_python=sys.version, controller_interpreter=sys.executable,
                  timeout_seconds=600, accelerator='NvidiaTeslaT4',
                  source_sha256=digest(stage / 'kernel.py'),
                  metadata_sha256=digest(stage / 'kernel-metadata.json'))
    save(root, record)
    return record


def cli(binary, *args):
    # No shell, inherited credentials stay in the CLI environment.
    result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'Kaggle command failed ({result.returncode}); inspect account job status')
    return result.stdout


def submit(root, record, binary, invoke=cli):
    if record['status'] != 'prepared':
        raise ValueError('Submission refused: investigate the existing remote ID; never repeat push')
    stage = root / 'source'
    if digest(stage / 'kernel.py') != record['source_sha256'] or digest(stage / 'kernel-metadata.json') != record['metadata_sha256']:
        raise ValueError('Prepared source or metadata changed')
    quota = json.loads(invoke(binary, 'quota', '--format', 'json'))
    gpu = next(row for row in quota if row['resource'] == 'GPU')
    if float(gpu['remaining'].removesuffix('h')) < 0.25:
        raise ValueError('Need at least 0.25 GPU hours remaining for this smoke')
    record.update(status='submitting', quota_before=quota,
                  submitted_at=datetime.now(timezone.utc).isoformat())
    save(root, record)  # durable before the side effect, including process interruption
    try:
        invoke(binary, 'kernels', 'push', '-p', str(stage), '--timeout', '600', '--accelerator', record['accelerator'])
    except Exception:
        record['status'] = 'unknown'
        save(root, record)
        raise
    record['status'] = 'submitted'
    save(root, record)
    return record


def status(root, record, binary, invoke=cli):
    if record['status'] == 'prepared':
        raise ValueError('Job has not been submitted')
    output = invoke(binary, 'kernels', 'status', record['remote_id'])
    # CLI prints KernelWorkerStatus.COMPLETE (2.x) or quoted lowercase (1.x).
    found = re.search(r'(?:KernelWorkerStatus\.|status\s+["\x27]?)(complete|running|queued|error|cancelled|canceled)\b', output, re.I)
    if not found:
        raise ValueError('Unrecognized remote status; preserve uncertainty')
    remote = found.group(1).lower()
    record['remote_status'] = remote
    record['checked_at'] = datetime.now(timezone.utc).isoformat()
    if record['status'] != 'verified':
        record['status'] = 'remote_' + remote
    save(root, record)
    return record


def verify(root, record):
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['job_id'] != record['job_id'] or receipt['source_sha256'] != record['source_sha256']:
        raise ValueError('Wrong job or source in remote receipt')
    if receipt['operational_status'] != 'succeeded' or receipt['scientific_verdict'] != 'not_applicable':
        raise ValueError('Unexpected receipt verdict')
    if set(receipt['outputs']) != {'checkpoint.pt', 'metrics.json'}:
        raise ValueError('Unexpected output set')
    for name, expected in receipt['outputs'].items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Artifact hash or size mismatch: ' + name)
    metrics = json.loads((root / 'metrics.json').read_text())
    import math
    losses = metrics['losses']
    if len(losses) != 64 or not all(math.isfinite(x) and x >= 0 for x in losses) or not 0 <= metrics['restored_mse'] < 0.01 or losses[-1] >= losses[0] / 100:
        raise ValueError('Numerical smoke acceptance failed')
    return receipt


def collect(root, record, binary, invoke=cli):
    status(root, record, binary, invoke)
    if record.get('remote_status') != 'complete':
        raise ValueError('Remote job must be complete before collection')
    if (root / 'artifacts').exists():
        verify(root / 'artifacts', record)
    else:
        with tempfile.TemporaryDirectory(prefix='download-', dir=root) as temporary:
            stage = Path(temporary)
            invoke(binary, 'kernels', 'output', record['remote_id'], '-p', str(stage), '--force')
            verify(stage, record)
            stage.rename(root / 'artifacts')
    record['status'] = 'verified'
    record['verified_at'] = datetime.now(timezone.utc).isoformat()
    record['receipt_sha256'] = digest(root / 'artifacts/receipt.json')
    save(root, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'status', 'collect'])
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--owner', default='m0hawk')
    parser.add_argument('--kaggle', default='kaggle')
    args = parser.parse_args()
    root = args.state.expanduser().resolve()
    if any((parent / '.git').exists() for parent in (root, *root.parents)):
        raise ValueError('Keep job state outside Git worktrees')
    if args.action == 'prepare':
        record = prepare(root, args.owner)
    else:
        with locked(root):
            record = json.loads((root / 'state.json').read_text())
            record = globals()[args.action](root, record, args.kaggle)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
