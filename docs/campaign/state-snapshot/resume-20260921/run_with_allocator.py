"""Apply and record a bounded CUDA allocator policy before the unchanged trainer."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

CONFIG = "backend:native,roundup_power2_divisions:[32:256,64:128,256:64,>:32],garbage_collection_threshold:0.8"
TRAINER = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt450-cadence64-continuation-preparation-v2/source/experiments/training/full_weight_cpt_trainer.py')
EXPECTED = '5184354d974cc9bedfbc055f1d6b4c7091582e50a6d2b68c677c6a43ddfc79ee'

def main():
    if hashlib.sha256(TRAINER.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('Frozen trainer source differs')
    lock_fd = int(os.environ['SEPALITH_CUDA_LOCK_FD'])
    actual = os.fstat(lock_fd)
    expected = os.stat('/home/m0hawk/.local/state/sepalith/campaign-20260915/resource-locks/cuda0.lock')
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('Inherited CUDA lease inode differs')
    if os.environ.get('PYTORCH_ALLOC_CONF') != CONFIG:
        raise RuntimeError('Requested allocator configuration differs')
    report = Path(os.environ['SEPALITH_ALLOCATOR_REPORT'])
    if report.exists():
        raise RuntimeError('Allocator report already exists')
    # Attestation has just re-read these immutable payloads. Release clean cache only.
    cp = Path(sys.argv[sys.argv.index('--resume') + 1])
    assert hashlib.sha256((cp / 'campaign-manifest.json').read_bytes()).hexdigest() == os.environ['SEPALITH_RESUME_MANIFEST_SHA256']
    manifest = json.loads((cp / 'campaign-manifest.json').read_text())
    released = []
    for name in ('model.safetensors', 'optimizer.pt'):
        path = cp / name
        before = path.stat()
        assert before.st_size == manifest['files'][name]['bytes']
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        after = path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        released.append({'path': str(path), 'bytes': before.st_size, 'metadata_unchanged': True})
    print(json.dumps({'event': 'post_attestation_clean_cache_release', 'files': released}), flush=True)
    import unsloth
    import torch
    effective = os.environ.get('PYTORCH_ALLOC_CONF')
    backend = torch.cuda.memory.get_allocator_backend()
    if effective != CONFIG or backend != 'native':
        raise RuntimeError('Allocator policy changed during imports')
    torch.cuda.memory.set_per_process_memory_fraction(0.95, device=0)
    fraction = torch.cuda.memory.get_per_process_memory_fraction(0)
    if abs(fraction - 0.95) > 1e-9:
        raise RuntimeError('CUDA memory fraction differs')
    value = dict(schema='sepalith.sft11.native-cache-runtime.v1', requested=CONFIG,
                 effective=effective, backend=backend, memory_fraction=fraction,
                 torch=torch.__version__, cuda=torch.version.cuda,
                 total_memory=torch.cuda.get_device_properties(0).total_memory,
                 trainer_sha256=EXPECTED, pid=os.getpid())
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(value), flush=True)
    sys.path.insert(0, str(TRAINER.parent))
    sys.argv[0] = str(TRAINER)
    runpy.run_path(str(TRAINER), run_name='__main__')

if __name__ == '__main__':
    main()
