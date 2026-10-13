"""Check completed theta0 quant artifacts without loading tensors for inference."""
from pathlib import Path
from collections import Counter
import datetime as dt
import gc
import hashlib
import json
import os
from gguf import GGUFReader

ROOT = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
OUT = ROOT / 'models/SFT-primary-step1000-quant-candidates-c'
terminal = json.loads((OUT / 'terminal.json').read_text())
guard = json.loads((ROOT / 'training/RUN-01-theta0-quant-c-host-supervision/terminal.json').read_text())
assert guard['status'] == 'completed' and guard['child_exit_code'] == 0
assert terminal['status'] == 'exported_pending_root_metadata_and_quality_review'
f16 = GGUFReader(ROOT / 'models/SFT-primary-step1000-runtime-gguf/model-F16.gguf')
expected = {t.name: [int(v) for v in t.shape] for t in f16.tensors}
metadata = {k: v.contents() for k, v in f16.fields.items() if k.startswith('tokenizer.')}
del f16
gc.collect()
artifacts = []
for i, kind in enumerate(('Q8_0', 'Q6_K')):
    artifact = json.loads((OUT / f'quant-{i}-artifact.json').read_text())
    path = Path(artifact['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        offset = 0
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
        after = os.fstat(stream.fileno())
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert digest.hexdigest() == artifact['sha256'] and before.st_size == artifact['bytes']
    reader = GGUFReader(path)
    actual = {t.name: [int(v) for v in t.shape] for t in reader.tensors}
    assert actual == expected and len(reader.tensors) == 381
    assert {k: v.contents() for k, v in reader.fields.items() if k.startswith('tokenizer.')} == metadata
    types = Counter(t.tensor_type.name for t in reader.tensors)
    expected_types = {'Q8_0': 296, 'F32': 85} if kind == 'Q8_0' else {'Q6_K': 294, 'Q8_0': 2, 'F32': 85}
    assert dict(types) == expected_types, types
    for t in reader.tensors:
        if t.name in ('output.weight', 'token_embd.weight'):
            assert t.tensor_type.name == 'Q8_0'
    extents = sorted((int(t.data_offset), int(t.n_bytes)) for t in reader.tensors)
    assert all(a + b == c for (a, b), (c, _) in zip(extents, extents[1:]))
    assert sum(extents[-1]) == before.st_size
    artifacts.append({**artifact, 'tensor_count': len(reader.tensors), 'tensor_types': dict(types),
                      'names_shapes_equal_F16': True, 'tokenizer_metadata_equal_F16': True,
                      'first_tensor_offset': extents[0][0], 'last_tensor_end': sum(extents[-1]),
                      'extent_closure': True, 'protected_embedding_and_output_Q8': True})
    del reader
    gc.collect()
receipt = {'task': 'RUN-01/RUN-09', 'owner': 'lead', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
           'status': 'quant_artifact_integrity_accepted_native_quality_pending', 'guard': guard,
           'artifacts': artifacts, 'source_F16_sha256': 'ee7ba2fd7e7b6446d74224e1e1f9e4724fbcfa54e56e60c4b623981d1e4f45ed',
           'remaining': ['Native EOG and integer tokenizer parity on theta0', 'DEV quantization quality',
                         'Final selected weights latency/editor/draft acceptance'],
           'limits': 'No tensor-value or native inference parity claim. Selected theta0 remains unchanged.'}
p = PLAN / 'docs/campaign/receipts/RUN-01-theta0-quant-c-lead-review.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'artifacts': artifacts}))
