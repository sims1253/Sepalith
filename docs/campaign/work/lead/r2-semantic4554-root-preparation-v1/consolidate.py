"""Publish one exact, review-only prediction/target stream after six shards close."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1')

def main():
    terminal = json.loads((ROOT / 'terminal.json').read_text())
    assert terminal['status'] == 'prepared_not_training_admitted'
    assert [x['shard'] for x in terminal['shards']] == list(range(6, 12))
    destination = ROOT / 'combined'
    temporary = ROOT / f'.combined-{os.getpid()}'
    assert not destination.exists() and not temporary.exists()
    temporary.mkdir()
    files = ['prediction-inputs.jsonl', 'training-sidecar.jsonl', 'preparation-holds.jsonl']
    streams = {name: (temporary / name).open('xb') for name in files}
    ids = {name: [] for name in files}
    hashes = {name: hashlib.sha256() for name in files}
    sizes = {name: 0 for name in files}
    for shard in terminal['shards']:
        manifest_path = Path(shard['manifest'])
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == shard['sha256']
        manifest = json.loads(manifest_path.read_text())
        for name in files:
            raw = (manifest_path.parent / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == manifest['outputs'][name]['sha256']
            assert len(raw) == manifest['outputs'][name]['bytes']
            records = [json.loads(line) for line in raw.splitlines()]
            assert len(records) == manifest['outputs'][name]['rows']
            ids[name].extend(row['row_id'] for row in records)
            streams[name].write(raw)
            hashes[name].update(raw)
            sizes[name] += len(raw)
    for stream in streams.values():
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
    assert ids[files[0]] == ids[files[1]]
    assert all(len(value) == len(set(value)) for value in ids.values())
    assert not set(ids[files[0]]) & set(ids[files[2]])
    assert len(ids[files[0]]) + len(ids[files[2]]) == 4554
    manifest = {'status': 'combined_prediction_inputs_pending_provider_admission', 'training_admission': False, 'supported_denominator': 4554,
                'prepared': len(ids[files[0]]), 'holds': len(ids[files[2]]),
                'parent_terminal_sha256': hashlib.sha256((ROOT / 'terminal.json').read_bytes()).hexdigest(),
                'files': {name: {'sha256': hashes[name].hexdigest(), 'bytes': sizes[name], 'rows': len(ids[name])} for name in files}}
    with (temporary / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(destination)
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
