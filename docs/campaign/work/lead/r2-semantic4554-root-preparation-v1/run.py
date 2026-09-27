"""CPU-only pre-edit reconstruction using the reviewed generic shard preparer."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

LEAD = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead')
SOURCE = LEAD / 'r2-semantic763-context-admission-v1'
INPUT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-semantic-streaming-queue-shards6plus-v1')
OUTPUT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4554-root-preparation-v1')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started = time.monotonic()
    assert not OUTPUT.exists(), 'fresh output required'
    manifest = json.loads((SOURCE / 'source-manifest.json').read_text())
    for row in manifest['files']:
        assert sha(SOURCE / row['path']) == row['sha256']
    OUTPUT.mkdir(parents=True)
    results = []
    for shard in range(6, 12):
        folder = INPUT / f'shard-{shard:04d}'
        path = folder / 'manifest.json'
        m = json.loads(path.read_text())
        binding = m['streaming_binding']
        command = [sys.executable, '-B', str(SOURCE / 'prepare_inputs.py')]
        inputs = {'semantic-manifest': {'path': str(path), 'sha256': sha(path)},
                  'semantic-ledger': m['output_binding'],
                  'provenance-ledger': binding['provenance_ledger'],
                  'candidate-packets': binding['candidate_packet']}
        for name, record in inputs.items():
            command += ['--' + name, record['path'], '--expected-' + name + '-sha256', record['sha256']]
        destination = OUTPUT / f'shard-{shard:04d}'
        command += ['--output', str(destination)]
        with (OUTPUT / f'shard-{shard:04d}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, 1500 - (time.monotonic() - started)))
        assert result.returncode == 0, f'shard {shard} failed; inspect log'
        report = json.loads((destination / 'preparation-manifest.json').read_text())
        assert report['exact_supported_id_accounting'] is True
        results.append({'shard': shard, 'manifest': str(destination / 'preparation-manifest.json'), 'sha256': sha(destination / 'preparation-manifest.json'), 'supported': report['semantic_supported'], 'prepared': report['prediction_inputs'], 'holds': report['preparation_holds']})
        print(json.dumps(results[-1]), flush=True)
    assert sum(r['supported'] for r in results) == 4554
    report = {'status': 'prepared_not_training_admitted', 'source_manifest_sha256': sha(SOURCE / 'source-manifest.json'), 'shards': results, 'elapsed_seconds': time.monotonic() - started, 'cuda_used': False}
    (OUTPUT / 'terminal.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
