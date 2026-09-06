"""Write a runner recipe; this command never enqueues or resumes a queue."""
import argparse
from pathlib import Path

from .provenance import file_hash, verify_freeze, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.frozen).resolve()
    frozen = verify_freeze(root)
    inputs = [root / name for name in ['freeze.json', 'recipe.json', 'dataset.json', 'inspection.json', 'data-audit.json']]
    inputs += [Path(frozen['checkpoint_path']) / name for name in frozen['checkpoint_files']]
    inputs += [Path(args.window).resolve()]
    write_json(args.output, {
        'schema_version': 1, 'id': 'latent-defaults-b4-screen-1273-v1', 'snapshot': args.snapshot,
        'resource': 'gpu', 'depends_on': [], 'python': str(Path(args.python).absolute()),
        'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'},
        'inputs': [{'path': str(p), 'sha256': file_hash(p)} for p in inputs],
        'provenance': {'kind': 'learnability gate, not edit training', 'freeze': frozen},
        'steps': [{'id': 'bounded-screen', 'argv': ['{python}', '{source}/experiments/latent_memory/entry.py',
                   'run', '--frozen', str(root), '--window', str(Path(args.window).resolve()),
                   '--output', '{run}/screen'],
                   'artifacts': ['screen/result.json', 'screen/predictions.jsonl', 'screen/ledger.jsonl',
                                 'screen/real-decoder-contract.json', 'screen/supervisor.json',
                                 'screen/local/final/checkpoint.json', 'screen/retrieval/final/checkpoint.json',
                                 'screen/latent/final/checkpoint.json']}]
    })


if __name__ == '__main__':
    main()
