#!/usr/bin/env python3
"""Exercise the full predecessor migration, parent, worker, and terminal path."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'materialize_all_eligible.py'


def main() -> None:
    spec = importlib.util.spec_from_file_location('dat10_v2_smoke', SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, _, _, current = module.preflight(module.DEFAULT_OUTPUT)
    migration = json.loads((HERE / 'source-migration.json').read_text())
    predecessor = dict(current)
    predecessor['materializer_sha256'] = migration['from_materializer_sha256']
    with tempfile.TemporaryDirectory(prefix='dat10-v2-full-resume-') as td:
        output = Path(td)
        (output / 'groups').mkdir()
        (output / '.staging').mkdir()
        (output / 'run-manifest.json').write_text(json.dumps(predecessor, indent=2, sort_keys=True) + '\n')
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run(
            [sys.executable, str(SOURCE), '--output', str(output), '--max-groups', '1'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads((output / 'run-manifest.json').read_text()) == current
        groups = list((output / 'groups').iterdir())
        assert len(groups) == 1 and groups[0].name.startswith('000775-')
        receipt = json.loads((groups[0] / 'receipt.json').read_text())
        terminal = json.loads((output / 'terminal.json').read_text())
        assert receipt['seeded_index'] == 775
        assert terminal['invocation_groups_processed'] == 1
        assert terminal['groups_committed'] == 1
        assert terminal['stopped_by_signal'] is False
        assert any(path.name.startswith('source-migration-') for path in output.iterdir())
        print(json.dumps({'status': 'pass', 'seeded_index': 775, 'counts': receipt['counts'],
                          'predecessor_migration_applied': True, 'parent_worker_terminal_path': True}, sort_keys=True))


if __name__ == '__main__':
    main()
