#!/usr/bin/env python3
"""Freeze the bounded W33 calibration/repair candidate without dispatching it."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from prepare_v1c import freeze
from v1c_artifacts import digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/sepalith/src'))
from sepalith.runner import Runner

INCLUDES = ['packages/sepalith/src', 'experiments/eval/eval_ablation.py',
            'experiments/eval/spec_bench.py', 'scripts/migration/w33_repair.py',
            'scripts/migration/v1c_artifacts.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'assets', 'archive', 'model', 'examples', 'saved', 'server_dir', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--recipe-id', help='Explicit identity for a reviewed recipe revision; prior attempts remain immutable')
    a = p.parse_args()
    if any(not path.is_absolute() for path in vars(a).values() if isinstance(path, Path)):
        p.error('Use absolute paths')
    for path in (a.state, a.assets, a.archive):
        if path.resolve().is_relative_to(ROOT):
            p.error('Keep state and raw artifacts outside Git')
    runner = Runner(a.state)
    if not runner.plan()['paused']:
        p.error('Preparation requires paused dispatch')
    files = {'model.gguf': a.model, 'examples.jsonl': a.examples}
    if not a.fresh:
        files['saved.jsonl'] = a.saved
    for path in a.server_dir.iterdir():
        if path.name == 'llama-server' or '.so' in path.name:
            files['runtime/' + path.name] = path
    if a.gpu:
        if not a.fresh:
            p.error('GPU placement is available only for the separately authorized fresh evaluation')
        # CUDA toolkit libraries are external inputs, not mutable checkout
        # dependencies. Relocate their exact bytes with the server libraries.
        env = dict(os.environ, LD_LIBRARY_PATH=str(a.server_dir), LD_PRELOAD='')
        for binary in [p for key, p in list(files.items()) if key.startswith('runtime/')]:
            loaded = subprocess.check_output(['ldd', str(binary)], text=True, env=env)
            for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)', loaded, re.MULTILINE):
                if name.startswith('/usr/local/cuda'):
                    files['runtime/' + Path(name).name] = Path(name)
    assets, inputs = freeze(a.assets, files)
    python = Path(sys.executable).resolve()
    system = {python}
    env = dict(os.environ, LD_LIBRARY_PATH=str(assets / 'runtime'), LD_PRELOAD='')
    for binary in [python, assets / 'runtime/llama-server', *sorted((assets / 'runtime').glob('*.so*'))]:
        loaded = subprocess.check_output(['ldd', str(binary)], text=True, env=env)
        if 'not found' in loaded:
            raise ValueError('Unresolved frozen runtime dependency')
        for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)', loaded, re.MULTILINE):
            path = Path(name).resolve()
            if not path.is_relative_to(assets):
                if not str(path).startswith(('/usr/lib/', '/lib/')):
                    raise ValueError('Runtime resolves outside frozen assets and system libraries')
                system.add(path)
    inputs += [{'path': str(path), 'sha256': digest(path)} for path in sorted(system)]
    snapshot = runner.snapshot(ROOT, INCLUDES)
    helper = '{source}/scripts/migration/w33_repair.py'
    recipe = dict(schema_version=1, id='w33-calibration-repair-20260908', snapshot=snapshot,
        resource='cpu', python=str(python), depends_on=[], inputs=inputs,
        env={'PYTHONPATH': '{source}/experiments/eval', 'PYTHONNOUSERSITE': '1',
             'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'LD_LIBRARY_PATH': str(assets / 'runtime'), 'LD_PRELOAD': '',
             'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
        provenance={'queue_item': 'W33', 'scope': '16 calibration rows, then absent rows only if exact calibration passes',
                    'scientific_gate': 'All 16 normalized predictions and scores equal saved rows',
                    'adoption': 'NOT-ASSESSED; original saved provenance remains unknown',
                    'runtime_bound_seconds': 5400, 'source_includes': INCLUDES,
                    'pi_review': {'opencode/muse-spark-1.3-contributor-free': 'max; completed',
                                  'zai/glm-5.3': 'max; time limit without report',
                                  'limit_seconds_each': 180}},
        steps=[dict(id=action, argv=['{python}', helper, action, '--run', '{run}', '--assets', str(assets)],
                    artifacts=artifacts) for action, artifacts in [
                        ('prepare', ['prepared.json']), ('measure', ['predictions.jsonl', 'server.log']),
                        ('evaluate', ['evaluation.json', 'verdict.json'])]])
    recipe['steps'].append(dict(id='archive', argv=['{python}', '{source}/scripts/migration/v1c_artifacts.py',
        'archive', '--run', '{run}', '--archive', str(a.archive)], artifacts=['archive.json']))
    if a.fresh:
        recipe['id'] = 'w33-fresh-plain921-20260908'
        recipe['steps'][0]['argv'].append('--fresh')
        recipe['provenance'].update(scope='User-authorized new evaluation of all 921 current frozen plain-split rows',
            scientific_gate='Exact row coverage, no context truncation, complete scores; not historical replication',
            adoption='NOT-ASSESSED; a new quality baseline, no paired model comparison',
            pi_review={'required_models': ['opencode/muse-spark-1.3-contributor-free', 'zai/glm-5.3'],
                       'thinking': 'max', 'limit_seconds_each': 180,
                       'results': 'See separate pre-launch review receipts for this fresh evaluation.'})
    if a.gpu:
        recipe['id'] = 'w33-fresh-plain921-gpu-20260908'
        recipe['resource'] = 'gpu'
        recipe['steps'][0]['argv'].append('--gpu')
        recipe['provenance'].update(placement='RTX 5090 full offload; separate from incomplete CPU attempt',
                                    runtime_gate='Server log must confirm all model layers offloaded')
    if a.recipe_id:
        recipe['id'] = a.recipe_id
    with a.output.open('x') as f:
        json.dump(recipe, f, indent=2)
        f.write('\n')
    print(a.output)


if __name__ == '__main__':
    main()
