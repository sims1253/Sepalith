#!/usr/bin/env python3
"""Freeze a V1c candidate and write its recipe. Never resume or launch a queue.

Run large input capture only in the coordinated resource window. The owner must
confirm this v7 column's scope before enqueue; it does not complete all V1c.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/sepalith/src'))
from sepalith.runner import Runner
from v1c_artifacts import digest

INCLUDES = ['experiments/eval/latency_load.py', 'experiments/eval/spec_bench.py',
            'scripts/migration/v1c_artifacts.py', 'packages/sepalith/src']


def recipe(snapshot, assets, archive, interpreter, inputs):
    helper = '{source}/scripts/migration/v1c_artifacts.py'
    return {
        'schema_version': 1, 'id': 'v1c-v7-runner-20260906',
        'description': 'V1c v7 serving column; full calibration remains separate',
        'snapshot': snapshot, 'resource': 'quiet', 'python': str(interpreter),
        'depends_on': [], 'inputs': inputs,
        'env': {'PYTHONPATH': '{source}/experiments/eval', 'PYTHONNOUSERSITE': '1',
                'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'LD_LIBRARY_PATH': str(assets / 'runtime'),
                'LD_PRELOAD': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
        'provenance': {
            'queue_item': 'V1c', 'scope': 'v7 column; owner reservation required',
            'scientific_acceptance': '100 serial + 140 sweep requests, aligned trace IDs, no errors/starvation, finite TTFT',
            'adoption': 'NOT-ASSESSED; instrument calibration and quiet review remain required',
            'deviations': 'Existing instrument levels 1,2,4 and v7-only column; design calls calibration pairs and levels 1,4,8',
            'source_includes': INCLUDES,
        },
        'steps': [
            {'id': 'prepare', 'argv': ['{python}', helper, 'prepare', '--run', '{run}',
                                     '--traces', str(assets / 'traces.jsonl')],
             'artifacts': ['prepared.json']},
            {'id': 'measure', 'argv': ['{python}', '{source}/experiments/eval/latency_load.py',
               '--foreground', '--model', str(assets / 'model.gguf'),
               '--server', str(assets / 'runtime/llama-server'),
               '--traces', str(assets / 'traces.jsonl'), '--out', '{run}/measurement',
               '--port', '18431', '--threads', '8', '--n-ttft', '50', '--n-sweep', '10',
               '--classes', '2k,8k', '--levels', '1,2,4'],
             'artifacts': ['measurement/results_v1c_ttft.jsonl',
                           'measurement/results_v1c_sweep.jsonl', 'measurement/summary.json',
                           'measurement/llama-server-v1c-legA.log',
                           'measurement/llama-server-v1c-legB.log']},
            {'id': 'evaluate', 'argv': ['{python}', helper, 'evaluate', '--run', '{run}'],
             'artifacts': ['evaluation.json', 'verdict.json', 'VERDICT.md']},
            {'id': 'archive', 'argv': ['{python}', helper, 'archive', '--run', '{run}',
                                     '--archive', str(archive)], 'artifacts': ['archive.json']},
        ],
    }


def freeze(destination, files):
    """Copy exact regular bytes; dereference known library symlinks into bundle."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.capture-', dir=destination) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        manifest = []
        for relative, origin in sorted(files.items()):
            origin = origin.resolve(strict=True)
            if not origin.is_file():
                raise ValueError(f'Not a regular input: {relative}')
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(origin)
            shutil.copyfile(origin, target)
            if digest(target) != before or digest(origin) != before:
                raise ValueError(f'Input changed during capture: {relative}')
            mode = 0o555 if os.access(origin, os.X_OK) else 0o444
            target.chmod(mode)
            with target.open('rb') as f:
                os.fsync(f.fileno())
            manifest.append({'path': relative, 'sha256': before, 'mode': mode})
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        target = destination / identity
        if target.exists():
            for item in manifest:
                if digest(target / item['path']) != item['sha256']:
                    raise ValueError('Existing immutable bundle changed')
        else:
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            stage.rename(target)
        return target, [{'path': str(target / item['path']), 'sha256': item['sha256']}
                        for item in manifest]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'assets', 'archive', 'model', 'traces', 'server_dir', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--python', type=Path, default=Path('/usr/bin/python3'))
    a = p.parse_args()
    for path in (a.state, a.assets, a.archive, a.output, a.python):
        if not path.is_absolute():
            p.error('Use absolute paths')
    for path in (a.state, a.assets, a.archive):
        if path.resolve().is_relative_to(ROOT):
            p.error('State, assets and archive must be outside the worktree')
    runner = Runner(a.state)
    if not runner.plan()['paused']:
        p.error('Preparation requires a paused state directory')
    files = {'model.gguf': a.model, 'traces.jsonl': a.traces}
    for path in a.server_dir.iterdir():
        if path.name == 'llama-server' or '.so' in path.name:
            files['runtime/' + path.name] = path
    if 'runtime/llama-server' not in files:
        p.error('Server directory has no llama-server')
    assets, inputs = freeze(a.assets, files)
    # Record actual loader dependencies after relocating the runtime bundle.
    # Never permit a baked-in loader path to fall back to the development tree.
    system_inputs = {a.python.resolve()}
    env = dict(os.environ, LD_LIBRARY_PATH=str(assets / 'runtime'), LD_PRELOAD='')
    runtime_files = [assets / 'runtime/llama-server', *sorted((assets / 'runtime').glob('*.so*'))]
    for binary in [a.python.resolve(), *runtime_files]:
        output = subprocess.check_output(['ldd', str(binary)], text=True, env=env)
        if 'not found' in output:
            raise ValueError(f'Missing runtime library for {binary}')
        for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)', output, re.MULTILINE):
            path = Path(name).resolve()
            if not path.is_relative_to(assets):
                if not str(path).startswith(('/usr/lib/', '/lib/')):
                    raise ValueError(f'Uncaptured runtime dependency: {path}')
                system_inputs.add(path)
    for path in sorted(system_inputs):
        inputs.append({'path': str(path), 'sha256': digest(path)})
    snapshot = runner.snapshot(ROOT, INCLUDES)
    result = recipe(snapshot, assets, a.archive, a.python.resolve(), inputs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(a.output)


if __name__ == '__main__':
    main()
