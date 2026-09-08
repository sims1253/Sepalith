#!/usr/bin/env python3
"""Freeze S1 model/trace inputs and bind the previously verified CUDA runtime."""
import argparse
import json
from pathlib import Path
import sys
from prepare_v1c import freeze
from v1c_artifacts import digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('state', 'assets', 'models', 'traces', 'runtime_recipe', 'archive', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--ngram-only', action='store_true')
    a = p.parse_args()
    if any(not path.is_absolute() for path in vars(a).values() if isinstance(path, Path)):
        p.error('Use absolute paths')
    runner = Runner(a.state)
    if not runner.plan()['paused']:
        p.error('Prepare while dispatch is paused')
    previous = json.loads(a.runtime_recipe.read_text())
    runtime = Path(previous['env']['LD_LIBRARY_PATH'])
    assets, inputs = freeze(a.assets, {'model.gguf': a.models / 'b4_qwen35_2b-Q8_0.gguf',
        'mtp.gguf': a.models / 'mtp-b4_qwen35_2b-Q8_0.gguf',
        'draft.gguf': a.models / 'b2_qwen35_08b-Q8_0.gguf', 'traces.jsonl': a.traces})
    # Bind the runtime by its existing content hashes; never copy or import
    # previous experimental predictions/model/input identities.
    for item in previous['inputs']:
        path = Path(item['path'])
        if path.is_relative_to(runtime) or str(path).startswith(('/usr/', '/lib/')):
            if digest(path) != item['sha256']:
                raise ValueError('Previously frozen runtime dependency changed: ' + str(path))
            inputs.append(item)
    includes = ['packages/sepalith/src', 'experiments/eval/spec_bench.py',
                'scripts/migration/s1_gpu.py', 'scripts/migration/v1c_artifacts.py']
    snapshot = runner.snapshot(ROOT, includes)
    recipe = dict(schema_version=1, id='s1-gpu-depth-sweep-20260908', snapshot=snapshot,
        resource='gpu', python=previous['python'], depends_on=[], inputs=inputs,
        env=previous['env'], provenance=dict(queue_item='S1', scope='Full registered GPU depth sweep, 100 traces per class, three reps, 17 arm configurations plus validation-only baseline bookend',
        source_includes=includes, runtime_bound_seconds=14400, pi_review='Both requested models at max, 300 seconds each; private pi-s1-gpu receipts',
        boundary='Local GPU tier only; historical CPU results remain separate'), steps=[])
    for action, artifacts in [('prepare', ['prepared.json']), ('measure', ['per_request.jsonl','telemetry.jsonl']),
                               ('evaluate', ['evaluation.json','verdict.json'])]:
        argv = ['{python}', '{source}/scripts/migration/s1_gpu.py', action, '--run', '{run}', '--assets', str(assets)]
        recipe['steps'].append(dict(id=action, argv=argv, artifacts=artifacts))
    if a.ngram_only:
        recipe['id'] = 's1-gpu-ngram-corrected-20260908'
        recipe['provenance']['runtime_bound_seconds'] = 7200
        recipe['provenance']['scope'] = 'Corrected ngram depth sweep: sizes 16/48 (smaller registered sizes cannot draft with pinned lookup N=12), fresh baseline and bookend, 100 traces/class, three cold/warm reps'
        recipe['provenance']['pi_review'] = 'Requested models at max, 180 seconds each; private pi-ngram-correction receipts'
        recipe['steps'][0]['argv'].append('--ngram-only')
        recipe['steps'].insert(1, dict(id='check-ngram-control', argv=['{python}', '{source}/scripts/migration/s1_gpu.py', 'check-ngram-control', '--run', '{run}', '--assets', str(assets)], artifacts=['ngram-control.json']))
    # Runtime directory is separately content-addressed, so expose it explicitly.
    recipe['env'] = dict(recipe['env'], S1_RUNTIME=str(runtime))
    recipe['steps'].append(dict(id='archive', argv=['{python}', '{source}/scripts/migration/v1c_artifacts.py',
        'archive', '--run', '{run}', '--archive', str(a.archive)], artifacts=['archive.json']))
    with a.output.open('x') as f:
        json.dump(recipe, f, indent=2)
    print(a.output)


if __name__ == '__main__':
    main()
