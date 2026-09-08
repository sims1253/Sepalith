#!/usr/bin/env python3
"""Prepare either local generation or remote judging between runner jobs."""
import argparse
import json
from pathlib import Path
import sys
from prepare_v1c import freeze
from v1c_artifacts import digest
from b4_other_intent import ARMS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['generation', 'judge'])
    for name in ('state', 'previous', 'suite', 'assets', 'archive', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--generation', type=Path)
    a = p.parse_args()
    if any(isinstance(v, Path) and not v.is_absolute() for v in vars(a).values()):
        p.error('Use absolute paths')
    runner = Runner(a.state)
    if not runner.plan()['paused']:
        p.error('Prepare between experiments')
    previous = json.loads(a.previous.read_text())
    inputs = []
    runtime = Path(previous['env']['S1_RUNTIME'])
    files = {'intent_suite_v1.jsonl': a.suite}
    for item in previous['inputs']:
        path = Path(item['path'])
        if path.is_relative_to(runtime) or str(path).startswith(('/usr/', '/lib/')):
            if digest(path) != item['sha256']:
                raise ValueError('Frozen runtime changed')
            inputs.append(item)
        if a.phase == 'generation' and path.name in [arm + '.gguf' for arm in ARMS]:
            if digest(path) != item['sha256']:
                raise ValueError('Frozen model changed')
            files[path.name] = path
    if a.phase == 'generation' and len(files) != 4:
        raise ValueError('Require all three frozen b4 exports')
    if a.phase == 'judge':
        if a.generation is None:
            p.error('Judge requires --generation pointing to a closed generation run')
        verdict = json.loads((a.generation / 'verdict.json').read_text())
        receipt = json.loads((a.generation / 'final-runner-receipt.json').read_text())
        if verdict['verdict'] != 'INTENT-GENERATION-COMPLETE' or receipt['status'] != 'succeeded':
            raise ValueError('Generation must have closed successfully')
        generation_recipe = json.loads((a.generation / 'recipe.json').read_text())
        archive = Path(json.loads((a.generation / 'archive.json').read_text())['path'])
        manifest = json.loads((archive / 'closed-manifest.json').read_text())
        request_hash = next(i['sha256'] for i in manifest if i['source_path'] == 'requests.jsonl')
        if digest(a.generation / 'requests.jsonl') != request_hash:
            raise ValueError('Generation rows differ from closed archive')
        expected = next(i['sha256'] for i in generation_recipe['inputs'] if Path(i['path']).name == 'intent_suite_v1.jsonl')
        if digest(a.suite) != expected:
            raise ValueError('Judge suite differs from generation suite')
        files['generation.jsonl'] = a.generation / 'requests.jsonl'
        files['generation-receipt.json'] = a.generation / 'final-runner-receipt.json'
        files['generation-recipe.json'] = a.generation / 'recipe.json'
    assets, frozen = freeze(a.assets, files)
    inputs += frozen
    includes = ['packages/sepalith/src', 'experiments/eval/run_intent_suite.py',
        'experiments/eval/spec_bench.py', 'scripts/migration/s1_gpu.py',
        'scripts/migration/b4_other_intent.py', 'scripts/migration/v1c_artifacts.py']
    recipe = dict(schema_version=1, id='b4-other-intent-' + a.phase + '-20260908',
        snapshot=runner.snapshot(ROOT, includes), resource='gpu' if a.phase == 'generation' else 'cpu',
        python=previous['python'], depends_on=[], inputs=inputs, env=previous['env'],
        provenance=dict(queue_item='S2/production quant intent', phase=a.phase,
            source_includes=includes, scope='44 frozen intent cases across three b4 exports',
            runtime_bound_seconds=900 if a.phase == 'generation' else 3600,
            pi_review='Both max/60s: Muse completed 43.58s; GLM timeout 60.02s. Separate generation, strict calibration, exact judge-body cache.',
            credentials='Inherited local ZAI_API_KEY only, never stored in recipe or requests',
            adoption='NOT-ASSESSED; small paired descriptive comparison'), steps=[])
    actions = [('prepare', ['prepared.json']), ('measure', ['requests.jsonl']),
        ('evaluate_generation', ['evaluation.json', 'verdict.json'])] if a.phase == 'generation' else [
        ('judge', ['judge-calls.jsonl', 'scores.jsonl']), ('evaluate_judge', ['evaluation.json', 'verdict.json'])]
    for action, artifacts in actions:
        recipe['steps'].append(dict(id=action, argv=['{python}', '{source}/scripts/migration/b4_other_intent.py',
            action, '--run', '{run}', '--assets', str(assets)], artifacts=artifacts))
    recipe['steps'].append(dict(id='archive', argv=['{python}', '{source}/scripts/migration/v1c_artifacts.py',
        'archive', '--run', '{run}', '--archive', str(a.archive)], artifacts=['archive.json']))
    with a.output.open('x') as f:
        json.dump(recipe, f, indent=2)
    print(a.output)


if __name__ == '__main__':
    main()
