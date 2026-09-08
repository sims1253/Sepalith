#!/usr/bin/env python3
"""Bind the existing b4 model/runtime and freeze complete episode inputs."""
import argparse
import json
from pathlib import Path
import sys
from prepare_v1c import freeze
from v1c_artifacts import digest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','assets','trajectories','runtime_recipe','archive','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if any(not path.is_absolute() for path in vars(a).values()):p.error('Use absolute paths')
    runner=Runner(a.state)
    if not runner.plan()['paused']:p.error('Freeze inputs between experiments')
    previous=json.loads(a.runtime_recipe.read_text());runtime=Path(previous['env']['S1_RUNTIME'])
    model=next(Path(i['path']) for i in previous['inputs'] if Path(i['path']).name=='model.gguf')
    assets,inputs=freeze(a.assets,{'trajectories.jsonl':a.trajectories})
    for item in previous['inputs']:
        path=Path(item['path'])
        if path==model or path.is_relative_to(runtime) or str(path).startswith(('/usr/','/lib/')):
            if digest(path)!=item['sha256']:raise ValueError('Model/runtime dependency changed')
            inputs.append(item)
    includes=['packages/sepalith/src','experiments/eval/spec_bench.py','experiments/eval/prediction_parser.py',
              'experiments/eval/episode_metrics.py','experiments/training/coding_simulator/judge_loop.py',
              'scripts/migration/s1_gpu.py','scripts/migration/v1a_b4.py','scripts/migration/v1c_artifacts.py']
    recipe=dict(schema_version=1,id='v1a-b4-complete-baseline-20260908',snapshot=runner.snapshot(ROOT,includes),
        resource='gpu',python=previous['python'],depends_on=[],inputs=inputs,
        env=dict(previous['env'],V1A_MODEL=str(model)),provenance=dict(queue_item='W16/V1a',
        scope='Complete b4 episode baseline, 60 seed3 trajectories x up to30points, 32K context, serial GPU',
        source_includes=includes,runtime_bound_seconds=7200,
        pi_review='Both requested models at max within180s; see private pi-v1a-b4 receipts',
        scientific_boundary='Simulator lexical acceptance baseline; no training, no paired claim with historical partial episodes'),steps=[])
    for action,artifacts in [('prepare',['prepared.json']),('measure',['requests.jsonl','episodes.jsonl','server.log']),('evaluate',['evaluation.json','verdict.json'])]:
        recipe['steps'].append(dict(id=action,argv=['{python}','{source}/scripts/migration/v1a_b4.py',action,'--run','{run}','--assets',str(assets)],artifacts=artifacts))
    recipe['steps'].append(dict(id='archive',argv=['{python}','{source}/scripts/migration/v1c_artifacts.py','archive','--run','{run}','--archive',str(a.archive)],artifacts=['archive.json']))
    with a.output.open('x') as f:json.dump(recipe,f,indent=2)
    print(a.output)


if __name__=='__main__':main()
