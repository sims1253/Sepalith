#!/usr/bin/env python3
"""Prepare the unrun S2 GPU timing column while the runner is paused."""
import argparse
import json
from pathlib import Path
import sys
from prepare_v1c import freeze
from v1c_artifacts import digest
from s2_gpu import FORMATS

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('state','assets','models','runtime_recipe','archive','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=parser.parse_args()
    if any(not p.is_absolute() for p in vars(a).values()):parser.error('Use absolute paths')
    runner=Runner(a.state)
    if not runner.plan()['paused']:parser.error('Prepare between experiments')
    previous=json.loads(a.runtime_recipe.read_text())
    traces=next(Path(i['path']) for i in previous['inputs'] if Path(i['path']).name=='traces.jsonl')
    files={fmt+'.gguf':a.models/('b1_ref24-'+fmt+'.gguf') for fmt in FORMATS}
    files['traces.jsonl']=traces
    assets,inputs=freeze(a.assets,files)
    runtime=Path(previous['env']['S1_RUNTIME'])
    for item in previous['inputs']:
        path=Path(item['path'])
        if path.is_relative_to(runtime) or str(path).startswith(('/usr/','/lib/')):
            if digest(path)!=item['sha256']:raise ValueError('Runtime dependency changed')
            inputs.append(item)
    includes=['packages/sepalith/src','experiments/eval/spec_bench.py','scripts/migration/s1_gpu.py',
              'scripts/migration/s2_gpu.py','scripts/migration/v1c_artifacts.py']
    recipe=dict(schema_version=1,id='s2-gpu-timing-20260908',snapshot=runner.snapshot(ROOT,includes),
        resource='gpu',python=previous['python'],depends_on=[],inputs=inputs,env=previous['env'],
        provenance=dict(queue_item='S2',scope='Unrun b1_ref24 GPU timing column, 5 formats x10traces x3reps',
            source_includes=includes,runtime_bound_seconds=1800,
            optimization_preroll='Skipped separate pre-roll: expected only a few minutes; reuse reviewed S1 GPU lifetime, offload and token-accounting safeguards. User permits value-based pre-rolls.',
            adoption='NOT-ASSESSED; quality within 1pp of Q8 required separately'),steps=[])
    for action,artifacts in [('prepare',['prepared.json']),('measure',['per_request.jsonl','telemetry.jsonl','warmups.jsonl']),('evaluate',['evaluation.json','verdict.json'])]:
        recipe['steps'].append(dict(id=action,argv=['{python}','{source}/scripts/migration/s2_gpu.py',action,'--run','{run}','--assets',str(assets)],artifacts=artifacts))
    recipe['steps'].append(dict(id='archive',argv=['{python}','{source}/scripts/migration/v1c_artifacts.py','archive','--run','{run}','--archive',str(a.archive)],artifacts=['archive.json']))
    with a.output.open('x') as f:json.dump(recipe,f,indent=2)
    print(a.output)


if __name__=='__main__':main()
