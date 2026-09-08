#!/usr/bin/env python3
"""Bind frozen S1 inputs and the separately frozen CPU runtime for a scout."""
import argparse
import json
from pathlib import Path
import sys
from v1c_artifacts import digest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/sepalith/src'))
from sepalith.runner import Runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','gpu_recipe','cpu_recipe','archive','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if any(not path.is_absolute() for path in vars(a).values()):p.error('Use absolute paths')
    r=Runner(a.state)
    if not r.plan()['paused']:p.error('Prepare between experiments')
    gpu=json.loads(a.gpu_recipe.read_text());cpu=json.loads(a.cpu_recipe.read_text())
    if gpu['provenance']['queue_item']!='S1':raise ValueError('S1 model/trace recipe required')
    runtime=Path(cpu['env']['LD_LIBRARY_PATH']);inputs=[]
    for item in cpu['inputs']:
        path=Path(item['path'])
        if path.is_relative_to(runtime) or str(path).startswith(('/usr/','/lib/')):inputs.append(item)
    for item in gpu['inputs']:
        if Path(item['path']).name in ('model.gguf','traces.jsonl'):inputs.append(item)
    for item in inputs:
        if digest(Path(item['path']))!=item['sha256']:raise ValueError('Frozen dependency changed')
    model=next(Path(i['path']) for i in inputs if Path(i['path']).name=='model.gguf')
    assets=next(Path(i['path']).parent for i in inputs if Path(i['path']).name=='traces.jsonl')
    includes=['packages/sepalith/src','experiments/eval/spec_bench.py','scripts/migration/s1_gpu.py',
              'scripts/migration/s1_cpu_scout.py','scripts/migration/v1c_artifacts.py']
    recipe=dict(schema_version=1,id='s1-cpu-ngram-scout-20260908',snapshot=r.snapshot(ROOT,includes),resource='quiet',
        python=cpu['python'],depends_on=[],inputs=inputs,env=dict(cpu['env'],SCOUT_MODEL=str(model),SCOUT_RUNTIME=str(runtime)),
        provenance=dict(queue_item='S1',scope='Reduced CPU ngram scout,20matched traces/class,1cold rep,6depths,baseline and bookend',
        source_includes=includes,runtime_bound_seconds=18000,pi_review='Muse85.19s and GLM304.84s, both max,600s caps; private pi-s1-cpu-value receipts',
        deviation='Not full S1; no warm measurement; no MTP or model-draft CPU sweep; candidates need full confirmation'),steps=[])
    for action,artifacts in [('prepare',['prepared.json']),('measure',['per_request.jsonl']),('evaluate',['evaluation.json','verdict.json'])]:
        recipe['steps'].append(dict(id=action,argv=['{python}','{source}/scripts/migration/s1_cpu_scout.py',action,'--run','{run}','--assets',str(assets)],artifacts=artifacts))
    recipe['steps'].append(dict(id='archive',argv=['{python}','{source}/scripts/migration/v1c_artifacts.py','archive','--run','{run}','--archive',str(a.archive)],artifacts=['archive.json']))
    with a.output.open('x') as f:json.dump(recipe,f,indent=2)
    print(a.output)


if __name__=='__main__':main()
