#!/usr/bin/env python3
"""Freeze both b4 timing tiers before dispatch."""
import argparse,json,sys
from pathlib import Path
from prepare_v1c import freeze
from v1c_artifacts import digest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'packages/sepalith/src'))
from sepalith.runner import Runner

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('state','owner','archive'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if any(not x.is_absolute() for x in vars(a).values()):p.error('Use absolute paths')
 runner=Runner(a.state)
 if not runner.plan()['paused']:p.error('Prepare between experiments')
 quality=json.loads((a.owner/'b4-quant-quality.recipe.json').read_text())
 traces_recipe=json.loads((a.owner/'s1-gpu.recipe.v3.json').read_text())
 cpu=json.loads((a.owner/'plain921.recipe.v2.json').read_text())
 files={}
 for item in quality['inputs']+traces_recipe['inputs']:
  path=Path(item['path'])
  if path.name in ('Q8_0.gguf','Q4_K_M.gguf','Q4_K_M_imatrix.gguf','traces.jsonl'):
   if digest(path)!=item['sha256']:raise ValueError('Frozen model or trace changed')
   files[path.name]=path
 if len(files)!=4:raise ValueError('Missing frozen assets')
 assets,common=freeze(a.owner/'assets-b4-timing',files)
 includes=['packages/sepalith/src','experiments/eval/spec_bench.py','scripts/migration/s1_gpu.py',
           'scripts/migration/b4_timing.py','scripts/migration/v1c_artifacts.py']
 snapshot=runner.snapshot(ROOT,includes)
 reviews=[json.loads((a.owner/'pi-b4-timing'/(provider+'.json')).read_text()) for provider in ('opencode','zai')]
 for tier,previous in [('gpu',quality),('cpu',cpu)]:
  runtime=Path(previous['env']['S1_RUNTIME'] if tier=='gpu' else previous['env']['LD_LIBRARY_PATH'])
  inputs=list(common)
  for item in previous['inputs']:
   path=Path(item['path'])
   if path.is_relative_to(runtime) or str(path).startswith(('/usr/','/lib/')):
    if digest(path)!=item['sha256']:raise ValueError('Frozen runtime changed')
    inputs.append(item)
  env=dict(previous['env'],S1_RUNTIME=str(runtime),B4_TIMING_TIER=tier)
  recipe=dict(schema_version=1,id='b4-quant-timing-'+tier+'-20260908',snapshot=snapshot,
   resource='gpu' if tier=='gpu' else 'quiet',python=previous['python'],depends_on=[],inputs=inputs,env=env,
   provenance=dict(queue_item='S2',scope='b4 Q8/stock Q4/imatrix Q4 and Q8 bookend; 10 fixed 2K traces x3 reps =120 rows',
     tier=tier,source_includes=includes,pi_reviews=reviews,runtime_bound_seconds=1800 if tier=='gpu' else 5400,
     adoption='NOT-ASSESSED; cycle timing and prior quality must be reviewed together'),steps=[])
  for action,artifacts in [('prepare',['prepared.json']),('measure',['per_request.jsonl','telemetry.jsonl','warmups.jsonl']),('evaluate',['evaluation.json','verdict.json'])]:
   recipe['steps'].append(dict(id=action,argv=['{python}','{source}/scripts/migration/b4_timing.py',action,'--run','{run}','--assets',str(assets)],artifacts=artifacts))
  recipe['steps'].append(dict(id='archive',argv=['{python}','{source}/scripts/migration/v1c_artifacts.py','archive','--run','{run}','--archive',str(a.archive/tier)],artifacts=['archive.json']))
  output=a.owner/('b4-timing-'+tier+'.recipe.json')
  with output.open('x') as f:json.dump(recipe,f,indent=2)
  print(output)

if __name__=='__main__':main()
