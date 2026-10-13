#!/usr/bin/env python3
"""Compare complete E750 native DEV75 with three pinned prior runs."""
import argparse,datetime,hashlib,json
from pathlib import Path

PANEL='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'
BASELINES={
 'incumbent':('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-task-global500-native-a/dev-results/results.json','a68190b6d87e175f01073021b16c1c8e16d57226b2d95977f2ad908e5a2fd2e3'),
 'C250':('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-c250-native-dev-v1/dev-results/results.json','1f4e10ee118a98405cb9364157cbbdabf3033b6f9e7135cddf8c24177e6cb9e5'),
 'D500':('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-d500-native-dev-v1/dev-results/results.json','ba909888aee088f3be4039eb186d2aad49ffa30c8199832d7ddbc70d4ca24ae9'),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,expected=None):
 path=Path(path);actual=sha(path)
 if expected and actual!=expected:raise ValueError('pinned baseline hash differs: '+str(path))
 value=json.loads(path.read_text());rows=value.get('results',[])
 if value.get('status')!='complete' or value.get('gate',{}).get('panel_sha256')!=PANEL:raise ValueError('complete corrected DEV75 required')
 if len(rows)!=75 or len({r['id'] for r in rows})!=75 or value.get('denominators')!={'attempted_cases':75,'complete_responses':75,'edit_cases':43,'expected_cases':75,'strict_noop_cases':32}:raise ValueError('DEV75 denominator differs')
 return value,actual,{r['id']:r for r in rows}
def metrics(v):return v['counts']
def main():
 p=argparse.ArgumentParser();p.add_argument('--e750',type=Path,required=True);p.add_argument('--e750-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('fresh comparison output required')
 runs={name:load(path,digest) for name,(path,digest) in BASELINES.items()};runs['E750']=load(a.e750,a.e750_sha256)
 ids=set(runs['incumbent'][2]);invariants=('expected_noop','family','operation_label','package_id','prompt_sha256','target_sha256','hf_prompt_ids_sha256')
 for name,(_,_,rows) in runs.items():
  if set(rows)!=ids:raise ValueError(name+' case IDs differ')
  for case in ids:
   for field in invariants:
    if rows[case][field]!=runs['incumbent'][2][case][field]:raise ValueError(f'{name} case contract differs: {case} {field}')
 result={'schema':'sepalith.native-dev75.four-way-comparison.v1','status':'complete_development_comparison','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'panel_sha256':PANEL,'runs':{name:{'path':str(Path(BASELINES[name][0]) if name in BASELINES else a.e750),'sha256':row[1],'counts':metrics(row[0])} for name,row in runs.items()},'promotion':False,'limitations':['DEV-only evidence; final set remains sealed.','Root must make any promotion decision after reviewing paired cases and aggregate tradeoffs.']}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'runs':result['runs'],'promotion':False}))
if __name__=='__main__':main()
