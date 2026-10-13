import json,os,time
from pathlib import Path
telemetry=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-expandable-pilot-v1/telemetry.jsonl')
recipe=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-full-corpus-expandable-pilot-root-v1/bound-recipe.json')
stop=Path(json.loads(recipe.read_text())['outputs']['graceful_stop'])
expected={'action':'save_and_stop','bound_recipe_sha256':'64aaf99bff0b4552347025a3167a85c5988de6d319fab14df754b6586ec6520e'}
started=time.monotonic()
while time.monotonic()-started<3600:
 if not Path('/proc/1763848').exists():raise SystemExit('trainer terminal before stop request')
 rows=[json.loads(x) for x in telemetry.read_text().splitlines()]
 if any(x.get('event')=='log' and x.get('step',0)>=89 for x in rows):
  stop.parent.mkdir(parents=True,exist_ok=True)
  if stop.exists():
   assert json.loads(stop.read_text())==expected
  else:
   temp=stop.with_name(stop.name+'.root90.tmp')
   with temp.open('x') as f:json.dump(expected,f);f.write('\n');f.flush();os.fsync(f.fileno())
   temp.replace(stop)
  print(json.dumps({'status':'graceful_stop_requested','at':time.time(),'after_observed_step':89,'target_next_step':90,'path':str(stop)}),flush=True)
  break
 time.sleep(1)
else:raise SystemExit('stop watcher expired without mutation')
