import json,datetime
from pathlib import Path
import live_observer as o
extra = """
out['checkpoint_files']={}
for rel in ('artifacts/archive/full/checkpoint-317','artifacts/cloud-persistence','artifacts/training/evaluations'):
 d=os.path.join(p,rel)
 out['checkpoint_files'][rel]=[{'name':n,'bytes':os.stat(os.path.join(d,n)).st_size} for n in os.listdir(d) if os.path.isfile(os.path.join(d,n))] if os.path.isdir(d) else []
print(json.dumps(out,sort_keys=True))
"""
o.REMOTE=o.REMOTE.replace('print(json.dumps(out,sort_keys=True))',extra)
v=o.observe(use_hf=False)
f=Path(__file__).with_name('root-checkpoint-observation-1140.json')
assert not f.exists()
f.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'status':v['provider']['provider_run_status'],'checkpoint_files':v['remote'].get('checkpoint_files',{}),'latest':v['remote'].get('telemetry',{}).get('last_records',[])[-2:]}))
