import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'r2-cloud-cpt-live-observer-v1'))
import live_observer as o
extra=r'''
out['evaluations']=[]
for f in sorted(glob.glob(os.path.join(p,'artifacts','archive','evaluations','step-*.json'))):
 if os.path.getsize(f)>2000000:continue
 raw=open(f,'rb').read();v=json.loads(raw)
 out['evaluations'].append({'file':os.path.basename(f),'sha256':__import__('hashlib').sha256(raw).hexdigest(),'status':v.get('status'),'step':v.get('step'),'result':v.get('result'),'identity_sha256':__import__('hashlib').sha256(json.dumps(v.get('identity'),sort_keys=True,separators=(',',':')).encode()).hexdigest()})
'''
o.REMOTE=o.REMOTE.replace('print(json.dumps(out,sort_keys=True))',extra+'\nprint(json.dumps(out,sort_keys=True))')
value=o.observe(use_hf=False);path=Path(__file__).with_name('observation.json');assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'provider':value['provider']['provider_run_status'],'remote_status':value['remote'].get('status'),'evaluations':value['remote'].get('evaluations'),'last_records':value['remote'].get('telemetry',{}).get('last_records',[])[-2:]}))
