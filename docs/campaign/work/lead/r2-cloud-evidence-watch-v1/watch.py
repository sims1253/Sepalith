import sys,json,time,datetime,hashlib,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'r2-cloud-cpt-live-observer-v1'))
import live_observer as o
OUT=Path('/mnt/e/sepalith/campaign-20260915/cloud-readback/CPT-v4-evidence-watch-v1')
DEADLINE=datetime.datetime(2026,9,14,18,42,tzinfo=datetime.timezone.utc).timestamp()
EXTRA=r"""
out['evaluations']=[]
for f in sorted(glob.glob(os.path.join(p,'artifacts','archive','evaluations','step-*.json'))):
 if os.path.getsize(f)>2000000: raise ValueError('evaluation exceeds admitted bound')
 raw=open(f,'rb').read();v=json.loads(raw)
 out['evaluations'].append({'file':os.path.basename(f),'sha256':__import__('hashlib').sha256(raw).hexdigest(),'evaluation':v})
out['telemetry']['all_records']=records
"""
o.REMOTE=o.REMOTE.replace('print(json.dumps(out,sort_keys=True))',EXTRA+'\nprint(json.dumps(out,sort_keys=True))')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 while time.time()<DEADLINE:
  at=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  try:
   result=o.observe(use_hf=False)
   raw=(json.dumps(result,indent=2)+'\n').encode()
   dest=OUT/(at+'.json');tmp=dest.with_suffix('.tmp')
   with open(tmp,'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
   tmp.rename(dest)
   print(json.dumps({'at':at,'status':result['provider']['provider_run_status'],'remote_status':result['remote'].get('status','observed'),'file':str(dest),'sha256':hashlib.sha256(raw).hexdigest(),'optimizer_steps':result['remote'].get('telemetry',{}).get('optimizer_steps')}),flush=True)
  except Exception as e:
   print(json.dumps({'at':at,'status':'observation_failed_not_terminal','error_type':type(e).__name__}),flush=True)
  time.sleep(min(120,max(0,DEADLINE-time.time())))
 print(json.dumps({'status':'watch_deadline_reached','training_terminal_claim':False}),flush=True)
if __name__=='__main__':main()
