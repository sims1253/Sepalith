#!/usr/bin/env python3
"""Root-only independent Anyscale deadline guard. Import/tests do not call APIs."""
import argparse,datetime,json,os,re,subprocess,time
from pathlib import Path
CLI='/home/m0hawk/.local/bin/anyscale'
TERMINAL={'SUCCEEDED','FAILED','TERMINATED','ERRORED','OUT_OF_RETRIES'}

def validate(name,deadline,now):
 if not re.fullmatch(r'sepalith-r2-control-[0-9a-f]{32}',name):raise ValueError('unique root run name required')
 dt=datetime.datetime.fromisoformat(deadline.replace('Z','+00:00'))
 if dt.tzinfo is None:raise ValueError('deadline requires timezone')
 epoch=dt.timestamp();ceiling=datetime.datetime(2026,9,14,6,15,tzinfo=datetime.timezone.utc).timestamp()
 if not now<epoch<=min(now+21600,ceiling):raise ValueError('deadline must be future, within six hours and Monday 06:15 UTC')
 return epoch

def cli(operation,name):
 selector='--id' if name.startswith('prodjob_') else '--name'
 argv=[CLI,'job',operation,selector,name]
 if selector=='--name':argv+=['--cloud','Anyscale Cloud']
 if operation=='status':argv+=['-o','json']
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
  if p.returncode:return {'ok':False,'exit_code':p.returncode}
  if operation=='terminate':return {'ok':True,'termination_requested':True}
  d=json.loads(p.stdout)
  return {'ok':True,'id':d.get('id'),'name':d.get('name'),'state':d.get('state')}
 except Exception as e:return {'ok':False,'error_type':type(e).__name__}

def monitor(name,deadline,record,*,clock=time.time,sleep=time.sleep,call=cli):
 start=clock();errors=0;last_id=None
 while True:
  now=clock()
  if now>=deadline:reason='absolute_deadline';break
  status=call('status',name);record({'event':'status','utc_epoch':clock(),**status})
  if status.get('ok'):
   if status.get('name')!=name or not re.fullmatch(r'prodjob_[a-z0-9]+',status.get('id') or ''):reason='unexpected_job_identity';break
   if last_id is not None and status['id']!=last_id:reason='job_id_changed';break
   last_id=status['id'];errors=0
   if status.get('state') in TERMINAL:return {'status':'terminal_observed','state':status['state'],'job_id':last_id,'termination_requested':False}
  elif last_id is None:
   # Submission/upload may still be running. Keep the unique-name guard
   # alive until the absolute deadline; do not abandon a job created later.
   record({'event':'pending_submission','utc_epoch':clock(),'name':name,
           'status':'no_exact_job_id_observed','absolute_deadline_epoch':deadline})
  elif now-start>=120:
   errors+=1
   if errors>=3:reason='status_api_failed';break
  sleep(min(10,max(0,deadline-clock())))
 result=call('terminate',last_id or name);record({'event':'terminate','reason':reason,'utc_epoch':clock(),**result})
 if not result.get('ok'):return {'status':'termination_unconfirmed','reason':reason,'job_id':last_id,'termination_requested':False}
 confirmation_end=clock()+120
 while clock()<confirmation_end:
  status=call('status',name);record({'event':'termination_status','utc_epoch':clock(),**status})
  if status.get('ok') and status.get('name')==name and status.get('state') in TERMINAL and (last_id is None or status.get('id')==last_id):
   return {'status':'terminal_observed','state':status['state'],'reason':reason,'job_id':status.get('id'),'termination_requested':True}
  sleep(min(5,max(0,confirmation_end-clock())))
 return {'status':'termination_unconfirmed','reason':reason,'job_id':last_id,'termination_requested':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--deadline-utc',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});deadline=validate(a.name,a.deadline_utc,time.time());a.output.mkdir(exist_ok=False)
 (a.output/'armed.json').write_text(json.dumps({'name':a.name,'deadline':a.deadline_utc,'pid':os.getpid(),'provider_timeout_required':True})+'\n')
 with (a.output/'observations.jsonl').open('x') as f:
  def record(r):f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  result=monitor(a.name,deadline,record)
 (a.output/'terminal.json').write_text(json.dumps(result,indent=2)+'\n')
 if result['status']!='terminal_observed':raise SystemExit(2)

if __name__=='__main__':main()
