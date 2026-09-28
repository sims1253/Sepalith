#!/usr/bin/env python3
"""Independent 29,100-second Anyscale guard with corrected ID operations."""
import argparse,datetime,json,os,re,subprocess,time
from pathlib import Path
CLI='/home/m0hawk/.local/bin/anyscale';TERMINAL={'SUCCEEDED','FAILED','TERMINATED','ERRORED','OUT_OF_RETRIES'}
def validate(name,deadline,now):
 if not re.fullmatch(r'sepalith-cpt-[0-9a-f]{32}',name):raise ValueError('unique CPT name required')
 d=datetime.datetime.fromisoformat(deadline.replace('Z','+00:00'))
 if d.tzinfo is None:raise ValueError('deadline timezone required')
 epoch=d.timestamp()
 if not now<epoch<=now+29100:raise ValueError('deadline exceeds29100 seconds')
 return epoch
def cli(operation,name):
 selector='--id' if name.startswith('prodjob_') else '--name';argv=[CLI,'job',operation,selector,name]
 if selector=='--name':argv+=['--cloud','Anyscale Cloud']
 if operation=='status':argv+=['-o','json']
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
  if p.returncode:return {'ok':False,'exit_code':p.returncode}
  if operation=='terminate':return {'ok':True,'termination_requested':True}
  x=json.loads(p.stdout);return {'ok':True,'id':x.get('id'),'name':x.get('name'),'state':x.get('state')}
 except Exception as e:return {'ok':False,'error_type':type(e).__name__}
def monitor(name,deadline,record,clock=time.time,sleep=time.sleep,call=cli):
 start=clock();job=None;errors=0
 while clock()<deadline:
  x=call('status',job or name);record({'event':'status','utc_epoch':clock(),**x})
  if x.get('ok'):
   if x.get('name')!=name or not re.fullmatch(r'prodjob_[a-z0-9]+',x.get('id') or ''):reason='unexpected_job_identity';break
   if job and x['id']!=job:reason='job_id_changed';break
   job=x['id'];errors=0
   if x.get('state') in TERMINAL:return {'status':'terminal_observed','state':x['state'],'job_id':job,'termination_requested':False}
  elif job and clock()-start>=120:
   errors+=1
   if errors>=3:reason='status_api_failed';break
  sleep(min(10,max(0,deadline-clock())))
 else:reason='absolute_deadline'
 x=call('terminate',job or name);record({'event':'terminate','reason':reason,'utc_epoch':clock(),**x})
 return {'status':'termination_requested' if x.get('ok') else 'termination_unconfirmed','reason':reason,'job_id':job,'termination_requested':bool(x.get('ok'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--deadline-utc',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();deadline=validate(a.name,a.deadline_utc,time.time());a.output.mkdir(exist_ok=False)
 (a.output/'armed.json').write_text(json.dumps({'name':a.name,'deadline':a.deadline_utc,'pid':os.getpid()})+'\n')
 with (a.output/'observations.jsonl').open('x') as f:
  def record(x):f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
  result=monitor(a.name,deadline,record)
 (a.output/'terminal.json').write_text(json.dumps(result,indent=2)+'\n')
 if result['status'] not in ('terminal_observed','termination_requested'):raise SystemExit(2)
if __name__=='__main__':main()
