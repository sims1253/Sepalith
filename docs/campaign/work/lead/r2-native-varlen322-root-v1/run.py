import datetime,json,os,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent
PLAN=P.parents[4]
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
write(P/'launch.json',{'at':now(),'pid':os.getpid(),'scope':'two eight-update canaries only; no production continuation'})
stage='preflight'
try:
 for arm in ('ordinary_reference','varlen_candidate'):
  A=P/arm
  with (A/'preflight.log').open('x') as log:
   result=subprocess.run(json.loads((A/'preflight-command.json').read_text()),cwd=PLAN,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
  assert result.returncode==0,f'{arm} preflight exit {result.returncode}'
  write(A/'preflight-result.json',{'at':now(),'exit_code':0})
 for arm in ('ordinary_reference','varlen_candidate'):
  stage=arm;A=P/arm;env=dict(os.environ);env.update(json.loads((A/'environment.json').read_text()))
  with (A/'guard.log').open('x') as log:
   proc=subprocess.Popen(json.loads((A/'guard-command.json').read_text()),cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT)
   write(A/'guard-process.json',{'at':now(),'pid':proc.pid});code=proc.wait()
  assert code==0,f'{arm} guard exit {code}'
  g=json.loads((A/'guard-command.json').read_text());terminal=json.loads((pathlib.Path(g[g.index('--output')+1])/'terminal.json').read_text());assert terminal['status']=='completed' and terminal['child_exit_code']==0,terminal
 write(P/'terminal.json',{'at':now(),'status':'both_commands_completed_requires_root_payload_metric_review','promotion':False})
except Exception as exc:
 write(P/'terminal.json',{'at':now(),'status':'stopped_without_retry','stage':stage,'error':str(exc)});raise
