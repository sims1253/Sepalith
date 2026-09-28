"""Bounded root-owned checkpoint322 verification and development evaluation."""
import datetime,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
PLAN=P.parents[4]
PRIOR=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-cpt194-to322-root-v1-host-supervision-b')
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj):
 with (P/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def pins():
 for file,digest in json.loads((P/'handoff-pins.json').read_text()).items():
  assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==digest,file
pins();write('handoff-launch.json',{'at':at(),'pid':os.getpid(),'wait_for_pids':[2064033,2068268],'max_wait_seconds':3600,'authorized_next':'verify complete checkpoint322 then guarded development evaluation only'})
start=time.monotonic();stage='waiting_for_training_terminal'
try:
 while any(pathlib.Path(f'/proc/{pid}').exists() for pid in (2064033,2068268)):
  if time.monotonic()-start>3600:raise TimeoutError('training did not become terminal within handoff wait')
  time.sleep(5)
 stage='verifying_training_terminal';terminal=json.loads((PRIOR/'terminal.json').read_text());assert terminal['status']=='completed' and terminal['child_exit_code']==0,terminal
 pins();stage='verifying_complete_native_and_durable_checkpoint';env=dict(os.environ);env.update(json.loads((P/'environment.json').read_text()));cpu=dict(env,CUDA_VISIBLE_DEVICES='')
 with (P/'prepare.log').open('x') as log:
  result=subprocess.run(['taskset','-c','12,14','nice','-n','10','/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(P/'prepare.py')],cwd=PLAN,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 assert result.returncode==0,f'checkpoint preparation exit {result.returncode}'
 review=json.loads((P/'checkpoint-review.json').read_text());assert review['step']==322 and review['cursor']==4096 and review['native_and_durable_payloads_verified'] is True
 stage='guarded_development_evaluation';pins();write('evaluation-launch-acceptance.json',{'at':at(),'checkpoint_review':review,'guard_command':json.loads((P/'guard-command.json').read_text()),'evaluation_only':True,'promotion_authorized':False})
 with (P/'guard.log').open('x') as log:
  child=subprocess.Popen(json.loads((P/'guard-command.json').read_text()),cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT)
  write('evaluation-guard-process.json',{'at':at(),'pid':child.pid});code=child.wait()
 assert code==0,f'evaluation guard exit {code}'
 write('handoff-terminal.json',{'at':at(),'status':'evaluation_command_completed_requires_root_metric_review','exit_code':0,'stage':stage,'training_resumed':False,'promoted':False})
except Exception as exc:
 write('handoff-terminal.json',{'at':at(),'status':'stopped_without_retry','stage':stage,'exception_type':type(exc).__name__,'detail':str(exc),'training_resumed':False,'promoted':False});raise
