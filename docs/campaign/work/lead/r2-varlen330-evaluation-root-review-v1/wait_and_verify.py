import datetime,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent;L=P.parent;S=L/'r2-varlen330-evaluation-preparation-v1';C=L/'r2-native-varlen322-root-v1';PLAN=L.parents[3]
def write(name,v):
 with (P/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pins():
 m=S/'source-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='47187abe1dbbe61e12ffdae78d8a4bbb8249370b8a70f6d5fa1677a08f652939'
 for x in json.loads(m.read_text())['files']:
  q=S/x['path'];assert q.stat().st_size==x['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256'],q
pins();write('launch.json',{'at':at(),'pid':os.getpid(),'waiting_for_controller':2179203,'cpu_only':True,'automatic_evaluation_launch':False});stage='wait_for_paired_canary_terminal';start=time.monotonic()
try:
 while pathlib.Path('/proc/2179203').exists():
  if time.monotonic()-start>7500:raise TimeoutError('paired canary controller wait exceeded')
  time.sleep(5)
 assert json.loads((C/'terminal.json').read_text())['status']=='both_commands_completed_requires_root_payload_metric_review'
 pins();stage='verify_full_payloads_prepare_matched_bindings';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (P/'prepare.log').open('x') as log:
  proc=subprocess.Popen(['taskset','-c','12,14','nice','-n','10','/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B',str(S/'prepare.py')],cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT)
  write('verify-process.json',{'at':at(),'pid':proc.pid});code=proc.wait(timeout=1800)
 assert code==0,f'payload verification exit {code}'
 write('terminal.json',{'at':at(),'status':'payloads_verified_bindings_prepared_root_evaluation_admission_pending','evaluation_launched':False})
except Exception as exc:
 write('terminal.json',{'at':at(),'status':'stopped_without_retry','stage':stage,'error':str(exc)});raise
