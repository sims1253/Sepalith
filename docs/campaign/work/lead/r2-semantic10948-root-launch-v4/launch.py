import pathlib,json,hashlib,subprocess,os,time,datetime,signal
P=pathlib.Path(__file__).resolve().parent;SOURCE=P.parent/'r2-semantic10948-provider-materialization-v2';OUT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic10948-provider-materialization-v2');PY='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,r):
 with (P/name).open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def verify():
 assert sha(SOURCE/'source-manifest.json')=='7546d5457c299edfbf69bc0681327f08bad5041e2a146fe60fd66278dd58c770'
 for x in json.loads((SOURCE/'source-manifest.json').read_text())['files']:
  f=SOURCE/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'],str(f)
verify();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
write('controller-launch.json',{'at':now(),'pid':os.getpid(),'source_manifest_sha256':sha(SOURCE/'source-manifest.json'),'scope':'prepare10948 exact inputs and16K/reserve2048 twoCPU render lanes; no training admission'})
children=[];logs=[];stage='prepare_inputs'
try:
 with (P/'prepare.log').open('x') as log:r=subprocess.run(['taskset','-c','0,2','nice','-n','10','ionice','-c','3',PY,'-B',str(P/'prepare_inputs_root.py')],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 assert r.returncode==0,f'input preparation exit {r.returncode}'
 m=json.loads((OUT/'render-inputs-v1/manifest.json').read_text());assert m['rows']==10948 and len(m['shards'])==15 and m['target_or_gold_copied'] is False
 assert not (OUT/'render-16k-01').exists();verify();stage='render16';launch=[]
 for lane,core in [(0,0),(1,2)]:
  command=['timeout','--signal=TERM','--kill-after=30s','9000','bash',str(SOURCE/'run_lane.sh'),str(lane),str(core),'16384','2048',str(OUT/'render-inputs-v1'),str(OUT/'render-16k-01')]
  log=(P/f'lane{lane}.log').open('x');logs.append(log);child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);launch.append({'lane':lane,'core':core,'pid':child.pid,'command':command})
 write('lanes-launch.json',{'at':now(),'controller_pid':os.getpid(),'input_manifest_sha256':sha(OUT/'render-inputs-v1/manifest.json'),'lanes':launch})
 while any(c.poll() is None for c in children):
  failed=[c for c in children if c.poll() not in (None,0)]
  if failed:raise RuntimeError('render lane failed:'+','.join(str(c.pid)+':'+str(c.returncode) for c in failed))
  time.sleep(2)
 assert all(c.returncode==0 for c in children)
 write('terminal.json',{'at':now(),'status':'render16_commands_completed_requires_root_output_review','lane_exit_codes':[c.returncode for c in children],'training_admitted':False,'next':'Verify all15 terminal outputs; prepare32K fallback using prediction input only'})
except Exception as e:
 for c in children:
  if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
 for c in children:
  try:c.wait(timeout=35)
  except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
 write('terminal.json',{'at':now(),'status':'failed_without_retry','stage':stage,'exception_type':type(e).__name__,'detail':str(e),'training_admitted':False});raise
finally:
 for log in logs:log.close()
