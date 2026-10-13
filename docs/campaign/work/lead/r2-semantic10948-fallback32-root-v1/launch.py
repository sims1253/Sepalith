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
write('controller-launch.json',{'at':now(),'pid':os.getpid(),'scope':'485 prediction-only held inputs32K reserve2048;2CPUlanes;no training'})
children=[];logs=[];stage='prepare_inputs'
try:
 m=json.loads((OUT/'fallback-32k-inputs-01/manifest.json').read_text());assert m['denominator']==10948 and m['rerun32_rows']==485 and m['target_or_gold_used'] is False and len(m['shards'])==15
 total=0
 for item in m['shards']:
  f=OUT/'fallback-32k-inputs-01'/item['path'];assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes'];rows=[json.loads(x)for x in f.open()];assert len(rows)==item['rows'];original={json.loads(x)['row_id']:json.loads(x)for x in (OUT/'render-inputs-v1'/item['path']).open()}
  for row in rows:assert row==original[row['row_id']]
  total+=len(rows)
 assert total==485 and not (OUT/'render-32k-01').exists();verify();stage='render32';launch=[]
 for lane,core in [(0,0),(1,2)]:
  command=['timeout','--signal=TERM','--kill-after=30s','3600','bash',str(SOURCE/'run_lane.sh'),str(lane),str(core),'32768','2048',str(OUT/'fallback-32k-inputs-01'),str(OUT/'render-32k-01')]
  log=(P/f'lane{lane}.log').open('x');logs.append(log);child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);launch.append({'lane':lane,'core':core,'pid':child.pid,'command':command})
 write('lanes-launch.json',{'at':now(),'controller_pid':os.getpid(),'input_manifest_sha256':sha(OUT/'fallback-32k-inputs-01/manifest.json'),'lanes':launch})
 while any(c.poll() is None for c in children):
  failed=[c for c in children if c.poll() not in (None,0)]
  if failed:raise RuntimeError('render lane failed:'+','.join(str(c.pid)+':'+str(c.returncode) for c in failed))
  time.sleep(2)
 assert all(c.returncode==0 for c in children)
 write('terminal.json',{'at':now(),'status':'render32_commands_completed_requires_root_output_review','lane_exit_codes':[c.returncode for c in children],'training_admitted':False,'next':'Verify485fallback outputs, finalize policy then target-aware materialization/dedup under rootreview'})
except Exception as e:
 for c in children:
  if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
 for c in children:
  try:c.wait(timeout=35)
  except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
 write('terminal.json',{'at':now(),'status':'failed_without_retry','stage':stage,'exception_type':type(e).__name__,'detail':str(e),'training_admitted':False});raise
finally:
 for log in logs:log.close()
