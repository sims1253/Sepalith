import pathlib,json,hashlib,subprocess,os,time,datetime,signal
P=pathlib.Path(__file__).resolve().parent;SOURCE=P.parent/'r2-semantic9535-provider-preparation-v1';OUT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1');INPUT=OUT/'render-inputs-v1'
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with(P/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert sha(SOURCE/'source-manifest.json')=='8c1231d49066dcc6335c4b6a9c244249231874e31895d2799bba327a5672d9dc'
for x in json.loads((SOURCE/'source-manifest.json').read_text())['files']:
 p=SOURCE/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
m=json.loads((INPUT/'manifest.json').read_text());assert m['provider_rows']==9534 and m['geometry_preparation_holds']==1 and m['exact_supported_denominator_closure']is True and m['training_admission']is False and [x['shard']for x in m['shards']]==list(range(27,41));allids=set();allholds=[]
def no_targets(v):
 if isinstance(v,dict):return all(not any(w in k.lower()for w in ('target','gold','completion','reward')) and no_targets(x)for k,x in v.items())
 if isinstance(v,list):return all(no_targets(x)for x in v)
 return True
for x in m['shards']:
 for key in ('provider','sidecar','preparation_hold'):
  b=x[key];p=INPUT/b['path'];assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256']
 rows=[json.loads(l)for l in(INPUT/x['provider']['path']).open()];side=[json.loads(l)for l in(INPUT/x['sidecar']['path']).open()];holds=[json.loads(l)for l in(INPUT/x['preparation_hold']['path']).open()]
 ids={r['row_id']for r in rows};assert len(rows)==len(ids)==len(side)==x['provider_rows'] and ids=={r['row_id']for r in side} and not ids&allids;allids|=ids;allholds+=holds
 for r in rows:
  assert no_targets(r) and hashlib.sha256(r['preedit_text'].encode()).hexdigest()==r['preedit_sha256']
 assert len(holds)==x['preparation_holds'] and len(rows)+len(holds)==x['supported_denominator']
assert len(allids)==9534 and len(allholds)==1
write('input-root-review.json',{'at':at(),'manifest_sha256':sha(INPUT/'manifest.json'),'rows':9534,'holds':allholds,'exact9535closure':True,'all_payload_hashes_and_row_ID_joins_verified':True,'target_keys_absent_and_preedit_hashes_verified':True,'training_admitted':False})
assert not(OUT/'render-16k-v1').exists();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');children=[];logs=[]
try:
 launch=[]
 for lane,core in [(0,8),(1,10)]:
  command=['timeout','--signal=TERM','--kill-after=30s','9000','bash',str(SOURCE/'run_lane.sh'),str(lane),str(core),'16384','2048',str(INPUT),str(OUT/'render-16k-v1')];log=(P/f'lane{lane}.log').open('x');logs.append(log);child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);launch.append({'lane':lane,'core':core,'pid':child.pid,'command':command})
 write('launch.json',{'at':at(),'controller_pid':os.getpid(),'input_manifest_sha256':sha(INPUT/'manifest.json'),'rows':9534,'preparation_holds':1,'lanes':launch})
 while any(c.poll()is None for c in children):
  failed=[c for c in children if c.poll()not in(None,0)]
  if failed:raise RuntimeError('render lane failed:'+','.join(str(c.pid)+':'+str(c.returncode)for c in failed))
  time.sleep(2)
 assert all(c.returncode==0 for c in children)
 write('terminal.json',{'at':at(),'status':'commands_complete_requires_root_output_review','exit_codes':[c.returncode for c in children],'training_admitted':False,'next':'verify9534outputs,32Kfallbackforpolicyholds, bindfinal10682dedup sources beforematerialization'})
except Exception as e:
 for c in children:
  if c.poll()is None:os.killpg(c.pid,signal.SIGTERM)
 for c in children:
  try:c.wait(timeout=35)
  except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
 write('terminal.json',{'at':at(),'status':'failed_no_retry','error':str(e),'training_admitted':False});raise
finally:
 for log in logs:log.close()
