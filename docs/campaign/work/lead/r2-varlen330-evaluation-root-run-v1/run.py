import datetime,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent;L=P.parent;PLAN=L.parents[3];S=L/'r2-varlen330-evaluation-preparation-v2';V=L/'r2-varlen330-evaluation-root-review-v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def at():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,v):
 with (P/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def pins():
 assert sha(S/'source-manifest.json')=='15c87d1ae6b5579efe14f20d4e96d55ad239c9f4da37f19e3f54d9cef2883077'
 for x in read(S/'source-manifest.json')['files']:
  p=S/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
def release_cache(arm):
 b=read(S/arm/'binding.review.json');root=pathlib.Path(b['model_path']);m=read(root/'campaign-manifest.json');assert sha(root/'campaign-manifest.json')==b['checkpoint_manifest_sha256'];out=[]
 for name in ('model.safetensors','optimizer.pt'):
  p=root/name;before=p.stat();assert before.st_size==m['files'][name]['bytes'];fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);out.append({'path':str(p),'bytes':before.st_size,'metadata_unchanged':True})
 return out
pins();write('launch.json',{'at':at(),'pid':os.getpid(),'waiting_for_payload_verifier':2281062,'authorized_scope':'after verified full paired checkpoints, two sequential matched development evaluations only','promotion_authorized':False});stage='wait_for_payload_verifier';started=time.monotonic()
try:
 while pathlib.Path('/proc/2281062').exists():
  if time.monotonic()-started>9000:raise TimeoutError('payload verifier wait exceeded')
  time.sleep(5)
 assert read(V/'terminal.json')['status']=='payloads_verified_bindings_prepared_root_evaluation_admission_pending'
 pins();review=read(S/'payload-review.json');assert review['status']=='complete_requires_root_eval_admission' and review['paired_draws_denominators_and_scientific_bindings'] is True and review['promotion'] is False
 for arm in ('ordinary_reference','varlen_candidate'):
  a=review['arms'][arm];assert a['native_and_durable_hashes_verified'] is True and a['files_verified_each_copy']==12;assert a['binding_review_sha256']==sha(S/arm/'binding.review.json') and a['checkpoint_review_sha256']==sha(S/arm/'checkpoint-review.json')
 admission=read(S/'root-admission.template.json');admission.update(status='admitted',launch_authorized=True,payload_review_sha256=sha(S/'payload-review.json'),binding_review_sha256={arm:sha(S/arm/'binding.review.json') for arm in ('ordinary_reference','varlen_candidate')});write('admission.json',admission)
 cmds=read(S/'commands.json');stage='admit_verified_bindings';command=[str(P/'admission.json') if x=='ROOT_WRITES_FRESH_ADMISSION_JSON' else x for x in cmds['admit_bindings_after_root_review']]
 with (P/'admit.log').open('x') as log:subprocess.run(command,cwd=PLAN,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
 write('clean-cache-release.json',{'at':at(),'operation':'POSIX_FADV_DONTNEED exact verified checkpoint files only','files':release_cache('ordinary_reference')+release_cache('varlen_candidate')})
 env=dict(os.environ);env.update(read(S/'environment.json'))
 for arm,key in [('ordinary_reference','ordinary_reference_guard_after_admission'),('varlen_candidate','varlen_candidate_guard_after_ordinary_terminal')]:
  pins();stage=arm;g=cmds[key]
  with (P/(arm+'.log')).open('x') as log:
   p=subprocess.Popen(g,cwd=PLAN,env=env,stdout=log,stderr=subprocess.STDOUT);write(arm+'-process.json',{'at':at(),'pid':p.pid});code=p.wait()
  assert code==0,f'{arm} guard exit {code}'
  terminal=read(pathlib.Path(g[g.index('--output')+1])/'terminal.json');assert terminal['status']=='completed' and terminal['child_exit_code']==0
 write('terminal.json',{'at':at(),'status':'matched_evaluation_commands_completed_requires_root_metric_review','promotion':False})
except Exception as exc:
 write('terminal.json',{'at':at(),'status':'stopped_without_retry','stage':stage,'error':str(exc)});raise
