import datetime, hashlib, json, os, pathlib, sys
P=pathlib.Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
W=pathlib.Path(__file__).parent; C=P/'work/corrected-rl-admission-audit-v1'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
assert sha(C/'artifact-manifest.json')=='46e5813f32be652d62ce114c89d18df7c49d254609c7e410507d6aa58a1dff59'
checks=[]
for x in json.loads((C/'artifact-manifest.json').read_text())['files']:
 p=P/x['path']; assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],p
 checks.append(str(p))
assert sha(P/'receipts/RL-08-corrected-rl-admission-audit-v1.json')=='590016b394d7c8862253e80fdaf3b7f1ef928a256339bfecb626d082daf57993'
r=json.loads((C/'fresh5.recipe.json').read_text()); assert sha(C/'fresh5.recipe.json')=='6661af89eb83fd95070f939290a3e74ff2c1e0b09c561d2d16a1704f32bbabee'
source=pathlib.Path(r['identity']['source']['frozen_source_root'])
sm=json.loads((source.parent/'manifest.json').read_text())
for x in sm['files']:assert sha(source/x['path'])==x['sha256'],x['path']
for x in json.loads((C/'source-pins.json').read_text())['files']:assert sha(x['path'])==x['sha256'],x['path']
guard=P/'work/lead/host_memory_guard_v3.py'
for p,s in [(guard,'9518f5d10ea92c424df10f4d49be32f5dd1bf8a42a0deae5d13ea2998f18a845'),(guard.with_name('host_memory_policy.py'),'ed39464f8981b562dd4f91ef8ad069edae03e53da999c274507973ff0df3831a'),(guard.with_name('post_load_cache.py'),'7204fc46c412d0285b6bfc931bb774b938bf5bfef87220346149e285d6f5f8a4')]:assert sha(p)==s,p
for k in ['output_dir','archive_root','telemetry_path']:assert not pathlib.Path(r[k]).exists(),k
data_receipt=P/'receipts/RL-08-corrected-data-root-admission.json'
put(data_receipt,{'task':'RL-08','owner':'lead','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'admitted_corrected_train_data','scope':'TRAIN data and schedule authority only; bounded fresh5 launch needs separate admission','worker_receipt_sha256':sha(P/'receipts/RL-08-corrected-rl-admission-audit-v1.json'),'artifact_manifest_sha256':sha(C/'artifact-manifest.json'),'data_identity_sha256':hashlib.sha256(json.dumps(r['identity']['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'rows':8246,'corrected_finish':2222,'unchanged':6024,'excluded':194,'source_draws':24000,'source_schedule_sha256':r['data']['source_draw_schedule_sha256'],'acceptance':'Root read scientific audit and exact validators; all candidate and frozen source pins verified; retained prompt/sidecar and TRAIN/DEV disjointness checks accepted. Prior corrected full R parse evidence is retained, not rerun.','final_access':False})
r['rl02_admission']={'path':str(data_receipt),'sha256':sha(data_receipt),'status':'admitted_corrected_train_data'}
r['identity']['source']['rl02_admission_receipt_sha256']=sha(data_receipt)
r['identity']['source']['rl02_admission_status']='admitted_corrected_train_data'
r['checkpoint_reserve_seconds']=750
r['preparation_status']='Root data admitted; launch subject to full preflight, current lease and host guard'
put(W/'recipe.json',r)
sys.path.insert(0,str(source/'experiments/training'))
from campaign_rl_entry import preflight_entry
v=preflight_entry(r)
put(W/'root-full-preflight.json',v)
command=json.loads((C/'launch-command.json').read_text())['argv'];command[command.index(str(C/'fresh5.recipe.json'))]=str(W/'recipe.json')
put(W/'command.json',command)
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','candidate_files':len(checks),'frozen_source_files':len(sm['files']),'recipe_sha256':sha(W/'recipe.json'),'identity_sha256':hashlib.sha256(json.dumps(r['identity'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'full_preflight_sha256':sha(W/'root-full-preflight.json'),'parent_weights_verified':True,'framework_imports':{k:k in sys.modules for k in ['torch','transformers','trl','unsloth']},'guard_sha256':sha(guard),'reserve_seconds':750,'max_attempt_seconds':1800,'max_guard_seconds':1860,'data_receipt_sha256':sha(data_receipt),'command_sha256':sha(W/'command.json')}
put(W/'root-verification.json',result);print(json.dumps(result))
