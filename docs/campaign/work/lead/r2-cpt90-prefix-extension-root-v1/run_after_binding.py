import os,json,time,hashlib,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
start=time.monotonic()
while not (p/'bound-recipe.json').exists():
 if time.monotonic()-start>3600:raise SystemExit('prefix binder did not publish within one hour; no launch')
 time.sleep(2)
a=p/'root-admission.json'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='d6f9664f538bdd025f15d997937680d0ccc313b34790a6962b5bdf052300dd91'
r=json.loads((p/'bound-recipe.json').read_text())
assert r['root_admission']['sha256']=='d6f9664f538bdd025f15d997937680d0ccc313b34790a6962b5bdf052300dd91'
assert r['transition']['source_global_step']==90 and r['transition']['source_cursor']==384 and r['transition']['global_optimizer_step_offset']==66
assert r['cohort']['unique_rows']==185318 and r['cohort']['documents']==179188 and r['cohort']['updates']==11583
assert r['runtime']['micro_batch']==1 and r['runtime']['gradient_accumulation']==16 and r['runtime']['mandatory_stop_step']==194
assert r['runtime']['checkpoint_every']==128
assert r['status']=='root_admitted_not_launched' and r['launch_authorized'] is True
# bind() publishes only after full source conservation, payload hashes, physical
# prefix, schedule joins, and cohort checks. Training rechecks them in-process.
proof={'status':'prefix_binder_published_all_required_checks_passed','at':time.time(),'bound_recipe_sha256':hashlib.sha256((p/'bound-recipe.json').read_bytes()).hexdigest(),'admission_sha256':'d6f9664f538bdd025f15d997937680d0ccc313b34790a6962b5bdf052300dd91','source_step':90,'source_cursor':384,'next_milestone':194,'packing':False}
(p/'binder-acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
guard=p.parent/'host-memory-guard-v4/cuda_host_guard.py'
assert hashlib.sha256(guard.read_bytes()).hexdigest()=='25f85d1da5bfee8da085a209982617e39e176b89dda11febc851c248a75240e0'
env={**os.environ,**json.loads((p/'environment.json').read_text())}
args=['python3',str(guard),'--command-json',str(p/'command.json'),'--output','/mnt/e/sepalith/campaign-20260915/training/SFT11-cpt90-prefix-extension-root-v1/guard-v1','--seconds','10800','--admission-free-mib','14336']
child=subprocess.Popen(args,env=env)
(p/'guard-start.json').write_text(json.dumps({'guard_pid':child.pid,'at':time.time(),'command':args},indent=2)+'\n')
raise SystemExit(child.wait())
