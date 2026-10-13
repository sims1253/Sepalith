"""CPU-only recovery verification. Copies frozen small sources into owned test space."""
import ast, contextlib, hashlib, importlib.util, io, json, os, pathlib, shutil, sys, unittest
sys.dont_write_bytecode=True
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
H=pathlib.Path(__file__).resolve().parent
W=H.parent
C=W/'r2-cloud-bootstrap-recovery-v1'
V=W/'r2-cloud-bootstrap-recovery-v2'
P=W/'lead/r2-cloud-control-preparation/payload'
R=W.parent/'receipts'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
pins={R/'PAR-R2-cloud-bootstrap-recovery.json':'491edffee78a906d23ed41dbd3ec6dc6e5688c56f0d0288583710983a124ed09',C/'artifact-manifest.json':'8561dcc5c034db0c2eaf0461f1afad303c8d9a8af912ea4a62efbca19df6f616',V/'deadline_watchdog.py':'0e1a2d8a02cd4ed5807dc9cc0ddcc90d64e5c4dda4b9d28a211b9ed67ca65721',V/'root_arm_watchdog.py':'056926f4eaf5994812cb90b3396f6daf73b4fdc9f136c7ddfeead32f787a0c7a',V/'manifest.json':'9fd226f2737045aa37bc9998c407608309e5bc029960be462f542578dc37829a',R/'PAR-R2-cloud-bootstrap-recovery-v2.json':'cd4314c6dd0cfa70e47e8f2b4aa44008c9dfb3cbc0084b3cf58e427d7ed0371e'}
for p,s in pins.items():assert digest(p)==s,str(p)
rows=json.loads((C/'artifact-manifest.json').read_text())['files']
for row in rows:
 p=C/row['path'];assert digest(p)==row['sha256'] and p.stat().st_size==row['bytes'],str(p)
core=json.loads((C/'trainer-source-manifest.json').read_text())['files']
equal=[]
for row in core:
 p=C/row['path'];original=P/row['path'];assert p.read_bytes()==original.read_bytes();equal.append(record(p))
assert len(equal)==11
wrappers=['cloud_launch.py','cloud_train.py','runtime_setup.py','artifact_upload.py','requirements.txt','package-pins.json','trainer-source-manifest.json']
for name in wrappers:assert (C/name).read_bytes()==(P/name).read_bytes(),name
# Exact installed resolver source is metadata/source only, without SDK initialization.
e=json.loads((C/'installed-cli-evidence.json').read_text());installed=pathlib.Path(e['source_path']);assert digest(installed)==e['source_sha256']
full=ast.parse(installed.read_text());extracted=ast.parse((C/'installed-job-resolver.py').read_text());wanted={r['name'] for r in e['exact_extracted_methods']}
for n in wanted:
 a=next(x for x in ast.walk(full) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==n)
 b=next(x for x in ast.walk(extracted) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==n)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),n
accepted=W/'r2-cloud-control-independent-review-v1/deadline_watchdog_delayed_submission.py'
assert digest(accepted)=='16d3e01bdb0f73bfa92574804a3af7feaad1e722bd87782a1a61c408c5b29514'
for n in ['monitor','validate','main']:
 def fn(p):return next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==n)
 assert ast.dump(fn(accepted),include_attributes=False)==ast.dump(fn(V/'deadline_watchdog.py'),include_attributes=False)
F=H/'replay-inputs';F.mkdir(exist_ok=True)
for row in rows:
 q=pathlib.Path(row['path'])
 if q.suffix in ('.py','.sh') or q.name=='trainer-source-manifest.json':
  dst=F/q;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C/q,dst)
# V2 guard is paired with unchanged V1 bootstrap entry/observer dependencies.
for name in ['deadline_watchdog.py','root_arm_watchdog.py']:shutil.copyfile(V/name,F/name)
sys.path.insert(0,str(F));c=load('cloud_entry',F/'cloud_entry.py');w=load('deadline_watchdog',F/'deadline_watchdog.py');arm=load('root_arm_watchdog',F/'root_arm_watchdog.py')
modules=[load(n,F/(n+'.py')) for n in ['test_cloud_entry','test_recovery','test_watchdog_recovery']]
suite=unittest.TestSuite()
for m in modules:
 for group in unittest.defaultTestLoader.loadTestsFromModule(m):
  for test in group:
   if 'real_torch_backward_profile' not in test.id():suite.addTest(test)
# Reuse accepted delayed-submission, no-job and post-ID failure tests against V2.
old=load('accepted_delayed_tests',W/'r2-cloud-control-independent-review-v1/test_watchdog_delayed_submission.py');old.w=w
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(old))
# The source test module imports an old guard under another name; use V2 explicitly.
with (H/'replay-tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
summary={'pins':[record(p) for p in pins],'artifact_files_verified':len(rows),'core_files_byte_identical':equal,'additional_unchanged_files':[record(C/n) for n in wrappers],'installed_resolver_methods_ast_equal':sorted(wanted),'v2_accepted_monitor_validate_main_ast_equal':True,'replayed_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped_test':'unchanged real_torch_backward_profile: model/framework import excluded from this bounded review; previously accepted CPU evidence retained','source_only_no_provider_calls':True,'affinity':sorted(os.sched_getaffinity(0))}
(H/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ('artifact_files_verified','replayed_tests','failures','errors','affinity')}))
raise SystemExit(not result.wasSuccessful())
