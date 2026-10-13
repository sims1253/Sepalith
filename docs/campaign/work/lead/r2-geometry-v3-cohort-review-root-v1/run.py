import pathlib,json,hashlib,subprocess,shlex,os,datetime
R=pathlib.Path(__file__).resolve().parent;P=R.parent/'r2-expanded-union-geometry-recovery-preparation-v3';A=json.loads((R/'admission.json').read_text())
assert hashlib.sha256((P/'source-manifest.json').read_bytes()).hexdigest()==A['source_manifest_sha256']
for x in json.loads((P/'source-manifest.json').read_text())['files']:assert hashlib.sha256((P/x['path']).read_bytes()).hexdigest()==x['sha256']
def write(n,v):
 v['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with(R/n).open('x')as f:json.dump(v,f,indent=2);f.write('\n')
write('launch.json',{'controller_pid':os.getpid(),'cores':[6],'cohorts':[15006,10682]})
for key in ('original15006','semantic10682'):
 args=shlex.split(json.loads((P/'commands.json').read_text())[key]);env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 while args and '='in args[0]:k,v=args.pop(0).split('=',1);env[k]=v
 with(R/(key+'.log')).open('x')as f:c=subprocess.run(['timeout','--signal=TERM','--kill-after=20s','600','taskset','-c','6',*args],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 write(key+'-terminal.json',{'exit_code':c})
 if c:raise SystemExit(c)
write('terminal.json',{'exit_code':0,'status':'diagnostic_outputs_complete_requires_root_review','training_admitted':False})
