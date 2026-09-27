"""Fresh isolated-process origin/closure negatives, with synthetic mutations only."""
import json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENTRY=str(HERE/'assemble_final.py')
cases=[
 ('unexpected-module',"sys.modules['unadmitted_injected_module']=type(sys)('unadmitted_injected_module')",'unexpected module'),
 ('module-path',"assembly.__file__='/synthetic/unexpected.py'",'module origin drift'),
 ('search-root',"sys.path.append('/synthetic/unadmitted')",'search roots changed'),
 ('factory-origin',"sys.modules['_collections_abc'].__spec__.origin='synthetic-drift'",'module factory/alias drift'),
 ('executable-map',"graph['runtime_policy']['executable_paths'].append('/synthetic/unobserved.so')",'client executable map changed'),
 ('factory-binding',"graph['runtime_policy']['modules']['assemble_inputs']['bindings']=[]",'factory binding missing'),
]
results=[]
env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','PYTHON','GGML_'))}
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name,change,expected in cases:
    code="import sys,importlib.machinery\n__file__="+repr(ENTRY)+"\n__cached__=None\n__loader__=importlib.machinery.SourceFileLoader('__main__',__file__)\n"
    code+="exec(compile(open(__file__).read().rsplit(\"if __name__=='__main__':raise SystemExit(main())\",1)[0],__file__,'exec'))\n"
    code+="make_parser();preload()\ngraph=json.loads((HERE/'assembly-source-closure.prepared.json').read_text())\npolicy.verify(graph,deep=True)\n"+change+'\n'
    code+="try: policy.verify(graph,deep=True)\nexcept ValueError as error:\n print(str(error))\n assert "+repr(expected)+" in str(error)\nelse: raise AssertionError('mutation accepted')\n"
    run=subprocess.run(['/home/m0hawk/Documents/Sepalith/.venv/bin/python','-I','-S','-B','-c',code],env=env,capture_output=True,text=True,timeout=30)
    results.append(dict(case=name,exit=run.returncode,stdout=run.stdout,stderr=run.stderr))
(HERE/'origin-negative-tests.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results));raise SystemExit(int(any(r['exit'] for r in results)))
