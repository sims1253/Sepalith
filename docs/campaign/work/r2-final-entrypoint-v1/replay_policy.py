"""CPU-only fresh-process replay of recorded entrypoint origin policies."""
import subprocess,sys,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
results=[]
for route,entry,envname,preload in [('constructor','construct_final.py','.venv','runtime.preload();g.gate._load_protocol()'),('native-client','evaluate_final.py','.venv-sft','preload()')]:
 path=str(HERE/entry)
 code="import sys,importlib.machinery\n__file__="+repr(path)+"\n__cached__=None\n__loader__=importlib.machinery.SourceFileLoader('__main__',__file__)\n"
 code+="exec(compile(open(__file__).read().rsplit(\"if __name__=='__main__':raise SystemExit(main())\",1)[0],__file__,'exec'))\n"
 code+=preload+"\ngraph=json.loads((HERE/"+repr(route+'-source-closure.admitted.json')+").read_text())\nprint(json.dumps(policy.verify(graph,deep=True)));print(json.dumps(g.binding.verify_source_closure(graph,g.binding.digest_json(graph))))\n"
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','PYTHON','GGML_'))}
 env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 r=subprocess.run(['/home/m0hawk/Documents/Sepalith/'+envname+'/bin/python','-I','-S','-B','-c',code],env=env,capture_output=True,text=True,timeout=60)
 results.append(dict(route=route,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
(HERE/'root-policy-replay.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
raise SystemExit(int(any(x['exit'] for x in results)))
