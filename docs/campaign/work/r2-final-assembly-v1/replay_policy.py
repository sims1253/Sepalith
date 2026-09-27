"""Fresh-process CPU policy replay. Never writes an admitted graph or opens final inputs."""
import json, os, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENTRY=str(HERE/'assemble_final.py')
results=[]
for route,actions in [('production-preload','make_parser();preload()'),('synthetic-full','make_parser();synthetic_work()')]:
    code="import sys,importlib.machinery\n__file__="+repr(ENTRY)+"\n__cached__=None\n__loader__=importlib.machinery.SourceFileLoader('__main__',__file__)\n"
    code+="exec(compile(open(__file__).read().rsplit(\"if __name__=='__main__':raise SystemExit(main())\",1)[0],__file__,'exec'))\n"
    code+=actions+"\ngraph=json.loads((HERE/'assembly-source-closure.prepared.json').read_text())\n"
    code+="print(json.dumps(policy.verify(graph,deep=True)))\n"
    # In-memory structural verification only. The on-disk graph stays preparation_only.
    code+="candidate=dict(graph,status='root_admitted')\nprint(json.dumps(g.binding.verify_source_closure(candidate,g.binding.digest_json(candidate))))\n"
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','PYTHON','GGML_'))}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    r=subprocess.run(['/home/m0hawk/Documents/Sepalith/.venv/bin/python','-I','-S','-B','-c',code],env=env,capture_output=True,text=True,timeout=60)
    results.append(dict(route=route,exit=r.returncode,stdout=r.stdout,stderr=r.stderr,graph_admitted=False))
(HERE/'policy-replay.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results));raise SystemExit(int(any(x['exit'] for x in results)))
