#!/usr/bin/env python3
"""Offline export reproducibility audit. Never replaces banked artifacts."""
import hashlib,json,os,subprocess,time
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path('/home/m0hawk/Documents/Sepalith/experiments')
    out=Path('/mnt/h/sepalith/runs/b4-quant-reproduction-20260908')
    out.mkdir(exist_ok=False)
    parent=root/'models/packaging_b4-f16.gguf'
    matrix=root/'models/quant-calibration/b4-smoke-imatrix.gguf'
    receipt=json.loads(matrix.with_suffix('.gguf.json').read_text())
    binary=root/'bin/llama/llama-b10453/llama-quantize'
    inputs={str(p):digest(p) for p in (parent,matrix,binary)}
    if inputs[str(parent)]!=receipt['model_sha256'] or inputs[str(matrix)]!=receipt['imatrix_sha256']:
        raise ValueError('Input receipt mismatch')
    results=[]
    env=dict(os.environ,LD_LIBRARY_PATH=str(binary.parent))
    for name,imatrix in [('control',False),('imatrix',True)]:
        target=out/(name+'.gguf')
        original=root/('models/packaging_b4_'+name+'-Q4_K_M.gguf')
        expected=digest(original)
        argv=[str(binary),'--output-tensor-type','q8_0','--token-embedding-type','q8_0']
        if imatrix:argv+=['--imatrix',str(matrix)]
        argv += [str(parent),str(target),'Q4_K_M','8']
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        actual=digest(target)
        if digest(original)!=expected:raise ValueError('Banked export changed during audit')
        results.append(dict(arm=name,argv=argv,seconds=time.monotonic()-start,expected=expected,actual=actual,byte_identical=actual==expected))
        print(json.dumps(results[-1]),flush=True)
    if any(digest(Path(p))!=h for p,h in inputs.items()):raise ValueError('Input changed during audit')
    record=dict(inputs=inputs,results=results,scope='Deterministic export identity; does not establish training lineage or release calibration quality.')
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps([dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())],indent=2)+'\n')
if __name__=='__main__':main()
