#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

CAPS=(192,384,768)

def sha(p, chunk=1024*1024):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--cap',type=int,choices=CAPS,required=True)
    ap.add_argument('--full-model-hash',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); d=json.loads(a.binding.read_text())
    assert d['status']=='prepared_candidate_binding_no_launch_no_release'
    assert d['selection']['primary']=='E750_Q8' and d['selection']['rollback']=='b4_Q8'
    assert d['release_constraints']['sealed_final_access'] is False
    files=d['candidate']['artifacts']
    checks=[]
    for key in ('tokenizer_json','tokenizer_config','runtime_server','vsix'):
        x=files[key]; p=Path(x['path']); st=p.stat()
        if st.st_size != x['bytes']: raise SystemExit(f'{key}: size changed')
        got=sha(p)
        if got != x['sha256']: raise SystemExit(f'{key}: hash changed')
        checks.append({'key':key,'path':str(p),'bytes':st.st_size,'sha256':got})
    m=files['model']; st=Path(m['path']).stat()
    if st.st_size != m['bytes']: raise SystemExit('model size changed')
    model={'key':'model','path':m['path'],'bytes':st.st_size,'expected_sha256':m['sha256'],'full_hash_performed':a.full_model_hash}
    if a.full_model_hash:
        model['sha256']=sha(Path(m['path']))
        if model['sha256'] != m['sha256']: raise SystemExit('model hash changed')
    checks.append(model)
    cap=next(x for x in d['output_cap_candidates'] if x['max_output_tokens']==a.cap)
    if a.cap==192 and cap['existing_accepted_vsix_compatible'] is not True: raise SystemExit('192 must retain accepted VSIX compatibility')
    if a.cap!=192 and cap['existing_accepted_vsix_compatible'] is not False: raise SystemExit('larger cap must remain adapter-gated')
    result={'schema':'sepalith.run06.e750-v1-binding-preflight.v1','status':'passed_preparation_only','cap':a.cap,'checks':checks,'launch_authorized':False,'release_claim':False}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
