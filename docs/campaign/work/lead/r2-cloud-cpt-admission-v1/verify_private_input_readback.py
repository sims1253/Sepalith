#!/usr/bin/env python3
"""Independently download and hash the committed private CPT input prefix."""

import datetime,hashlib,json,os
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO='scholzmx/sepalith-lora'
REVISION='1e4df9c9db3cee1a7df5bbcc7caa62920d2ab2db'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)

def main():
 token=os.environ.get('HF_TOKEN');require(bool(token),'private readback token absent')
 import httpx
 from huggingface_hub import HfApi,hf_hub_url
 from huggingface_hub.utils import build_hf_headers
 manifest=json.loads((HERE/'transport-manifest.json').read_text());prefix=manifest['prefix']
 api=HfApi(token=token);info=api.repo_info(REPO,repo_type='model',revision=REVISION)
 require(info.private is True and info.sha==REVISION,'private immutable revision differs')
 expected={row['remote_path']:{'bytes':row['bytes'],'sha256':row['sha256']} for row in manifest['files']}
 manifest_remote=prefix+'/transport-manifest.json';expected[manifest_remote]={'bytes':(HERE/'transport-manifest.json').stat().st_size,'sha256':sha(HERE/'transport-manifest.json')}
 tree=list(api.list_repo_tree(REPO,repo_type='model',path_in_repo=prefix,revision=REVISION,recursive=True,expand=True))
 remote_paths={row.path for row in tree if hasattr(row,'size')}
 require(remote_paths==set(expected),'immutable prefix inventory differs')
 verified=[];total=0;headers=build_hf_headers(token=token)
 with httpx.Client(follow_redirects=True,timeout=None,headers=headers) as client:
  for number,(name,want) in enumerate(expected.items(),1):
   digest=hashlib.sha256();size=0;url=hf_hub_url(REPO,filename=name,revision=REVISION,repo_type='model')
   with client.stream('GET',url) as response:
    response.raise_for_status()
    for block in response.iter_bytes(chunk_size=4*1024*1024):digest.update(block);size+=len(block)
   actual={'bytes':size,'sha256':digest.hexdigest()};require(actual==want,f'remote byte readback differs: {name}')
   verified.append({'path':name,**actual});total+=actual['bytes'];print(json.dumps({'verified':number,'of':len(expected),'bytes_total':total}),flush=True)
 receipt={'schema':'sepalith.cloud-cpt.private-input-readback.v1','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'immutable_revision_all_bytes_streamed_and_hashed','repo':REPO,'private':True,'revision':REVISION,'prefix':prefix,'files':len(verified),'bytes':total,'transport_manifest_sha256':sha(HERE/'transport-manifest.json'),'records':verified,'credential_persisted':False,'local_cache_or_payload_copy_created':False}
 (HERE/'private-input-readback.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'files':len(verified),'bytes':total}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  print(json.dumps({'status':'readback_failed','error_type':type(error).__name__}),flush=True);raise SystemExit(1)
