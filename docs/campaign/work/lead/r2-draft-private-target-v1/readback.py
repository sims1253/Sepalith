"""Independently stream the committed private target and verify all bytes."""
import datetime,hashlib,json,pathlib,time
import requests
from huggingface_hub import HfApi,get_token,hf_hub_url
W=pathlib.Path(__file__).resolve().parent

def main():
 r=json.loads((W/'upload-receipt.json').read_text());token=get_token();api=HfApi(token=token)
 assert api.repo_info(r['repo'],repo_type='model',revision=r['commit']).private is True
 checked=[];started=time.monotonic()
 with requests.Session() as session:
  for pin in r['files']:
   url=hf_hub_url(r['repo'],r['prefix']+'/'+pin['path'],revision=r['commit'])
   h=hashlib.sha256();size=0
   with session.get(url,headers={'Authorization':'Bearer '+token},stream=True,timeout=(15,45)) as response:
    response.raise_for_status()
    for chunk in response.iter_content(1024*1024):
     if time.monotonic()-started>1200:raise TimeoutError('Independent readback deadline exceeded')
     h.update(chunk);size+=len(chunk)
   assert size==pin['bytes'] and h.hexdigest()==pin['sha256'],pin['path']
   checked.append(pin);print('verified '+pin['path'],flush=True)
 out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'private_target_commit_stream_readback_verified','repo':r['repo'],'prefix':r['prefix'],'commit':r['commit'],'verified_files':checked,'seconds':time.monotonic()-started,'local_model_copy_created':False}
 (W/'readback-receipt.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
