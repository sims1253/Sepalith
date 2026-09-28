import pathlib,json,hashlib
from huggingface_hub import hf_hub_download
w=pathlib.Path(__file__).parent;p=json.loads((w/'persistence-provider.json').read_text());base=p['prefix']+'/artifacts/';names=[n for n in p['files'] if n.startswith('archive/full/checkpoint-250/')];done=[]
for n in names:
 f=pathlib.Path(hf_hub_download(p['repo'],filename=base+n,revision=p['commit'],local_dir=w/'readback'));h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==p['files'][n]['sha256'] and f.stat().st_size==p['files'][n]['bytes'],n
 done.append({'name':n,**p['files'][n]})
(w/'checkpoint-byte-readback.json').write_text(json.dumps({'commit':p['commit'],'files':done,'all_pass':True},indent=2)+'\n');print(json.dumps({'verified_files':len(done),'total_bytes':sum(x['bytes'] for x in done)}))
