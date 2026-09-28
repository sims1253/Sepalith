import json,pathlib,hashlib,datetime
from huggingface_hub import HfApi,hf_hub_download
w=pathlib.Path(__file__).parent;p=json.loads((w/'persistence-provider.json').read_text());api=HfApi();base=p['prefix']+'/artifacts/';info=api.repo_info(p['repo'],revision=p['commit']);assert info.private
remote={x.path[len(base):]:x for x in api.list_repo_tree(p['repo'],path_in_repo=base.rstrip('/'),revision=p['commit'],recursive=True,expand=True) if hasattr(x,'size')}
checks=[]
for name,pin in p['files'].items():
 x=remote[name];lfs=getattr(x,'lfs',None);actual=getattr(lfs,'sha256',None) if lfs else x.blob_id;expected=pin['sha256'] if lfs else pin['git_blob_sha1'];assert x.size==pin['bytes'] and actual==expected,name
 checks.append(name)
selected=['upload-manifest.json','entry-failure.json','local-terminal.json','supervision.json','training.log','training.log.guard.json','training/cloud-backward-profile.json','training/target-only-pre-update-gate.json','training/telemetry.jsonl','archive/full/checkpoint-250/campaign-manifest.json','archive/full/checkpoint-250/campaign-state.json','archive/full/checkpoint-250/trainer_state.json','recipe.json','binding.json']
read=[]
for name in selected:
 f=pathlib.Path(hf_hub_download(p['repo'],filename=base+name,revision=p['commit'],local_dir=w/'readback'));assert hashlib.sha256(f.read_bytes()).hexdigest()==p['files'][name]['sha256'];read.append(name)
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'private':info.private,'commit':info.sha,'remote_metadata_verified_count':len(checks),'independent_byte_readback_verified':read,'readback_root':str(w/'readback'/base)};(w/'verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
