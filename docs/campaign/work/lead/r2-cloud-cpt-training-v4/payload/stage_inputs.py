"""Credential-scoped download, byte verification, and recipe relocation."""
import argparse, json, os, subprocess
from pathlib import Path
from cloud_contract import REPO,RECIPE_SHA,TRANSPORT_SHA,require,sha,write

HERE=Path(__file__).resolve().parent

def replace(value,mapping):
 if isinstance(value,dict):return {k:replace(v,mapping) for k,v in value.items()}
 if isinstance(value,list):return [replace(v,mapping) for v in value]
 if isinstance(value,str):return mapping.get(value,value)
 return value

ALLOWED_ORIGINAL_ROOTS=(Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb'),Path('/home/m0hawk/.local/state/sepalith/campaign-20260915'),Path('/mnt/e/sepalith/campaign-20260915'))

def materialize_exact(downloaded,original,allowed_roots=ALLOWED_ORIGINAL_ROOTS,runner=subprocess.run):
 target=Path(original);require(target.is_absolute() and any(target.is_relative_to(root) for root in allowed_roots),'original path outside frozen roots')
 if target.exists():require(target.is_file() and not target.is_symlink() and sha(target)==sha(downloaded),'preexisting original path differs');return target
 runner(['sudo','install','-D','-m','0444',str(downloaded),str(target)],check=True,capture_output=True)
 require(target.is_file() and not target.is_symlink() and sha(target)==sha(downloaded),'exact path materialization differs')
 return target

def relocate(recipe,mapping,root,binding,deadline,max_attempt):
 r=replace(recipe,mapping);old_model=recipe['model_path']
 r['model_path']=mapping.get(old_model,old_model)
 r['output_dir']=str(root/'artifacts/training');r['archive_dir']=str(root/'artifacts/archive')
 r['deadline']=deadline;r['max_attempt_seconds']=max_attempt;r['termination_grace_seconds']=60
 r['checkpoint_reserve_seconds']=300;r['launch_authorized']=True
 require(r['stage']=='cpt_raw_r_v1' and r['parameters']['max_steps']==1902,'CPT horizon differs')
 require(r['parameters']['per_device_batch']*r['parameters']['gradient_accumulation']==16,'effective batch differs')
 require(r['identity']['policy']['initialization']=='new_lora_on_merged_cpt_parent','CPT initialization differs')
 require(Path(r['model_path']).is_absolute(),'model path is not absolute')
 resume=binding.get('relocated_resume')
 if resume:
  r['resume_from']=resume['path'];r['resume_binding']=resume['binding']
 else:r['resume_from']=None;r.pop('resume_binding',None)
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('binding',type=Path);p.add_argument('run',type=Path);p.add_argument('--deadline',required=True);p.add_argument('--max-attempt',required=True,type=float);a=p.parse_args()
 b=json.loads(a.binding.read_text());manifest=HERE.parent/'transport-manifest.json'
 require(sha(manifest)==TRANSPORT_SHA,'transport manifest differs')
 token=os.environ.get('HF_TOKEN');require(bool(token),'private input token absent')
 from huggingface_hub import HfApi,hf_hub_download
 api=HfApi(token=token);require(api.repo_info(REPO,repo_type='model').private is True,'input repository is not private')
 spec=json.loads(manifest.read_text());require(spec['recipe_sha256']==RECIPE_SHA and spec['repo_id']==REPO,'transport identity differs')
 mapping={};recipe_path=None;checks=[]
 for row in spec['files']:
  downloaded=Path(hf_hub_download(REPO,filename=row['remote_path'],revision=b['input_revision'],repo_type='model',token=token,local_dir=a.run/'input'))
  require(downloaded.stat().st_size==row['bytes'] and sha(downloaded)==row['sha256'],'downloaded input differs')
  checks.append({'relative_path':row['relative_path'],'bytes':row['bytes'],'sha256':row['sha256']})
  if row['role']=='recipe':recipe_path=downloaded
  elif row['role']=='recipe_input':mapping[row['original_path']]=str(materialize_exact(downloaded,row['original_path']))
 require(recipe_path is not None and sha(recipe_path)==RECIPE_SHA,'frozen recipe absent')
 recipe=json.loads(recipe_path.read_text())
 resume=b.get('resume')
 if resume:
  inventory_path=Path(hf_hub_download(REPO,filename=resume['inventory_path'],revision=resume['revision'],repo_type='model',token=token,local_dir=a.run/'resume-inventory'))
  require(sha(inventory_path)==resume['inventory_sha256'],'resume inventory differs')
  inv=json.loads(inventory_path.read_text());require(inv['schema']=='sepalith.cloud-cpt.remote-checkpoint.v1' and inv['step']==resume['step'],'resume step identity differs')
  require(inv['schedule_sha256']==recipe['draw_schedule']['sha256'] and inv['consumed_draws']==resume['step']*16,'resume sampler differs')
  resume_root=a.run/f"resume-checkpoint-{resume['step']}";resume_root.mkdir()
  for name,row in inv['files'].items():
   q=Path(hf_hub_download(REPO,filename=inv['prefix']+'/'+name,revision=resume['revision'],repo_type='model',token=token,local_dir=a.run/'resume-download'))
   require(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'resume checkpoint file differs')
   target=resume_root/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(q,target)
  require(sha(resume_root/'campaign-manifest.json')==inv['campaign_manifest_sha256'],'resume campaign manifest differs')
  b['relocated_resume']={'path':str(resume_root),'binding':{'schema':'sepalith.cpt.full-cadence-resume.v1','reason':'external_interruption','checkpoint_step':resume['step'],'checkpoint_manifest_sha256':inv['campaign_manifest_sha256'],'schedule_sha256':inv['schedule_sha256'],'consumed_draws':inv['consumed_draws']}}
 relocated=relocate(recipe,mapping,a.run,b,a.deadline,a.max_attempt)
 destination=a.run/'relocated-recipe.json';write(destination,relocated)
 write(a.run/'artifacts/staging-receipt.json',{'status':'private_inputs_staged_exact_original_paths_and_verified','revision':b['input_revision'],'files':checks,'credential_persisted':False,'path_mode':'exact immutable original paths reconstructed with sudo install; no source or parent-manifest rewrite','relocated_recipe':str(destination)})
 print(destination)
if __name__=='__main__':main()
