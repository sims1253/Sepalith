"""Upload the pinned intermediate target to the authorized private model repo."""
import datetime,hashlib,json,pathlib
from huggingface_hub import HfApi,CommitOperationAdd,get_token
W=pathlib.Path(__file__).resolve().parent
MODEL=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-a-250-merged')
REPO='scholzmx/sepalith-lora'
PREFIX='r2-draft-target/b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4'
PINS={'model.safetensors':'b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4','config.json':'f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991','generation_config.json':'7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269','tokenizer.json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','tokenizer_config.json':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 api=HfApi(token=get_token());assert api.repo_info(REPO,repo_type='model').private is True
 assert not api.file_exists(REPO,PREFIX+'/target-manifest.json',repo_type='model'),'Prefix already exists; inspect before reuse'
 files=[]
 for name,pin in PINS.items():
  f=MODEL/name;assert f.is_file() and not f.is_symlink() and digest(f)==pin
  files.append({'path':name,'bytes':f.stat().st_size,'sha256':pin})
 manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'frozen_intermediate_draft_target_not_release','files':files,'training_sources_only':True}
 (W/'target-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+x['path'],path_or_fileobj=str(MODEL/x['path'])) for x in files]
 operations.append(CommitOperationAdd(path_in_repo=PREFIX+'/target-manifest.json',path_or_fileobj=str(W/'target-manifest.json')))
 commit=api.create_commit(REPO,repo_type='model',operations=operations,commit_message='Persist frozen intermediate target for bounded draft profile',num_threads=1)
 r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'private_target_uploaded_readback_pending','repo':REPO,'prefix':PREFIX,'commit':commit.oid,'files':files}
 (W/'upload-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
