#!/usr/bin/env python3
"""Bind a cadence64 checkpoint450 continuation after root evidence acceptance."""
import argparse,copy,hashlib,importlib.util,json,os,tempfile
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');PACKET=Path(__file__).resolve().parent
SOURCE_RECIPE=PLAN/'docs/campaign/work/lead/r2-cpt354-to450-continuation-root-v1/runtime-recipe.json';SOURCE_RECIPE_SHA='26b78532f907ca5e3a34827857ad58a47e0d56a167fb26a61dbf9b555b50a1d5';SOURCE_IDENTITY='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
SOURCE_MANIFEST=PACKET/'source-manifest.json';SOURCE_MANIFEST_SHA='e9e05e8b21fd6b05e54bef66fb6555408a0b3d3fc64e52c2b41ed9a4a869e590'
RESUME=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-recovery354-to450-cadence24-v1/runtime/checkpoint-450')
TRAINER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-cpt450-cadence64-to706-v1/runtime');ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-cpt450-cadence64-to706-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def accepted(path,expected_sha):
 path=Path(path);assert sha(path)==expected_sha;v=json.loads(path.read_text());assert v.get('status')=='accepted' and v.get('checkpoint_step')==450;return v
def build_recipe(source,migration_path,migration_sha):
 recipe=copy.deepcopy(source);recipe['runtime']['checkpoint_every']=64;recipe['runtime']['mandatory_stop_step']=706
 for name in ('evaluation_steps','selected_milestones'):recipe['runtime'][name]=sorted(set(recipe['runtime'][name]+[706]))
 recipe['outputs'].update(trainer=str(TRAINER),archive=str(ARCHIVE),graceful_stop='/home/m0hawk/.local/state/sepalith/campaign-20260915/control/SFT11-cpt450-cadence64-to706-v1-save-stop.json');recipe['checkpoint_storage'].update(trainer_root=str(TRAINER),archive_root=str(ARCHIVE));recipe['runtime_source']={'manifest_path':str(SOURCE_MANIFEST),'manifest_sha256':SOURCE_MANIFEST_SHA};recipe['runtime_source_migration']={'admission':str(Path(migration_path).resolve()),'admission_sha256':migration_sha}
 return recipe
def prepare(args):
 assert sha(SOURCE_RECIPE)==SOURCE_RECIPE_SHA and sha(SOURCE_MANIFEST)==SOURCE_MANIFEST_SHA
 accepted(args.checkpoint_review,args.checkpoint_review_sha256);accepted(args.matched_review,args.matched_review_sha256)
 assert len(args.checkpoint_manifest_sha256)==64 and args.checkpoint_manifest_sha256!='ROOT_BINDS_VERIFIED_CHECKPOINT450_MANIFEST'
 source=json.loads(SOURCE_RECIPE.read_text());recipe=build_recipe(source,args.migration_admission,args.migration_admission_sha256)
 spec=importlib.util.spec_from_file_location('trainer',PACKET/'source/experiments/training/full_weight_cpt_trainer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 old,new=m.identity(source),m.identity(recipe);assert canonical(old)==SOURCE_IDENTITY
 contract=__import__('sys');contract.path.insert(0,str(PACKET/'source/experiments/training'));from native_runtime_contract import validate_cadence64_identity
 migration=json.loads(Path(args.migration_admission).read_text());assert sha(args.migration_admission)==args.migration_admission_sha256
 destination=validate_cadence64_identity(source,recipe,m.identity,migration)
 write(args.output,recipe);recipe_sha=sha(args.output)
 continuation={'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'ROOT_MUST_SET_admitted','decision':'continue','launch_authorized':False,'bound_recipe_sha256':recipe_sha,'checkpoint':str(RESUME),'checkpoint_manifest_sha256':args.checkpoint_manifest_sha256,'step':450}
 stop={'schema':'sepalith.sft11.native-cpt-execution-stop.v1','status':'ROOT_MUST_SET_admitted','launch_authorized':False,'bound_recipe_sha256':recipe_sha,'resume_global_step':450,'stop_at_global_step':706}
 output_dir=Path(args.output).resolve().parent;write(output_dir/'continuation-450.template.json',continuation);write(output_dir/'execution-stop-706.template.json',stop)
 return {'status':'prepared_root_admissions_pending','recipe_sha256':recipe_sha,'source_identity_sha256':SOURCE_IDENTITY,'destination_identity_sha256':destination,'resume_step':450,'resume_cursor':6144,'stop_step':706,'stop_cursor':10240,'updates':256,'draws':4096,'scheduled_saves':[514,578,642,706]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint-review',required=True);p.add_argument('--checkpoint-review-sha256',required=True);p.add_argument('--matched-review',required=True);p.add_argument('--matched-review-sha256',required=True);p.add_argument('--checkpoint-manifest-sha256',required=True);p.add_argument('--migration-admission',required=True);p.add_argument('--migration-admission-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a),sort_keys=True))
