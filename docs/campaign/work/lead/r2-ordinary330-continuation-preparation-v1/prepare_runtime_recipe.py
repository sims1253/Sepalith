#!/usr/bin/env python3
"""Prepare a fresh runtime recipe without changing scientific training identity."""
import argparse,copy,hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');PACKET=Path(__file__).resolve().parent
ORIGINAL=PLAN/'docs/campaign/work/lead/r2-native-cpt194-root-admission-v1/runtime-recipe.json';ORIGINAL_SHA='076b9105b88ce125c406d52db37865e782be217824e79f3e33b9ff0fec27fef1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def prepare(source_manifest,migration_admission,trainer_root,archive_root,output):
 need(sha(ORIGINAL)==ORIGINAL_SHA,'original runtime recipe differs');source_manifest=Path(source_manifest);migration_admission=Path(migration_admission)
 source_sha=sha(source_manifest);sm=json.loads(source_manifest.read_text());need(sm.get('schema')=='sepalith.sft11.native-cpt-trainer-source.v2','new runtime source schema differs')
 ma=json.loads(migration_admission.read_text());need(ma.get('schema')=='sepalith.sft11.native-runtime-source-migration-admission.v1' and ma.get('status')=='admitted' and ma.get('launch_authorized')is True,'runtime source migration not admitted');need(ma.get('runtime_source_manifest_sha256')==source_sha and ma.get('scientific_source_manifest_sha256')=='8dd7ecf617e37e1778dabee6721857e1b79074e0314f17e0e3f55b57a809a98c' and ma.get('canonical_bound_recipe_sha256')=='579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70' and ma.get('stage_receipt_sha256')=='a72c1b699fe931e0db0fa2ddfc96a43486e74fc962802e8a541540e16596b88d','runtime source migration binding differs')
 original=json.loads(ORIGINAL.read_text());value=copy.deepcopy(original);value['runtime_source']={'manifest_path':str(source_manifest.resolve()),'manifest_sha256':source_sha};value['runtime_source_migration']={'admission':str(migration_admission.resolve()),'admission_sha256':sha(migration_admission)}
 trainer_root=Path(trainer_root);archive_root=Path(archive_root);need(str(trainer_root).startswith('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/') and str(archive_root).startswith('/mnt/e/sepalith/campaign-20260915/checkpoints/'),'fresh output roots differ');value['outputs']['trainer']=str(trainer_root);value['outputs']['archive']=str(archive_root);value['outputs']['graceful_stop']='/home/m0hawk/.local/state/sepalith/campaign-20260915/control/'+trainer_root.name+'-save-stop.json';value['checkpoint_storage']['trainer_root']=str(trainer_root);value['checkpoint_storage']['archive_root']=str(archive_root)
 sys.path.insert(0,str(PACKET/'source/experiments/training'));spec=importlib.util.spec_from_file_location('continuation_trainer',PACKET/'source/experiments/training/full_weight_cpt_trainer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);need(m.identity(original)==m.identity(value),'runtime recipe changed production identity')
 write(output,value);return {'status':'prepared_root_admitted_source_only_not_launch_authorized','runtime_recipe':str(Path(output).resolve()),'runtime_recipe_sha256':sha(output),'production_identity_sha256':hashlib.sha256(m.canonical(m.identity(value))).hexdigest(),'source_manifest_sha256':source_sha}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True);p.add_argument('--runtime-source-migration-admission',required=True);p.add_argument('--trainer-root',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a.source_manifest,a.runtime_source_migration_admission,a.trainer_root,a.archive_root,a.output),sort_keys=True))
if __name__=='__main__':main()
