#!/usr/bin/env python3
"""Prepare same-identity checkpoint354 to checkpoint450 continuation."""
import copy,hashlib,importlib.util,json,os,tempfile
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');PACKET=Path(__file__).resolve().parent
BASE=PLAN/'docs/campaign/work/lead/r2-selected330-recovery-root-v2/runtime-recipe.json';BASE_SHA='d74c3da8a254eaee572ef80ee12b186d592e91fd1479fae008a60ff84908e3d7'
SOURCE=PLAN/'docs/campaign/work/lead/r2-selected330-recovery-preparation-v2/source/experiments/training';SOURCE_MANIFEST_SHA='d17b49bf5beb26aa6308fba1e1a6fda26b6163e42d636bd6200c55556e3d0f1a'
RESUME=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-selected330-recovery-cadence24-v2/runtime/checkpoint-354')
TRAINER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-recovery354-to450-cadence24-v1/runtime')
ARCHIVE=Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-recovery354-to450-cadence24-v1')
ROOT=PLAN/'docs/campaign/work/lead/r2-cpt354-to450-continuation-root-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 with os.fdopen(fd,'w')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def prepare(output=PACKET/'runtime-recipe.review.json'):
 assert sha(BASE)==BASE_SHA;recipe=copy.deepcopy(json.loads(BASE.read_text()));assert recipe['runtime_source']['manifest_sha256']==SOURCE_MANIFEST_SHA
 recipe['outputs'].update(trainer=str(TRAINER),archive=str(ARCHIVE),graceful_stop='/home/m0hawk/.local/state/sepalith/campaign-20260915/control/SFT11-recovery354-to450-cadence24-v1-save-stop.json')
 recipe['checkpoint_storage'].update(trainer_root=str(TRAINER),archive_root=str(ARCHIVE))
 spec=importlib.util.spec_from_file_location('trainer',SOURCE/'full_weight_cpt_trainer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 old,new=m.identity(json.loads(BASE.read_text())),m.identity(recipe);assert old==new and canonical(new)=='46a5a1aced9f533e8ba4cd25aebe795dedc64d390a0d563cfc7e406c03324739'
 output=Path(output);write(output,recipe);recipe_sha=sha(output)
 continuation={'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1','status':'ROOT_MUST_SET_admitted','decision':'continue','launch_authorized':False,'bound_recipe_sha256':recipe_sha,'checkpoint':str(RESUME),'checkpoint_manifest_sha256':'ROOT_BINDS_VERIFIED_CHECKPOINT354_MANIFEST','step':354}
 stop={'schema':'sepalith.sft11.native-cpt-execution-stop.v1','status':'ROOT_MUST_SET_admitted','launch_authorized':False,'bound_recipe_sha256':recipe_sha,'resume_global_step':354,'stop_at_global_step':450}
 write(PACKET/'continuation-354.template.json',continuation);write(PACKET/'execution-stop-450.template.json',stop)
 return {'schema':'sepalith.sft11.cpt354-to450-preparation.v1','status':'prepared_metric_review_and_admissions_pending','recipe_sha256':recipe_sha,'identity_sha256':canonical(new),'resume':str(RESUME),'resume_step':354,'resume_cursor':4608,'stop_step':450,'stop_cursor':6144,'updates':96,'draws':1536,'fresh_trainer':str(TRAINER),'fresh_archive':str(ARCHIVE),'ordinary_transition_argument_required':False}
if __name__=='__main__':print(json.dumps(prepare(),sort_keys=True))
