#!/usr/bin/env python3
"""Bind one preparation-only canary arm onto an admitted native runtime recipe."""
import argparse,copy,hashlib,json,os,tempfile
from pathlib import Path

DECISION=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/SFT-11-varlen-numerics-v2-root-decision.json')
OPPORTUNITY=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/receipts/SFT-11-packed-schedule-opportunity-root.json')
DECISION_SHA='cb5ed35d95304d719db6c1aec205785e7d6052750b6e69f0409b087c13d03474'
OPPORTUNITY_SHA='8d1ad42b9eb52a12441c114abda2e773bc8ce22435b2c2d6ef4fd658267b69db'
OVERLAY=Path('/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-overlay')
EXTENSION_SHA='bb2a59af5ed03aa28ea0e1ed705384fcb6d5d0e6ab8bd825b004ffbb3af9d669'

def require(v,m):
 if not v:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def ordinary_identity(recipe):
 return {'parent':{'candidate_id':recipe['parent']['candidate_id'],'weights_sha256':recipe['parent']['files']['model.safetensors'],'saved_precision':recipe['parent']['saved_precision'],'source_checkpoint_manifest_sha256':recipe['transition']['source_checkpoint']['manifest_sha256']},'tokenizer':{'sha256':recipe['parent']['files']['tokenizer.json'],'bos':0,'eos_pad':1},'renderer':{'kind':'pretokenized_raw_r_cpt_v1','max_sequence_tokens':recipe['cohort']['max_sequence_tokens']},'data':{'cohort_id':recipe['cohort']['id'],'rows_sha256':recipe['cohort']['rows']['sha256'],'streaming_cache_manifest_sha256':recipe['cohort']['streaming_cache']['manifest_sha256']},'source':{'manifest_sha256':recipe['source']['manifest_sha256']},'policy':{'stage':'full_weight_cpt_stage_transition_v1','optimizer':recipe['runtime']['optimizer']},'schedule':{**{k:recipe['runtime'][k] for k in ('max_steps','effective_batch','micro_batch','gradient_accumulation','learning_rate','scheduler','warmup_steps','checkpoint_every','mandatory_stop_step')},'global_optimizer_step_offset':recipe['transition']['global_optimizer_step_offset'],'destination_updates':recipe['cohort']['updates']}}
def prepare(recipe_path,resume,manifest_sha,arm,source_manifest,output,runtime_migration_admission=None,updates=8):
 require(arm in ('ordinary_reference','varlen_candidate'),'canary arm differs');require(type(updates)is int and updates in (1,8),'canary updates differ');require(sha(DECISION)==DECISION_SHA and sha(OPPORTUNITY)==OPPORTUNITY_SHA,'review evidence differs')
 recipe_path=Path(recipe_path);recipe=json.loads(recipe_path.read_text());resume=Path(resume);mp=resume/'campaign-manifest.json';require(mp.is_file()and sha(mp)==manifest_sha,'source checkpoint manifest differs')
 manifest=json.loads(mp.read_text());state=json.loads((resume/'campaign-state.json').read_text());require(manifest.get('identity')==ordinary_identity(recipe)and state.get('identity')==ordinary_identity(recipe),'source is not exact ordinary recipe identity')
 step=manifest['step'];offset=recipe['transition']['global_optimizer_step_offset'];cursor=state['sampler']['cursor'];require(cursor==(step-offset)*16,'source cursor differs')
 source_manifest=Path(source_manifest);sm=json.loads(source_manifest.read_text());require(sm.get('schema')=='sepalith.sft11.native-cpt-trainer-source.v2','canary source schema differs')
 extension=OVERLAY/'xformers/_C.so';require(extension.is_file()and sha(extension)==EXTENSION_SHA,'reviewed xformers extension differs')
 value=copy.deepcopy(recipe);value['runtime_source']={'manifest_path':str(source_manifest.resolve()),'manifest_sha256':sha(source_manifest)}
 if runtime_migration_admission is None:value['runtime_source_migration']={'admission':'ROOT_FRESH_CANARY_RUNTIME_MIGRATION_ADMISSION','admission_sha256':'ROOT_FILL_AFTER_ADMISSION'}
 else:
  migration=Path(runtime_migration_admission);require(migration.is_file(),'runtime migration admission missing');mv=json.loads(migration.read_text());require(mv.get('schema')=='sepalith.sft11.native-runtime-source-migration-admission.v1'and mv.get('status')=='admitted'and mv.get('launch_authorized')is True,'runtime migration is not root-admitted');require(mv.get('runtime_source_manifest_sha256')==sha(source_manifest),'runtime migration source differs');value['runtime_source_migration']={'admission':str(migration.resolve()),'admission_sha256':sha(migration)}
 value['varlen_canary']={'schema':'sepalith.sft11.native-varlen-canary.v1','status':'prepared_requires_root_admission','production_admitted':False,'arm':arm,'updates':updates,'logical_rows':16,'source_checkpoint':str(resume.resolve()),'source_checkpoint_manifest_sha256':manifest_sha,'source_global_step':step,'target_global_step':step+updates,'source_cursor':cursor,'target_cursor':cursor+16*updates,'physical_execution':{'ordinary_reference':{'trainer_gradient_accumulation':16,'physical_batches':16},'varlen_candidate':{'trainer_gradient_accumulation':1,'physical_packs':'derived within exact window'}}[arm],'logical_objective':{'membership':'exact next 16 consecutive frozen draws','loss_reduction':'global_supervised_token_mean','optimizer_steps':updates,'scheduler_steps':updates},'numerics_decision':{'path':str(DECISION.resolve()),'sha256':DECISION_SHA},'schedule_opportunity':{'path':str(OPPORTUNITY.resolve()),'sha256':OPPORTUNITY_SHA},'attention_backend':{'selected':'xformers','overlay_root':str(OVERLAY),'extension_sha256':EXTENSION_SHA}}
 stem=f'SFT11-native-varlen-canary-{step}-{arm}-v1';trainer=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training')/stem/'runtime';archive=Path('/mnt/e/sepalith/campaign-20260915/checkpoints')/stem
 value['outputs']['trainer']=str(trainer);value['outputs']['archive']=str(archive);value['outputs']['graceful_stop']=str(Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/control')/(stem+'-save-stop.json'))
 value['checkpoint_storage'].update(trainer_root=str(trainer),archive_root=str(archive));value['canary_root_admission']=None
 write(output,value)
 return {'status':'prepared_no_admission_no_launch','arm':arm,'updates':updates,'source_step':step,'target_step':step+updates,'source_cursor':cursor,'target_cursor':cursor+16*updates,'recipe_sha256':sha(output),'output':str(Path(output).resolve())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--native-runtime-recipe',required=True);p.add_argument('--resume',required=True);p.add_argument('--resume-manifest-sha256',required=True);p.add_argument('--arm',choices=('ordinary_reference','varlen_candidate'),required=True);p.add_argument('--updates',type=int,choices=(1,8),default=8);p.add_argument('--source-manifest',required=True);p.add_argument('--runtime-migration-admission');p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare(a.native_runtime_recipe,a.resume,a.resume_manifest_sha256,a.arm,a.source_manifest,a.output,a.runtime_migration_admission,a.updates),sort_keys=True))
if __name__=='__main__':main()
