#!/usr/bin/env python3
"""Create an admitted physical-path overlay without changing training identity.

This does not authorize training.  The resulting recipe is accepted only by a
future trainer source that explicitly verifies ``storage_relocation`` and
uses canonical paths for source-decision comparisons.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os
from pathlib import Path
import tempfile
from sepalith.training.paths import under_checkpoint_root

def require(v,m):
    if not v:raise ValueError(m)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def identity_view(recipe):
    return {
      'parent':{'candidate_id':recipe['parent']['candidate_id'],'weights_sha256':recipe['parent']['files']['model.safetensors'],'saved_precision':recipe['parent']['saved_precision'],'source_checkpoint_manifest_sha256':recipe['transition']['source_checkpoint']['manifest_sha256']},
      'tokenizer':{'sha256':recipe['parent']['files']['tokenizer.json'],'bos':0,'eos_pad':1},
      'renderer':{'kind':'pretokenized_raw_r_cpt_v1','max_sequence_tokens':recipe['cohort']['max_sequence_tokens']},
      'data':{'cohort_id':recipe['cohort']['id'],'rows_sha256':recipe['cohort']['rows']['sha256'],'streaming_cache_manifest_sha256':recipe['cohort']['streaming_cache']['manifest_sha256']},
      'source':{'manifest_sha256':recipe['source']['manifest_sha256']},
      'policy':{'stage':'full_weight_cpt_stage_transition_v1','optimizer':recipe['runtime']['optimizer']},
      'schedule':{**{k:recipe['runtime'][k] for k in ('max_steps','effective_batch','micro_batch','gradient_accumulation','learning_rate','scheduler','warmup_steps','checkpoint_every','mandatory_stop_step')},'global_optimizer_step_offset':recipe['transition']['global_optimizer_step_offset'],'destination_updates':recipe['cohort']['updates']},
    }

def target(obj):
    root=Path(obj['staged_path'])
    if obj['kind']=='manifest_tree':return root
    require(obj['kind']=='file' and len(obj['files'])==1,'file staging record differs')
    return root/next(iter(obj['files']))

def make_overlay(recipe_path,receipt_path,receipt_sha256,admission_path,output_path,*,runtime_source_manifest,migration_admission,trainer_root,archive_root):
    recipe_path=Path(recipe_path);receipt_path=Path(receipt_path);admission_path=Path(admission_path)
    recipe=json.loads(recipe_path.read_text());original=copy.deepcopy(recipe)
    require(digest(receipt_path)==receipt_sha256,'stage receipt differs')
    receipt=json.loads(receipt_path.read_text());admission=json.loads(admission_path.read_text())
    require(admission.get('schema')=='sepalith.sft11.native-relocation-admission.v1' and admission.get('status')=='admitted','native relocation is not root-admitted')
    require(admission.get('bound_recipe_sha256')==digest(recipe_path),'relocation admission refers to another recipe')
    require(admission.get('stage_receipt_sha256')==receipt_sha256,'relocation admission refers to another stage')
    by_name={obj['name']:obj for obj in receipt['objects']}
    expected={'streaming_cache':'cache','rows':'rows','draw_schedule':'schedule'}
    roles=admission.get('object_roles');require(roles==expected or roles=={'source_checkpoint':'checkpoint',**expected},'relocation object roles differ')
    if 'source_checkpoint' in roles:expected={'source_checkpoint':'checkpoint',**expected}
    objects={role:by_name[name] for role,name in expected.items()}
    canonical={role:str(Path(obj['canonical_source']).resolve()) for role,obj in objects.items()}
    if 'source_checkpoint' in canonical:
        require(str(Path(recipe['transition']['source_checkpoint']['path']).resolve())==canonical['source_checkpoint'],'canonical transition checkpoint differs')
        require(str(Path(recipe['parent']['path']).resolve())==canonical['source_checkpoint'],'canonical parent checkpoint differs')
    require(str(Path(recipe['cohort']['streaming_cache']['path']).resolve())==canonical['streaming_cache'],'canonical cache differs')
    require(str(Path(recipe['cohort']['rows']['path']).resolve())==canonical['rows'],'canonical rows differ')
    require(str(Path(recipe['cohort']['draw_schedule']['path']).resolve())==canonical['draw_schedule'],'canonical schedule differs')
    if 'source_checkpoint' in objects:
        recipe['parent']['path']=str(target(objects['source_checkpoint']))
        recipe['transition']['source_checkpoint']['path']=str(target(objects['source_checkpoint']))
    recipe['cohort']['streaming_cache']['path']=str(target(objects['streaming_cache']))
    recipe['cohort']['rows']['path']=str(target(objects['rows']))
    recipe['cohort']['draw_schedule']['path']=str(target(objects['draw_schedule']))
    require(identity_view(recipe)==identity_view(original),'physical relocation changed training identity')
    recipe['storage_relocation']={'schema':'sepalith.sft11.native-runtime-overlay.v1','canonical_bound_recipe':str(recipe_path.resolve()),'canonical_bound_recipe_sha256':digest(recipe_path),'stage_receipt':str(receipt_path.resolve()),'stage_receipt_sha256':receipt_sha256,'admission':str(admission_path.resolve()),'admission_sha256':digest(admission_path),'canonical_paths':canonical,'bundle_id':receipt['bundle_id']}
    runtime_source_manifest=Path(runtime_source_manifest);migration_admission=Path(migration_admission)
    source_manifest=json.loads(runtime_source_manifest.read_text());require(source_manifest.get('schema')=='sepalith.sft11.native-cpt-trainer-source.v2','native runtime source schema differs')
    recipe['runtime_source']={'manifest_path':str(runtime_source_manifest.resolve()),'manifest_sha256':digest(runtime_source_manifest)}
    recipe['runtime_source_migration']={'admission':str(migration_admission.resolve()),'admission_sha256':digest(migration_admission)}
    trainer_root=Path(trainer_root);archive_root=Path(archive_root)
    require(str(trainer_root).startswith('/home/m0hawk/.local/state/sepalith/campaign-20260915/') and under_checkpoint_root(archive_root),'native/E output roots differ')
    recipe['outputs']['trainer']=str(trainer_root);recipe['outputs']['archive']=str(archive_root)
    recipe['checkpoint_storage']={'mode':'native_hot_to_e_durable_atomic','c_hot_stage':'root_admitted','required_same_filesystem':False,'trainer_root':str(trainer_root),'archive_root':str(archive_root),'native_capacity_root':'/home/m0hawk/.local/state/sepalith/campaign-20260915','expected_full_checkpoint_bytes':18*1024**3,'trainer_transient_save_total_limit':2,'native_retained_after_durable_publication':1}
    recipe['retention']['trainer_save_total_limit']=1
    require(identity_view(recipe)==identity_view(original),'runtime/storage overlay changed training identity')
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{output_path.name}.',dir=output_path.parent)
    with os.fdopen(fd,'w') as f:json.dump(recipe,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,output_path)
    return {'schema':'sepalith.sft11.native-runtime-overlay-result.v1','status':'prepared_not_launch_authorized','runtime_recipe':str(output_path.resolve()),'runtime_recipe_sha256':digest(output_path),'identity_unchanged':True,'bundle_id':receipt['bundle_id']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--stage-receipt',required=True);p.add_argument('--stage-receipt-sha256',required=True);p.add_argument('--admission',required=True);p.add_argument('--runtime-source-manifest',required=True);p.add_argument('--migration-admission',required=True);p.add_argument('--trainer-root',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(make_overlay(a.recipe,a.stage_receipt,a.stage_receipt_sha256,a.admission,a.output,runtime_source_manifest=a.runtime_source_manifest,migration_admission=a.migration_admission,trainer_root=a.trainer_root,archive_root=a.archive_root),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
