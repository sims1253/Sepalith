#!/usr/bin/env python3
"""Prepare, but never admit or launch, the native checkpoint-194 continuation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PACKET = Path(__file__).resolve().parent
LEAD = PLAN / 'docs/campaign/work/lead'
RECIPE = LEAD / 'r2-cpt90-prefix-extension-root-v2/bound-recipe.json'
CHECKPOINT = Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-v1/full/checkpoint-194')
STAGE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/native-staging/cpt-prefix-extension-reusable-data-v1.receipt.json')
DECISION = PLAN / 'docs/campaign/receipts/SFT-11-cpt194-root-decision.json'
REVIEW = LEAD / 'r2-cpt194-review-root-v1/checkpoint-review.json'

PINS = {
    'recipe': '579269f0f203ae1eed9cdb96af0f535f19c5953e2cf3e19c52615b7545e10f70',
    'checkpoint_manifest': '2ad012580f0a7dccfdfd7f546ad5431893ee56c3df2e4adf01ddc4bab8e4b072',
    'stage_receipt': 'a72c1b699fe931e0db0fa2ddfc96a43486e74fc962802e8a541540e16596b88d',
    'decision': 'f2c3710658fcc4fd9035296bd8b695720acb3dd8b2e4ea5a4e82394af8e612a1',
    'review': 'f170b6778fa5d791be842d222108ca4634828412e6874ebd2f22ff52a2360caa',
}
TRAINER_ROOT = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-CPT-prefix-extension-v3-from194/runtime')
ARCHIVE_ROOT = Path('/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-full-weight-CPT-prefix-extension-native-v3-from194')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def inspect(source_manifest: Path):
    for name, path in [('recipe', RECIPE), ('stage_receipt', STAGE), ('decision', DECISION), ('review', REVIEW)]:
        require(sha(path) == PINS[name], f'{name} differs')
    manifest_path = CHECKPOINT / 'campaign-manifest.json'
    require(sha(manifest_path) == PINS['checkpoint_manifest'], 'checkpoint194 manifest differs')
    recipe, manifest, state, stage, decision, review = map(read, (RECIPE, manifest_path, CHECKPOINT/'campaign-state.json', STAGE, DECISION, REVIEW))
    require(manifest.get('full') is True and manifest.get('checkpoint_kind') == 'full_weights' and manifest.get('step') == 194, 'checkpoint194 is not complete full state')
    require(manifest['identity']['schedule']['global_optimizer_step_offset'] == 66, 'checkpoint offset differs')
    require(manifest['identity']['schedule']['checkpoint_every'] == 128, 'checkpoint cadence differs')
    sampler = state['sampler']
    require(sampler['global_step'] == 194 and sampler['cursor'] == sampler['stage_cursor'] == 2048, 'checkpoint cursor differs')
    require(sampler['global_optimizer_step_offset'] == 66 and sampler['draw_schedule_sha256'] == recipe['cohort']['draw_schedule']['sha256'], 'sampler identity differs')
    require(manifest['identity']['policy']['optimizer'] == recipe['runtime']['optimizer'], 'optimizer identity differs')
    required = {'model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','campaign-state.json','tokenizer.json'}
    require(required <= set(manifest['files']), 'full state inventory incomplete')
    require(review == {'checkpoint_manifest_sha256': PINS['checkpoint_manifest'], 'files_verified': 12, 'step': 194, 'cursor': 2048, 'in_process_anchor_nll': 0.9838079802365687, 'anchor_ids_and_denominators_match_checkpoint90': True, 'cuda_launched': False}, 'root checkpoint review differs')
    require(decision['checkpoint_manifest_sha256'] == PINS['checkpoint_manifest'] and decision['editing_release_promotion'] is False and decision['final_eval_opened'] is False and 'Continue' in decision['decision'], 'root continuation decision differs')
    require(stage['status'] == 'staged_immutable_verified' and stage['total_bytes'] == 9330242806 and len(stage['objects']) == 3, 'native staged data differs')
    require({x['name'] for x in stage['objects']} == {'cache','rows','schedule'}, 'native staged object roles differ')
    runtime_source = read(source_manifest)
    require(runtime_source.get('schema') == 'sepalith.sft11.native-cpt-trainer-source.v2', 'runtime source schema differs')
    checkpoint_bytes = sum(item['bytes'] for item in manifest['files'].values())
    require(checkpoint_bytes == 17250948075, 'checkpoint byte inventory differs')
    native_two_checkpoint_budget = stage['total_bytes'] + 2 * (18 * 1024**3)
    free_after_two = stage['native_free_after_bytes'] - 2 * (18 * 1024**3)
    require(native_two_checkpoint_budget <= 70 * 1024**3 and free_after_two >= 70 * 1024**3, 'native two-checkpoint capacity floor fails')
    return recipe, manifest, stage, runtime_source, checkpoint_bytes, native_two_checkpoint_budget, free_after_two


def prepare(source_manifest: Path, output: Path):
    recipe, manifest, stage, runtime_source, checkpoint_bytes, budget, free_after_two = inspect(source_manifest)
    common = {'status':'preparation_only','launch_authorized':False}
    relocation = {
        'schema':'sepalith.sft11.native-relocation-admission.v1', **common,
        'bound_recipe_sha256':PINS['recipe'], 'stage_receipt_sha256':PINS['stage_receipt'],
        'object_roles':{'streaming_cache':'cache','rows':'rows','draw_schedule':'schedule'},
        'checkpoint194_remains_on_durable_e':True,
    }
    migration = {
        'schema':'sepalith.sft11.native-runtime-source-migration-admission.v1', **common,
        'canonical_bound_recipe_sha256':PINS['recipe'], 'scientific_source_manifest_sha256':recipe['source']['manifest_sha256'],
        'runtime_source_manifest_sha256':sha(source_manifest), 'stage_receipt_sha256':PINS['stage_receipt'],
        'optimizer_scheduler_rng_sampler_identity_must_remain_unchanged':True,
        'allowed_changes':['v2 source-schema front-door correction','native immutable input paths','native-hot two-checkpoint transient window and durable-E publication','root-admitted execution stop at global step 322'],
    }
    continuation = {
        'schema':'sepalith.sft11.cpt-stage-transition-continuation-admission.v1', **common,
        'decision':'root_decision_supports_continuation_but_admission_pending', 'bound_recipe_sha256':'ROOT_BIND_RUNTIME_RECIPE_SHA256',
        'checkpoint':str(CHECKPOINT), 'checkpoint_manifest_sha256':PINS['checkpoint_manifest'], 'step':194,
        'root_decision':{'path':str(DECISION),'sha256':PINS['decision']},
    }
    stop = {
        'schema':'sepalith.sft11.native-cpt-execution-stop.v1', **common,
        'bound_recipe_sha256':'ROOT_BIND_RUNTIME_RECIPE_SHA256', 'resume_global_step':194, 'stop_at_global_step':322,
        'stage_local_resume_step':128, 'stage_local_stop_step':256, 'expected_cursor_at_stop':4096,
    }
    capacity = {
        'schema':'sepalith.sft11.native-cpt194-capacity.v1','status':'pass_preparation_only',
        'staged_data_bytes':stage['total_bytes'],'checkpoint194_payload_bytes':checkpoint_bytes,
        'expected_full_checkpoint_ceiling_bytes':18*1024**3,'transient_native_checkpoint_slots':2,
        'native_usage_ceiling_bytes':70*1024**3,'calculated_staged_plus_two_checkpoint_ceiling_bytes':budget,
        'native_free_after_staging_bytes':stage['native_free_after_bytes'],'calculated_free_after_two_checkpoint_ceiling_bytes':free_after_two,
        'minimum_free_floor_bytes':70*1024**3,'capacity_pass':True,
    }
    paths = {
        'schema':'sepalith.sft11.native-cpt194-paths.v1','status':'prepared_fresh_paths',
        'canonical_recipe':str(RECIPE),'resume_checkpoint':str(CHECKPOINT),'native_staged_data_receipt':str(STAGE),
        'trainer_root':str(TRAINER_ROOT),'durable_archive_root':str(ARCHIVE_ROOT),
        'protect_existing_archive':str(Path(recipe['outputs']['archive'])),'protect_existing_run_result':str(Path(recipe['outputs']['archive'])/'run-result.json'),
        'fresh_paths_must_not_exist_before_launch':True,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('relocation-admission.prepared.json',relocation),('runtime-source-migration-admission.prepared.json',migration),('continuation-admission-194.prepared.json',continuation),('execution-stop-322.prepared.json',stop),('capacity-review.json',capacity),('paths.json',paths)]:
        atomic_json(output/name,value)
    result={'schema':'sepalith.sft11.native-cpt194-preparation-result.v1','status':'prepared_no_admission_no_launch','checkpoint_step':194,'resume_cursor':2048,'stop_global_step':322,'stop_cursor':4096,'source_manifest_sha256':sha(source_manifest),'prepared_files':sorted(p.name for p in output.iterdir())}
    atomic_json(output/'preparation-result.json',result)
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source_manifest,args.output),sort_keys=True))
