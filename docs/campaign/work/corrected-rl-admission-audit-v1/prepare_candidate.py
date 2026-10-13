"""Materialize a fresh corrected RL data candidate. Never load a model or train."""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PLAN = OUT.parents[1]
ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
recipe_path = ROOT / 'work/main-rl-preparation/primary-mb4-full5-e.recipe.json'
old = json.loads(recipe_path.read_text())
source = Path(old['identity']['source']['frozen_source_root'])
sys.path.insert(0, str(source / 'experiments/training'))
from campaign_rl_data import load_training_records, sha256_file
from campaign_rl_train import preflight_rl_recipe, CampaignRepeatSampler, source_draw_sequence_sha256
from campaign_rl_entry import _validate_entry_contract, _validate_schedule

for path_key, hash_key in [('rows_path', 'rows_sha256'), ('sidecar_path', 'sidecar_artifact_sha256'),
                           ('selected_ids_path', 'selected_ids_sha256'), ('source_draw_schedule_path', 'source_draw_schedule_sha256')]:
    assert sha256_file(Path(old['data'][path_key])) == old['data'][hash_key]
assert sha256_file(source / 'experiments/training/campaign_rl_train.py') == old['identity']['source']['trainer_sha256']
assert sha256_file(source / 'packages/sepalith/src/sepalith/campaign_protocol.py') == old['identity']['source']['protocol_sha256']

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return sha256_file(path)

data = OUT / 'candidate-data'
data.mkdir(exist_ok=False)
corrected_path = ROOT / 'work/lead/finish-corrected-train-v1/train-token-rows.jsonl'
corrected_manifest_path = ROOT / 'work/lead/finish-corrected-train-v1/manifest.json'
corrected_manifest = json.loads(corrected_manifest_path.read_text())
assert sha256_file(corrected_path) == corrected_manifest['output']['sha256']
old_order = json.loads(Path(old['data']['selected_ids_path']).read_text())['row_ids']
old_set = set(old_order)
keep = set()
rows_path = data / 'eligible-train-rows.jsonl'
with corrected_path.open('rb') as inp, rows_path.open('xb') as out:
    for line in inp:
        row = json.loads(line)
        if row['id'] in old_set:
            assert row['id'] not in keep
            keep.add(row['id'])
            out.write(line)
assert len(keep) == 8246 and len(old_set - keep) == 194
selected = [i for i in old_order if i in keep]
selected_path = data / 'selected-train-ids.json'
write(selected_path, {'schema_version': 'sepalith.prm07.selected-train-ids.v1', 'split': 'train', 'row_ids': selected})
sidecar_path = data / 'context-sidecar.jsonl'
with Path(old['data']['sidecar_path']).open('rb') as inp, sidecar_path.open('xb') as out:
    for line in inp:
        if json.loads(line)['row_id'] in keep:
            out.write(line)
records, manifest = load_training_records(rows_path, sha256_file(rows_path), sidecar_path, sha256_file(sidecar_path), selected_path, sha256_file(selected_path))

pool_path = EXEC / 'experiments/training/campaign_rl_pool.py'
module_spec = importlib.util.spec_from_file_location('reviewed_pool_allocator', pool_path)
pool = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = pool
module_spec.loader.exec_module(pool)
metadata, exclusions = pool.classify_records(records, approved_candidate_files=sorted({r.source_identity['candidate_file'] for r in records}))
assert not exclusions and len(metadata) == 8246
old_schedule = json.loads(Path(old['data']['source_draw_schedule_path']).read_text())
by_id = {r.row_id: r for r in metadata}
family_targets = {'finish_block': 3825, 'format_propagation': 6000, 'na_rm_propagation': 375,
                  'no_op': 4800, 'pipe_rewrite': 2400, 'rename_propagation': 6000, 'roxygen_drafting': 600}
capacities = {r.row_id: pool._row_capacity(r, small_pack_cap=3, ordinary_replay_cap=8) for r in metadata}
exposures, source_exposures, order = pool._allocate_row_exposures(metadata, family_targets, capacities, seed=3407, small_pack_cap=3, ordinary_replay_cap=8)
assert len(order) == 24000 and set(order) <= keep
assert dict(Counter(by_id[i].family for i in order)) == family_targets
assert all(n <= capacities[i] for i, n in exposures.items())
assert order == pool._allocate_row_exposures(metadata, family_targets, capacities, seed=3407, small_pack_cap=3, ordinary_replay_cap=8)[2]

quota_path = data / 'allocation-policy.json'
quota_sha = write(quota_path, {'schema': 1, 'status': 'candidate_pending_root_admission', 'policy': 'preserve accepted explicit family exposure; rebuild finite source rotation and integer-deficit interleave over corrected valid pool',
    'family_targets': family_targets, 'seed': 3407, 'small_pack_cap': 3, 'ordinary_replay_cap': 8,
    'candidate_count': 4, 'per_device_batch': 4, 'gradient_accumulation_steps': 8, 'buffer_reuse': 8,
    'allocator': {'path': str(pool_path), 'sha256': sha256_file(pool_path), 'function': '_allocate_row_exposures'},
    'parent_schedule_sha256': sha256_file(Path(old['data']['source_draw_schedule_path'])),
    'changed_target_authority': sha256_file(corrected_manifest_path), 'exposures': exposures, 'source_exposures': source_exposures})
schedule_path = data / 'source-row-draw-sequence.json'
schedule = copy.deepcopy(old_schedule)
schedule.update(status='candidate_corrected_pool_root_review_required', row_ids=order,
                selected_ids_sha256=manifest.selected_ids_sha256, ordered_ids_sha256=manifest.ordered_ids_sha256,
                row_identity_sha256=manifest.row_identity_sha256, sequence_sha256=source_draw_sequence_sha256(order),
                allocation_policy_sha256=quota_sha,
                allocation_policy_canonical_sha256=hashlib.sha256(json.dumps(json.loads(quota_path.read_text()), sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
schedule.pop('quota_override_sha256', None)
schedule.pop('quota_override_canonical_sha256', None)
schedule['ordering'] = {'policy': 'integer_largest_current_deficit_v1', 'seed': 3407,
                        'parent_schedule_sha256': sha256_file(Path(old['data']['source_draw_schedule_path'])),
                        'within_family_row_order': 'existing finite source rotation over corrected eligible rows',
                        'exact_old_draw_multiset_preserved': False, 'family_exposure_preserved': True,
                        'no_ad_hoc_tail_padding': True}
schedule_sha = write(schedule_path, schedule)

prep_path = OUT / 'data-preparation.json'
prep_sha = write(prep_path, {'schema': 1, 'status': 'candidate_cpu_verified_root_admission_pending',
    'launch_admitted': False, 'rows': len(records), 'excluded_old_ids': 194, 'changed_finish_targets': 2222,
    'data_identity': manifest.to_identity(), 'source_schedule_sha256': schedule_sha,
    'source_sequence_sha256': schedule['sequence_sha256'], 'source_draws': len(order),
    'corrected_train_manifest_sha256': sha256_file(corrected_manifest_path),
    'script_sha256': sha256_file(Path(__file__)), 'allocator_sha256': sha256_file(pool_path),
    'decision': 'Root must admit the new data/source/recipe identity before launch.'})

candidate = copy.deepcopy(old)
candidate.update(id='corrected-train-grpo-theta0-fresh5-v1', resume_from=None, max_steps=5,
                 evaluation_steps=[5], decision_steps=[5], max_attempt_seconds=1800,
                 checkpoint_reserve_seconds=600, termination_grace_seconds=60,
                 deadline='2026-09-13T22:00:00Z')
native_root = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-corrected-theta0-fresh5-v1')
candidate.update(output_dir=str(native_root / 'output'), archive_root=str(native_root / 'archive'), telemetry_path=str(native_root / 'telemetry.jsonl'))
candidate['data'].update(rows_path=str(rows_path), rows_sha256=manifest.rows_sha256,
    selected_ids_path=str(selected_path), selected_ids_sha256=manifest.selected_ids_sha256,
    sidecar_path=str(sidecar_path), sidecar_artifact_sha256=manifest.sidecar_artifact_sha256, context_sha256=manifest.context_sha256,
    source_draw_schedule_path=str(schedule_path), source_draw_schedule_sha256=schedule_sha,
    source_draw_sequence_sha256=schedule['sequence_sha256'], source_draws=24000)
candidate['identity']['data'] = manifest.to_identity()
candidate['identity']['schedule'].update(stage_id='corrected-train-grpo-theta0-fresh5-v1', source_draw_schedule_sha256=schedule_sha, source_draw_sequence_sha256=schedule['sequence_sha256'])
dev_path = ROOT / 'work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
dev_sha = sha256_file(dev_path)
candidate['development_panel'] = {'path': str(dev_path), 'sha256': dev_sha}
candidate['main_development_panel'].update(path=str(dev_path), sha256=dev_sha)
candidate['identity']['schedule']['development_panel_sha256'] = dev_sha
candidate['rl02_admission'] = {'path': str(prep_path), 'sha256': prep_sha, 'status': 'candidate_cpu_verified_root_admission_pending'}
candidate['identity']['source'].update(source_schedule_sha256=schedule_sha, rl02_admission_receipt_sha256=prep_sha,
    rl02_admission_status='candidate_cpu_verified_root_admission_pending', corrected_train_manifest_sha256=sha256_file(corrected_manifest_path),
    corrected_dev_manifest_sha256=sha256_file(ROOT / 'work/lead/corrected-dev75-v1/manifest.json'), data_adapter_sha256=sha256_file(Path(__file__)), pool_allocator_sha256=sha256_file(pool_path))
candidate['preparation_status'] = 'CPU candidate; root launch admission required'
candidate_path = OUT / 'fresh5.recipe.json'
write(candidate_path, candidate)
packet = preflight_rl_recipe(candidate)
_validate_entry_contract(candidate, packet['identity'])
_validate_schedule(candidate, packet['geometry'])
assert candidate['identity']['parent'] == old['identity']['parent']
assert candidate['identity']['policy'] == old['identity']['policy']
assert candidate['identity']['tokenizer'] == old['identity']['tokenizer']
assert candidate['identity']['renderer'] == old['identity']['renderer']
for key in ('learning_rate', 'warmup_steps', 'generation_kwargs', 'cuda_memory_fraction', 'model_load_max_seq_length', 'generation_groups_per_call'):
    assert candidate[key] == old[key]

prefixes = {}
for steps in (5, 25):
    ids = order[:steps * 8]
    tokens = sum(by_id[i].completion_tokens for i in ids)
    noop_tokens = sum(by_id[i].completion_tokens for i in ids if by_id[i].family == 'no_op')
    prefixes[str(steps)] = {'draws': len(ids), 'unique_rows': len(set(ids)), 'families': dict(Counter(by_id[i].family for i in ids)),
        'reference_target_plus_eos_tokens': tokens, 'noop_reference_tokens': noop_tokens, 'noop_reference_token_fraction': noop_tokens / tokens,
        'source_cursor': steps * 8, 'sampler_consumed_rows': steps * 256, 'generated_candidates': steps * 32}

write(OUT / 'candidate-cpu-preflight.json', {'status': 'CPU_data_policy_development_schedule_pass; root model/source/resource admission pending',
    'framework_imports': packet['framework_imports'], 'rows': packet['records'], 'geometry': packet['geometry'], 'prefixes': prefixes,
    'source_draws': len(order), 'unique_drawn_rows': len(set(order)), 'family_targets': family_targets,
    'repeat_allocation_deterministic': True, 'all_exposures_within_caps': True, 'identical_parent_policy_tokenizer_renderer': True,
    'old_resume_removed': True, 'parent_weights_read_or_hashed': False, 'full_entry_preflight_not_run': 'It verifies parent weight hashes; root-only future check.',
    'candidate_recipe_sha256': sha256_file(candidate_path), 'data_preparation_sha256': prep_sha,
    'next': 'Root independent review and source/data/budget admission; full entry preflight; guard 1860s before launch.'})
print(json.dumps({'rows': len(records), 'draws': len(order), 'prefixes': prefixes, 'recipe_sha256': sha256_file(candidate_path), 'framework_imports': packet['framework_imports']}, indent=2))
