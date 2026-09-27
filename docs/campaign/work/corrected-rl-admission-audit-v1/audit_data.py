"""Read only named TRAIN/DEV/source artifacts; emit aggregate audit facts."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.dont_write_bytecode = True
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
OUT = Path(__file__).resolve().parent
recipe_path = PLAN / 'work/main-rl-preparation/primary-mb4-full5-e.recipe.json'
recipe = json.loads(recipe_path.read_text())
source = Path(recipe['identity']['source']['frozen_source_root'])
sys.path.insert(0, str(source / 'experiments/training'))
from campaign_rl_data import _validate_row, _validate_sidecar_record, _validate_semantics
from campaign_rl_train import CampaignPRM03Reward, line_f1

pins = []
def rows(path):
    path = Path(path)
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for line in stream:
            h.update(line)
            count += 1
            yield json.loads(line)
    pins.append({'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size, 'rows': count})

def pin(path):
    path = Path(path)
    pins.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})

def compact(row):
    return {k: v for k, v in row.items() if k not in ('input_ids', 'target_body_tokens', 'target_terminal_tokens')}

old = {r['id']: compact(r) for r in rows(recipe['data']['rows_path'])}
corrected = {}
all_corrected = {}
validation = Counter()
for row in rows(PLAN / 'work/lead/finish-corrected-train-v1/train-token-rows.jsonl'):
    all_corrected[row['id']] = compact(row)
    if row['id'] in old:
        try:
            _validate_row(row)
            validation['corrected_protocol_and_prompt_pass'] += 1
        except ValueError as error:
            validation['corrected_protocol_or_prompt_fail'] += 1
        corrected[row['id']] = compact(row)
    else:
        validation['corrected_without_old_RL_context'] += 1

selected_path = Path(recipe['data']['selected_ids_path'])
selected = json.loads(selected_path.read_text())['row_ids']
pin(selected_path)
selected_set = set(selected)
assert len(selected) == len(selected_set) == len(old) == 8440
assert selected_set == set(old)
changed = {i for i in corrected if old[i]['target_body_text'] != corrected[i]['target_body_text']}
excluded = selected_set - set(corrected)
assert not any(old[i]['prompt_text'] != corrected[i]['prompt_text'] or old[i]['target_start'] != corrected[i]['target_start'] for i in corrected)
assert not any(old[i]['family'] != corrected[i]['family'] for i in corrected)

contexts = {}
groups = set()
packages = set()
for sidecar in rows(recipe['data']['sidecar_path']):
    i = sidecar['row_id']
    groups.add(sidecar['source_identity']['group_id'])
    packages.add(sidecar['package_id'])
    if i not in corrected:
        continue
    try:
        _, context, _, _, _ = _validate_sidecar_record(sidecar, corrected[i])
        _validate_semantics(corrected[i], context)
        validation['unchanged_sidecar_corrected_semantics_pass'] += 1
    except ValueError:
        validation['unchanged_sidecar_corrected_semantics_fail'] += 1
        raise
    if i in changed:
        contexts[i] = context

dev = list(rows(PLAN / 'work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'))
dev_groups = {r['group_id'] for r in dev}
dev_packages = {r['package_id'] for r in dev}

def stats(ids, table):
    items = [table[i] for i in ids]
    total_tokens = sum(r['target_token_count'] + 1 for r in items)
    noop = [r for r in items if r['target_operation'] == 'no_op']
    return {'draws': len(items), 'unique_rows': len(set(ids)), 'families': dict(Counter(r['family'] for r in items)),
            'no_op_source_draws': len(noop), 'no_op_source_fraction': len(noop) / len(items) if items else 0,
            'canonical_target_plus_eos_tokens': total_tokens,
            'no_op_target_plus_eos_tokens': sum(r['target_token_count'] + 1 for r in noop),
            'no_op_reference_token_fraction': sum(r['target_token_count'] + 1 for r in noop) / total_tokens if total_tokens else 0,
            'prompt_plus_bos_max': max((r['target_start'] for r in items), default=0),
            'prompt_plus_bos_over_2048': sum(r['target_start'] > 2048 for r in items),
            'target_plus_eos_max': max((r['target_token_count'] + 1 for r in items), default=0),
            'target_plus_eos_over_192': sum(r['target_token_count'] + 1 > 192 for r in items),
            'exact_target_total_over_2240': sum(r['target_start'] + r['target_token_count'] + 1 > 2240 for r in items)}

schedule_path = Path(recipe['data']['source_draw_schedule_path'])
schedule = json.loads(schedule_path.read_text())
pin(schedule_path)
order = schedule['row_ids']
retained = [i for i in order if i in corrected]
reachable = {i for i in corrected if corrected[i]['target_token_count'] + 1 <= 192}
reachable_order = [i for i in retained if i in reachable]
prefixes = {}
for label, sequence in [('original', order), ('retained_corrected_only', retained), ('retained_corrected_reachable_192', reachable_order)]:
    table = old if label == 'original' else corrected
    prefixes[label] = {}
    for step in (5, 25):
        ids = sequence[:step * 8]
        value = stats(ids, table)
        value['corrected_finish_draws'] = sum(i in changed for i in ids)
        value['excluded_under_correction'] = sum(i in excluded for i in ids)
        value['expected_source_cursor'] = step * 8
        value['completion_candidates'] = step * 8 * 4
        prefixes[label][str(step)] = value

overlay_count = Counter()
for overlay in rows(PLAN / 'work/finish-target-overlay-v1/finish-target-overlay.jsonl'):
    i = overlay['key']['original_id']
    if i not in selected_set:
        continue
    overlay_count['old_RL_overlap'] += 1
    overlay_count[overlay['status']] += 1
    if i in changed:
        assert overlay['original']['target_body_text'] == old[i]['target_body_text']
        assert overlay['repaired']['target_body_text'] == corrected[i]['target_body_text']
        if overlay['original']['provenance']['finish_splice'].get('outer_closing_brace_in_label') is False:
            overlay_count['old_RL_target_explicitly_omits_outer_brace'] += 1
        if overlay['repaired'].get('document_parse_ok') is True:
            overlay_count['prior_overlay_reports_repaired_full_document_parse_ok'] += 1
        assert corrected[i]['target_body_text'] == old[i]['target_body_text'] + '}'
        assert not contexts[i].suffix_lines
        overlay_count['corrected_target_exactly_appends_one_brace_and_context_has_no_suffix'] += 1
        # Apply the actual source geometry; do not substitute a label for source.
        provenance = overlay['original']['provenance']
        document = provenance['selection_source']['document_text']
        geometry = provenance['region_geometry']
        lines = document.split('\n')
        start = sum(len(s) + 1 for s in lines[:geometry['start_line']]) + geometry['start_codepoint']
        end = sum(len(s) + 1 for s in lines[:geometry['end_line']]) + geometry['end_codepoint']
        applied = document[:start] + corrected[i]['target_body_text'] + document[end:]
        assert hashlib.sha256(applied.encode()).hexdigest() == overlay['repaired']['document_sha256']
        overlay_count['full_applied_corrected_R_bytes_match_prior_parse_evidence'] += 1

reward_probes = []
for i in sorted(changed)[:3]:
    context = contexts[i]
    for target_kind, generated_kind in [('old', 'old'), ('corrected', 'old'), ('corrected', 'corrected')]:
        table = old if target_kind == 'old' else corrected
        generated = old[i] if generated_kind == 'old' else corrected[i]
        # Explicit decoder control probes real reward code, not model/tokenizer generation.
        reward = CampaignPRM03Reward(decoder=lambda ids, text=generated['target_text'], **kw: text)
        score, record = reward.score_one(context, 'replace', table[i]['target_body_text'], [100, 1])
        reward_probes.append({'target': target_kind, 'generated': generated_kind, 'score': score,
                              'exact_region': record['exact_region'], 'protocol_valid': record['protocol_valid'],
                              'scope': 'actual reward function with explicit decoder control; no model or tokenizer proof'})

for path in [recipe_path, PLAN / 'work/lead/finish-corrected-train-v1/manifest.json', PLAN / 'work/lead/corrected-dev75-v1/manifest.json', source / 'experiments/training/campaign_rl_train.py', source / 'experiments/training/campaign_rl_data.py', source / 'experiments/training/campaign_rl_entry.py', source / 'packages/sepalith/src/sepalith/campaign_protocol.py']:
    pin(path)

result = {'scope': {'one_cpu': True, 'model_or_weights': False, 'GPU': False, 'training': False, 'final_data': False},
          'rows': {'old_RL': len(old), 'corrected_overlap': len(corrected), 'changed_targets': len(changed), 'excluded_old_RL': len(excluded), 'unchanged_overlap': len(corrected) - len(changed), 'reachable_corrected_192': len(reachable)},
          'validation': dict(validation), 'all_corrected_TRAIN': stats(list(all_corrected), all_corrected), 'old': stats(selected, old), 'corrected_intersection': stats([i for i in selected if i in corrected], corrected),
          'corrected_reachable_192': stats([i for i in selected if i in reachable], corrected),
          'train_dev_overlap': {'row_ids': len(selected_set & {r['id'] for r in dev}), 'group_ids': len(groups & dev_groups), 'package_ids': len(packages & dev_packages)},
          'schedule': {'original_draws': len(order), 'retained_draws': len(retained), 'deleted_draws': len(order) - len(retained), 'retained_mod_8': len(retained) % 8,
                       'reachable_draws': len(reachable_order), 'reachable_mod_8': len(reachable_order) % 8,
                       'original': stats(order, old), 'retained': stats(retained, corrected), 'reachable': stats(reachable_order, corrected)},
          'prefixes': prefixes, 'finish_overlay': dict(overlay_count), 'reward_control_probes': reward_probes,
          'limits': ['Full-R parse results are prior overlay evidence; this worker did not execute R.', 'Canonical reference token shares are not realized BNPO token weights; generated completion lengths and active group advantages control those.', 'Filtered prefixes are diagnostic proposals, not an admitted or materialized draw schedule.'], 'input_pins': pins}
(OUT / 'data-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('rows', 'validation', 'train_dev_overlap', 'finish_overlay')}, indent=2))
