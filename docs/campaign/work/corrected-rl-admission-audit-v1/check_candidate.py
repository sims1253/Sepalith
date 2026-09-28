"""Repeatable CPU preflight; full parent-weight verification is explicit root-only opt-in."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recipe', type=Path, default=Path(__file__).with_name('fresh5.recipe.json'))
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--verify-parent-weights', action='store_true')
args = parser.parse_args()
recipe = json.loads(args.recipe.read_text())
source = Path(recipe['identity']['source']['frozen_source_root'])
sys.path.insert(0, str(source / 'experiments/training'))
from campaign_rl_data import sha256_file
from campaign_rl_train import preflight_rl_recipe, CampaignRepeatSampler
from campaign_rl_entry import _validate_entry_contract, _validate_schedule, preflight_entry

assert sha256_file(source / 'experiments/training/campaign_rl_train.py') == recipe['identity']['source']['trainer_sha256']
assert sha256_file(source / 'packages/sepalith/src/sepalith/campaign_protocol.py') == recipe['identity']['source']['protocol_sha256']
packet = preflight_rl_recipe(recipe)
_validate_entry_contract(recipe, packet['identity'])
_validate_schedule(recipe, packet['geometry'])
if args.verify_parent_weights:
    packet = preflight_entry(recipe)
assert recipe['resume_from'] is None and recipe['max_steps'] == 5
assert recipe['evaluation_steps'] == recipe['decision_steps'] == [5]
assert recipe['max_attempt_seconds'] == 1800 and recipe['checkpoint_reserve_seconds'] == 600
schedule = json.loads(Path(recipe['data']['source_draw_schedule_path']).read_text())
selected = recipe['identity']['data']['selected_ids']
indices = {key: i for i, key in enumerate(selected)}
sequence = [indices[key] for key in schedule['row_ids']]
sampler = CampaignRepeatSampler(list(range(len(selected))), candidate_count=4, prompt_groups_per_batch=8, repeat_count=8,
    seed=3407, source_draw_sequence=sequence,
    source_draw_sequence_sha256=schedule['sequence_sha256'], source_draw_schedule_sha256=recipe['data']['source_draw_schedule_sha256'],
    generation_batch_size=32, per_device_train_batch_size=4, gradient_accumulation_steps=8, steps_per_generation=8)
states = {}
for step in (5, 25):
    state = sampler.state(consumed_rows=step * 256)
    assert state['source_draw_cursor'] == step * 8
    expected = sequence[step * 8:step * 8 + 8]
    observed = sampler.indices_from(step * 256, 256)
    assert observed == [index for _ in range(8) for index in expected for _ in range(4)]
    states[str(step)] = {'consumed_rows': step * 256, 'source_draw_cursor': state['source_draw_cursor'], 'next_update_order_exact': True}

identity_sha = hashlib.sha256(json.dumps(recipe['identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
result = {'at': datetime.now(timezone.utc).isoformat(), 'status': 'CPU_preflight_pass; root admission remains external',
          'recipe_sha256': sha256_file(args.recipe), 'identity_sha256': identity_sha, 'rows': packet['records'],
          'geometry': packet['geometry'], 'sampler_boundaries': states, 'parent_weights_verified': args.verify_parent_weights,
          'framework_imports': packet['framework_imports'], 'launch_performed': False}
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
