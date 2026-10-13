#!/usr/bin/env python3
"""Derive fresh P2 smoke attempts from the reviewed P1 recipes."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
RUNNER = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state')
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
CONTROL = PLAN / 'docs/campaign'
OLD = CONTROL / 'work/rl-smoke-live-preparation/generated-rl03-step1000'
OUT = CONTROL / 'work/rl-generation-throughput-preparation/generated-p2-a'
assert not OUT.exists()
sys.path.insert(0, str(EXEC / 'packages/sepalith/src'))
from sepalith.runner import Runner

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

prior = json.loads((CONTROL / 'receipts/RL-06-reviewed-source-snapshot-v2.json').read_text())
includes = prior['includes']
changed = []
old_source = Path(prior['path']) / 'source'
for name in includes:
    if sha(EXEC / name) != sha(old_source / name): changed.append(name)
assert set(changed).issubset({
    'experiments/training/campaign_rl_train.py',
    'experiments/training/test_campaign_rl_train.py',
    'experiments/training/test_campaign_rl_entry.py',
})
assert 'experiments/training/campaign_rl_train.py' in changed
snapshot = Runner(RUNNER).snapshot(EXEC, includes)
snapshot_id = snapshot if isinstance(snapshot, str) else snapshot.id
source = RUNNER / 'snapshots' / snapshot_id / 'source'
OUT.mkdir(parents=True)
base_native = NATIVE / 'training/RL-03-two-update-p2-a'
assert not base_native.exists()
receipts = []
identities = []
for arm, duration in [('uninterrupted', 900), ('split-first', 720), ('split-resume', 720)]:
    inp = OLD / f'rl-two-update-{arm}.recipe.json'
    recipe = json.loads(inp.read_text())
    recipe['id'] = f'rl03-two-update-p2-{arm}-a'
    recipe['generation_groups_per_call'] = 2
    recipe['identity']['policy']['generation_groups_per_call'] = 2
    recipe['identity']['source']['frozen_source_root'] = str(source)
    recipe['identity']['source']['trainer_sha256'] = sha(source / 'experiments/training/campaign_rl_train.py')
    recipe['deadline'] = '2026-09-12T18:50:00Z'
    recipe['max_attempt_seconds'] = duration
    recipe['output_dir'] = str(base_native / arm / 'output')
    recipe['archive_root'] = str(base_native / arm / 'archive')
    recipe['telemetry_path'] = str(base_native / arm / 'telemetry.jsonl')
    if arm == 'split-resume':
        recipe['resume_from'] = str(base_native / 'split-first/archive/full/checkpoint-1')
    identity_sha = hashlib.sha256(json.dumps(recipe['identity'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    identities.append(identity_sha)
    path = OUT / f'{arm}.recipe.json'
    write(path, recipe)
    command = [str(Path('/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python')),
               str(source / 'experiments/training/campaign_rl_launch.py'), str(path),
               '--receipt', str(base_native / f'{arm}-supervision.json')]
    command_path = OUT / f'{arm}.command.json'
    write(command_path, command)
    receipts.append({'arm': arm, 'recipe': str(path), 'recipe_sha256': sha(path),
                     'derived_from': str(inp), 'derived_from_sha256': sha(inp),
                     'identity_sha256': identity_sha, 'command': str(command_path),
                     'max_attempt_seconds': duration})
assert len(set(identities)) == 1
packet = {'task': 'RL-03/RL-06', 'owner': 'lead', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
          'status': 'prepared_for_bounded_p2_smoke', 'snapshot_id': snapshot_id,
          'path': str(source.parent), 'includes': includes, 'changed_files': changed,
          'preserves_uncommitted_bytes': True, 'recipes': receipts,
          'generation_groups_per_call': 2, 'candidate_count': 4,
          'policy_batch_size': 8, 'gradient_accumulation_steps': 4,
          'cuda_memory_fraction': 0.75, 'checkpoint_reserve_seconds': 120,
          'budget_change': {'old_gate_deadline': '2026-09-12T18:20:00Z', 'new_gate_deadline': '2026-09-12T18:50:00Z',
              'reason': 'Observed P1 setup and 161-second step exhausted the original 600-second smoke before update 2; P2 tests replace remaining P1 attempts.',
              'displaced_work': 'Unused PRM-07 profiling (18 minutes) and optional CUDA kernel attribution (12 minutes).',
              'cloud_spend_change': 0, 'main_stop_unchanged': '2026-09-13T22:00:00Z'},
          'acceptance': ['Actual two complete updates with finite nonzero gradients and DEV readouts.',
                         'Same-P split/resume source/output and optimizer-state comparison.',
                         'Host watchdog and allocator remain healthy; longest TRAIN geometry still requires measured admission before main RL.'],
          'main_rl_admitted': False}
write(CONTROL / 'receipts/RL-03-p2-retry-preparation.json', packet)
print(json.dumps({'snapshot_id': snapshot_id, 'changed_files': changed, 'recipes': receipts}, indent=2))
