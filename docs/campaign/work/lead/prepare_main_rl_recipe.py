#!/usr/bin/env python3
"""Prepare a main-stage recipe; this script never launches CUDA."""
from pathlib import Path
import hashlib
import json

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
NATIVE = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')
CONTROL = PLAN / 'docs/campaign'
source_recipe = CONTROL / 'work/rl-generation-throughput-preparation/generated-p2-a/uninterrupted.recipe.json'
assert hashlib.sha256(source_recipe.read_bytes()).hexdigest() == 'e4cc42dd72fec1ea9c41d831c896171a3317472416426f0253e4e3a7d5a91ba3'
recipe = json.loads(source_recipe.read_text())
panel = Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl')
panel_sha = hashlib.sha256(panel.read_bytes()).hexdigest()
assert panel_sha == 'b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21'
cases = [json.loads(line) for line in panel.read_text().splitlines()]
assert len(cases) == 75 and all(row['split'] == 'dev' for row in cases)
identity = recipe['identity']
identity['schedule'].pop('smoke_updates')
identity['schedule'].update({
    'stage_id': 'primary-grpo-p2-2048x192-v1',
    'development_panel_sha256': panel_sha,
    'development_cases': 75,
    'light_save_steps': 5,
    'full_save_steps': 25,
    'development_every': 25,
})
out = CONTROL / 'work/main-rl-preparation'
assert not out.exists()
out.mkdir()
run = NATIVE / 'training/RL-primary-p2-a'
assert not run.exists()
recipe.update({
    'id': 'primary-grpo-p2-2048x192-v1-a',
    'max_steps': 3000,
    'decision_steps': [25],
    'light_save_steps': 5,
    'full_save_steps': 25,
    'evaluation_steps': list(range(25, 3001, 25)),
    'development_panel': {'path': str(panel), 'sha256': panel_sha},
    'development_case_ids': [row['id'] for row in cases],
    'max_attempt_seconds': 5400,
    'checkpoint_reserve_seconds': 600,
    'termination_grace_seconds': 60,
    'deadline': '2026-09-13T22:00:00Z',
    'output_dir': str(run / 'output'),
    'archive_root': str(run / 'archive'),
    'telemetry_path': str(run / 'telemetry.jsonl'),
    'resume_from': None,
})
path = out / 'primary-a.recipe.json'
path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
command = [
    '/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',
    str(Path(identity['source']['frozen_source_root']) / 'experiments/training/campaign_rl_launch.py'),
    str(path), '--receipt', str(run / 'supervision.json'),
]
(out / 'primary-a.command.json').write_text(json.dumps(command, indent=2) + '\n')
packet = {
    'task': 'RL-04/RL-05', 'owner': 'lead', 'status': 'prepared_not_admitted',
    'recipe': str(path), 'recipe_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'identity_sha256': hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    'stage_id': identity['schedule']['stage_id'],
    'first_stop_step': 25, 'full_every': 25, 'light_every': 5,
    'DEV_cases': 75, 'DEV_edits': 43, 'DEV_noops': 32,
    'initial_attempt_seconds': 5400, 'host_guard_seconds': 5460,
    'main_all_in_hours_ceiling': 26,
    'fresh_theta0_start': True,
    'reason_for_fresh_start': 'Main-stage identity binds the full75 DEV panel and new checkpoint cadence. Two smoke updates are mechanism validation and are not promoted into the main trajectory.',
    'main_launch_admitted': False,
    'remaining_gates': ['Same-P full-state resume proof', 'Longest TRAIN P2 memory gate', 'Root stage contract and budget admission'],
}
(CONTROL / 'receipts/RL-04-main-recipe-preparation.json').write_text(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
