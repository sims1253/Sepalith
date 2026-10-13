#!/usr/bin/env python3
"""Bind a smaller policy microbatch without changing the source draw stream."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
C = PLAN / 'docs/campaign'
N = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')

old_path = C / 'work/main-rl-preparation/primary-mixed-a.recipe.json'
require(sha(old_path) == 'e6a6ba7acaebfac8f8a53335841a474eaa29e8ae91353febb6e938af54cdc4e0', 'prior recipe changed')
r = json.loads(old_path.read_text())
source = Path(r['identity']['source']['frozen_source_root'])
sys.path[:0] = [str(source / 'experiments/training'), str(source / 'packages/sepalith/src')]
from campaign_rl_train import load_source_draw_schedule, CampaignRepeatSampler, resolve_trl_geometry

old_schedule_path = Path(r['data']['source_draw_schedule_path'])
old = json.loads(old_schedule_path.read_text())
require(sha(old_schedule_path) == '2e75e10bf86f6b72e33fd72cfed96194bdd7c7c4741db8793681985292581574', 'v5 changed')
v6 = dict(old)
for field in ('buffer_reuse', 'gradient_accumulation_steps', 'steps_per_generation'):
    require(old[field] == 4, 'unexpected old geometry')
    v6[field] = 8
v6['status'] = 'frozen_lead_interleaved_microbatch4'
require(v6['row_ids'] == old['row_ids'], 'draw stream changed')
schedule_path = C / 'work/rl-pool/source-row-draw-sequence-v6-interleaved-mb4.json'
write(schedule_path, v6)
schedule_sha = sha(schedule_path)
selected = json.loads(Path(r['data']['selected_ids_path']).read_text())['row_ids']
loaded = load_source_draw_schedule(
    schedule_path, schedule_sha, selected_ids=selected,
    selected_ids_sha256=v6['selected_ids_sha256'], ordered_ids_sha256=v6['ordered_ids_sha256'],
    row_identity_sha256=v6['row_identity_sha256'], candidate_count=4,
    source_draws_per_update=8, buffer_reuse=8,
)
geometry = resolve_trl_geometry(4, rollout_rows_per_update=32,
    per_device_train_batch_size=4, gradient_accumulation_steps=8)
indices = {row_id: i for i, row_id in enumerate(selected)}
sampler = CampaignRepeatSampler(selected, candidate_count=4, prompt_groups_per_batch=8,
    repeat_count=8, generation_batch_size=32, per_device_train_batch_size=4,
    gradient_accumulation_steps=8, steps_per_generation=8,
    source_draw_sequence=[indices[x] for x in v6['row_ids']],
    source_draw_sequence_sha256=v6['sequence_sha256'], source_draw_schedule_sha256=schedule_sha)
stream = list(sampler)
require(len(stream) == 768000, 'wrong sampler length')
for update in range(25):
    expected = [indices[x] for x in v6['row_ids'][update*8:(update+1)*8] for _ in range(4)] * 8
    require(stream[update*256:(update+1)*256] == expected, 'wrong repeated generation buffer')
boundaries = {}
for update in (1, 5, 25, 100, 250):
    cursor = update * 256
    state = sampler.state(consumed_rows=cursor)
    require(state['source_draw_cursor'] == update * 8, 'source cursor mismatch')
    require(sampler.indices_from(cursor, 256) == stream[cursor:cursor+256], 'resume suffix mismatch')
    boundaries[str(update)] = {'consumed_rows': cursor, 'source_draw_cursor': update * 8}

admission = json.loads((C / 'receipts/RL-02-v5-interleaved-admission.json').read_text())
admission.update(at=dt.datetime.now(dt.timezone.utc).isoformat(),
    scope='Unchanged TRAIN rows, quotas and exact ordered draw sequence; buffer metadata now matches policy microbatch4/accumulation8. Main launch requires a separate root resource admission.')
admission['source_schedule'] = {'path': str(schedule_path), 'sha256': schedule_sha,
    'sequence_sha256': v6['sequence_sha256']}
admission['microbatch_change'] = {'from': [8,4], 'to': [4,8], 'rollout_rows_per_update': 32,
    'sampler_rows_per_update': 256, 'source_draws_per_update': 8,
    'production_loader_passed': True, 'first25_sampler_buffers_passed': True,
    'resume_boundaries': boundaries, 'total_sampler_rows': len(stream),
    'BNPO_note': 'Per-microbatch token normalization changes effective row weighting. This is a new stage from theta0; no optimizer-equivalence claim to8/4.'}
admission['independent_order_review'] = {'path': str(C / 'receipts/RL-02-prefix-mixture-review.json'),
    'sha256': sha(C / 'receipts/RL-02-prefix-mixture-review.json'),
    'root_accepted': True, 'scope': 'v5 order and exact quotas; v6 has identical row_ids'}
admission_path = C / 'receipts/RL-02-v6-microbatch4-admission.json'
write(admission_path, admission)

run = N / 'training/RL-primary-p2-mixed-mb4-a'
require(not run.exists(), 'run already exists')
r.update(id='primary-grpo-p2-2048x192-v4-mb4-a', output_dir=str(run/'output'),
    archive_root=str(run/'archive'), telemetry_path=str(run/'telemetry.jsonl'))
r['data'].update(source_draw_schedule_path=str(schedule_path),
    source_draw_schedule_sha256=schedule_sha, buffer_reuse=8)
r['rl02_admission'] = {'path': str(admission_path), 'sha256': sha(admission_path)}
i = r['identity']
i['policy'].update(per_device_train_batch_size=4, gradient_accumulation_steps=8)
i['schedule'].update(stage_id='primary-grpo-p2-2048x192-v4-mb4', buffer_reuse=8,
    steps_per_generation=8, source_draw_schedule_sha256=schedule_sha)
i['source'].update(source_schedule_sha256=schedule_sha,
    rl02_admission_receipt_sha256=sha(admission_path))
recipe_path = C / 'work/main-rl-preparation/primary-mixed-mb4-a.recipe.json'
write(recipe_path, r)
command = ['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python',
    str(source/'experiments/training/campaign_rl_launch.py'),str(recipe_path),
    '--receipt',str(run/'supervision.json')]
command_path = recipe_path.with_name('primary-mixed-mb4-a.command.json')
write(command_path, command)
packet = {'task': 'RL-04/RL-05/RL-06', 'owner': 'lead',
    'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'prepared_not_launched',
    'recipe': str(recipe_path), 'recipe_sha256': sha(recipe_path),
    'command': str(command_path), 'source_snapshot': source.parent.name,
    'identity_sha256': hashlib.sha256(json.dumps(i, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    'geometry': geometry.to_dict(), 'generation_groups_per_call':2,
    'sequence_sha256': v6['sequence_sha256'], 'schedule_sha256': schedule_sha,
    'CPU_loader_sampler_pass':True, 'BNPO_weighting_change':admission['microbatch_change']['BNPO_note'],
    'acceptance':['Actual mixed first backward and finite optimizer update below memory limits',
        '32 generation/reward rows and8 sources per update',
        'Full checkpoint25 withDEV75 before continuation'],
    'budget':{'ceiling_seconds':93600,'charged_before_seconds':3183.145718829008,
        'remaining_before_seconds':90416.85428117099,'attempt_seconds':5400,
        'cloud_spend_delta':0,'hard_stop':'2026-09-13T22:00:00Z'},
    'displaced_work':'No added cloud or deadline. This retry replaces failed8/4 mixed run; optional kernel work remains cut.'}
write(C / 'receipts/RL-04-mixed-mb4-preparation.json', packet)
print(json.dumps(packet, indent=2))
