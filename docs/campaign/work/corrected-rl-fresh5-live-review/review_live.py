"""Bounded read-only telemetry review; no model, checkpoint or framework imports."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
OUT = Path(__file__).resolve().parent
PLAN = OUT.parents[1]
RUN = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-corrected-theta0-fresh5-v1')
SCHEDULE = PLAN / 'work/corrected-rl-admission-audit-v1/candidate-data/source-row-draw-sequence.json'
schedule_bytes = SCHEDULE.read_bytes()
schedule = json.loads(schedule_bytes)
schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
read_log = []
failures = []

def json_small(path, cap=4*1024*1024):
    if not path.exists():
        return None
    size = path.stat().st_size
    if size > cap:
        failures.append('bounded JSON read limit exceeded: ' + path.name)
        return None
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    read_log.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return data

def jsonl_bounded(path, cap, *, tail=False):
    if not path.exists():
        return []
    size = path.stat().st_size
    if not tail and size > cap:
        failures.append('bounded record file exceeds cap: ' + path.name)
        return []
    start = max(0, size-cap) if tail else 0
    with path.open('rb') as stream:
        stream.seek(start)
        raw = stream.read(min(size-start, cap))
    lines = raw.split(b'\n')
    if start:
        lines = lines[1:]  # Discard partial leading identity row, never parse it.
    lines = lines[:-1]  # An append without final LF is not a complete record.
    result = []
    skipped = 0
    for line in lines:
        if len(line) > 64*1024:
            skipped += 1
            continue
        try:
            result.append(json.loads(line))
        except ValueError:
            failures.append('invalid completed JSONL record: ' + path.name)
    read_log.append({'path': str(path), 'file_bytes_observed': size, 'read_offset': start,
                     'read_bytes': len(raw), 'complete_rows_parsed': len(result), 'oversized_rows_skipped': skipped,
                     'slice_sha256': hashlib.sha256(raw).hexdigest()})
    return result

load = json_small(RUN / 'output/load-audit.json')
load_summary = None
if load:
    load_summary = {key: load[key] for key in ('dtype', 'model_load', 'cuda_allocator')}
    load_summary['adapter'] = {k: v for k, v in load['adapter'].items() if k != 'module_names'}
    tc = load['tokenizer_contract']
    load_summary['tokenizer'] = {'status': tc['status'], 'vocab_mapping_unchanged': tc['vocab_mapping_unchanged'],
        'after': tc['after'], 'prompt_rows_verified': tc['prompt_parity']['checked_rows']}
    if load['dtype'] != 'torch.bfloat16' or load['model_load']['observed_capacity_tokens'] != 4096:
        failures.append('load dtype or capacity mismatch')
    if load['adapter']['attachments'] != 294 or load['adapter']['trainable_parameters'] != 25116672:
        failures.append('adapter load identity mismatch')
    if load['adapter']['lora_rank'] != 16 or load['adapter']['lora_alpha'] != 16 or load['cuda_allocator']['cuda_allocator_fraction'] != 0.8:
        failures.append('adapter or allocator policy mismatch')
    if tc['status'] != 'verified' or tc['prompt_parity']['checked_rows'] != 8246 or not tc['vocab_mapping_unchanged']:
        failures.append('tokenizer parity mismatch')
    if [tc['after'][k] for k in ('bos_token_id','eos_token_id','pad_token_id','vocab_size')] != [0,1,1,130560]:
        failures.append('tokenizer special ID mismatch')
    parent = load['parent']
    if parent['manifest_sha256'] != '1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12' or parent['merged_weights_sha256'] != '499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d':
        failures.append('loaded parent metadata mismatch')

telemetry = jsonl_bounded(RUN / 'telemetry.jsonl', 128*1024, tail=True)
generations = jsonl_bounded(RUN / 'output/generation-records.jsonl', 8*1024*1024)
rewards = jsonl_bounded(RUN / 'output/reward-records.jsonl', 512*1024)
gradients = jsonl_bounded(RUN / 'output/gradient-records.jsonl', 256*1024)
steps = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in telemetry if row.get('event') == 'optimizer_step']
for index, row in enumerate(gradients):
    if row.get('finite') is not True or row.get('nonfinite') is not False or not math.isfinite(row.get('norm', math.nan)):
        failures.append('nonfinite gradient record')
    if row['global_step'] != index:
        failures.append('gradient update sequence mismatch')

complete = min(len(generations), len(rewards)) // 4 * 4
groups = []
family = defaultdict(lambda: {'candidates': 0, 'generated_tokens': 0, 'reward_sum': 0.0, 'exact': 0,
                              'groups': 0, 'nonzero_variance_groups': 0, 'absolute_derived_advantage': 0.0,
                              'absolute_advantage_token_proxy': 0.0, 'failures': Counter()})
for first in range(0, complete, 4):
    gs, rs = generations[first:first+4], rewards[first:first+4]
    step = gs[0]['global_step']
    index = gs[0]['group_geometry']['group_index']
    if (step,index) != (first//32,(first//4)%8):
        failures.append('generation buffer/group sequence mismatch')
    expected_id = schedule['row_ids'][step*8+index]
    for g, r in zip(gs, rs):
        if g['source_schedule_sha256'] != schedule_sha:
            failures.append('source schedule hash mismatch')
        if g['generated_ids_sha256'] != r['output_ids_sha256']:
            failures.append('generation/reward token hash alignment mismatch')
        actual_hash = hashlib.sha256(json.dumps(g['generated_ids'], separators=(',', ':')).encode('ascii')).hexdigest()
        if actual_hash != g['generated_ids_sha256'] or len(g['generated_ids']) != r['generated_tokens']:
            failures.append('actual logged generated IDs do not match digest/count')
        if g['prompt_ids_sha256'] != gs[0]['prompt_ids_sha256']:
            failures.append('within-group prompt hash mismatch')
        if g['global_step'] != step or g['group_geometry']['group_index'] != index or r['id'] != expected_id:
            failures.append('source group order mismatch')
        if not math.isfinite(r['reward']):
            failures.append('nonfinite reward')
    values = [r['reward'] for r in rs]
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    advantages = [(v-mean)/(std+1e-4) for v in values]
    name = rs[0]['family']
    f = family[name]
    f['groups'] += 1
    f['nonzero_variance_groups'] += std > 0
    f['candidates'] += 4
    f['reward_sum'] += sum(values)
    f['exact'] += sum(r['exact_region'] for r in rs)
    f['generated_tokens'] += sum(g['generated_token_count'] for g in gs)
    f['absolute_derived_advantage'] += sum(abs(a) for a in advantages)
    f['absolute_advantage_token_proxy'] += sum(abs(a)*g['generated_token_count'] for a, g in zip(advantages, gs))
    f['failures'].update(r['failure'] for r in rs if r['failure'])
    groups.append({'global_step_before_update': step, 'group_index': index, 'source_id': expected_id, 'family': name,
                   'rewards': values, 'sample_std': std, 'derived_advantages': advantages,
                   'generated_tokens': [g['generated_token_count'] for g in gs]})

total_tokens = sum(v['generated_tokens'] for v in family.values())
total_abs = sum(v['absolute_derived_advantage'] for v in family.values())
total_proxy = sum(v['absolute_advantage_token_proxy'] for v in family.values())
for value in family.values():
    value['generated_token_share'] = value['generated_tokens']/total_tokens if total_tokens else 0
    value['absolute_derived_advantage_share'] = value['absolute_derived_advantage']/total_abs if total_abs else 0
    value['absolute_advantage_token_proxy_share'] = value['absolute_advantage_token_proxy']/total_proxy if total_proxy else 0
    value['reward_mean'] = value['reward_sum']/value['candidates']
    value['failures'] = dict(value['failures'])

terminal = json_small(RUN.with_name(RUN.name+'-host-supervision') / 'terminal.json')
result = {'at': datetime.now(timezone.utc).isoformat(), 'scope': 'bounded CPU telemetry only; no model/checkpoint tensor or startup identity parsing',
          'load': load_summary, 'optimizer_steps': steps, 'gradients': gradients,
          'generation_rows': len(generations), 'reward_rows': len(rewards), 'paired_complete_candidates': complete,
          'complete_source_groups': len(groups), 'expected_full5_source_groups': 40, 'expected_full5_candidates': 160,
          'nonzero_variance_groups': sum(g['sample_std'] > 0 for g in groups),
          'family': dict(family), 'groups': groups, 'terminal': terminal,
          'failures': sorted(set(failures)), 'reads': read_log,
          'limits': ['Advantages are reconstructed from actual logged group rewards with pinned TRL sample-standard-deviation plus1e-4 formula; no trainer advantage tensor was logged.',
                     'Absolute advantage/token products are diagnostic proxies, not measured gradient shares.',
                     'Reward, gradient and source counts are not DEV quality or promotion evidence.']}
(OUT / 'latest.json').write_text(json.dumps(result, indent=2)+'\n')
tag = result['at'].replace(':','').replace('+','_')
(OUT / ('observation-'+tag+'.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('at','optimizer_steps','generation_rows','reward_rows','complete_source_groups','nonzero_variance_groups','family','failures','terminal')}, indent=2))
