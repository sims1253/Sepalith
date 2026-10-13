#!/usr/bin/env python3
"""Verify a completed SFT readout and full state; never select or launch."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--recipe', type=Path, required=True)
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--step', type=int, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
recipe = json.loads(a.recipe.read_text())
snapshot = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots') / recipe['identity']['source'] / 'source'
sys.path[:0] = [str(snapshot / 'experiments/training'), str(snapshot / 'packages/sepalith/src')]
from campaign_checkpoint import verify_checkpoint
from campaign_eval import classify
from sepalith.campaign_protocol import PromptContext
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
checkpoint = a.archive / 'full' / f'checkpoint-{a.step}'
manifest = verify_checkpoint(checkpoint, recipe['identity'], require_full=True)
state = json.loads((checkpoint / 'campaign-state.json').read_text())
assert state['step'] == a.step and state['full'] is True
assert state['sampler']['consumed_draws'] == a.step * 16
assert state['sampler']['schedule_sha256'] == recipe['draw_schedule']['sha256']
panel = Path(recipe['development_panel']['path'])
assert sha(panel) == recipe['development_panel']['sha256']
cases = {r['id']: r for r in map(json.loads, panel.open())}
assert len(cases) == 75 and set(cases) == set(recipe['development_case_ids'])
casepath = a.archive / 'evaluations' / f'cases-step-{a.step}.json'
readout = json.loads(casepath.read_text())
assert readout['status'] == 'complete' and readout['step'] == a.step
results = readout['results']; summary = readout['summary']
assert len(results) == 75 and len({r['id'] for r in results}) == 75
assert {r['id'] for r in results} == set(cases) == set(summary['case_ids'])
counts = Counter(); family = defaultdict(Counter)
for r in results:
    case = cases[r['id']]
    assert case['split'] == 'dev' and r['family'] == case['family']
    assert r['package_id'] == case['package_id']
    context = PromptContext.from_mapping(case['context'])
    expected_noop = list(context.region_old) == case['region_new']
    assert r['expected_noop'] == expected_noop
    actual = classify(r['raw_output'], context, case['region_new'], r['generated_ids'])
    assert all(r[k] == v for k, v in actual.items())
    assert r['generated_tokens'] == len(r['generated_ids']) <= recipe['development_max_new_tokens']
    capped = len(r['generated_ids']) == recipe['development_max_new_tokens'] and r['generated_ids'][-1] != 1
    assert r['cap_hit'] == capped
    assert all(math.isfinite(v) and v >= 0 for v in r['loss'].values())
    row_counts = {k: int(r[k]) for k in ['protocol_valid','exact_region','predicted_noop','suggestion','cap_hit']}
    row_counts.update(edit_exact=int(not expected_noop and r['exact_region']),
                      strict_noop_correct=int(expected_noop and r['predicted_noop']),
                      strict_noop_false_suggestions=int(expected_noop and r['suggestion']))
    counts.update(row_counts); family[r['family']].update(row_counts); family[r['family']]['cases'] += 1
assert dict(counts) == summary['counts']
denom = {'cases':75,'packages':len({r['package_id'] for r in results}),
         'strict_noop':sum(r['expected_noop'] for r in results),'edits':sum(not r['expected_noop'] for r in results),
         'prompt_loss_tokens':sum(r['loss']['prompt_tokens'] for r in results),
         'target_loss_tokens':sum(r['loss']['target_tokens'] for r in results)}
assert denom == summary['denominators'] and denom['strict_noop'] == 32 and denom['edits'] == 43
for field in ['prompt','target']:
    recomputed = sum(r['loss'][field+'_nll_sum'] for r in results) / denom[field+'_loss_tokens']
    assert math.isclose(recomputed, summary[field+'_nll'], abs_tol=1e-12, rel_tol=1e-12)
import torch
torch.set_num_threads(1)
optimizer = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
assert len(optimizer['state']) == 588
steps = {int(v['step']) for v in optimizer['state'].values()}; assert steps == {a.step}
assert all(torch.isfinite(t).all() for v in optimizer['state'].values() for t in v.values() if torch.is_tensor(t))
scheduler = torch.load(checkpoint/'scheduler.pt', map_location='cpu', weights_only=False)
assert scheduler['last_epoch'] == a.step and scheduler['_step_count'] == a.step + 1
evidence = {'status':'complete_readout_and_full_state_verified','step':a.step,
            'checkpoint':str(checkpoint),'manifest_sha256':sha(checkpoint/'campaign-manifest.json'),
            'per_case_sha256':sha(casepath),'counts':dict(counts),'denominators':denom,
            'family':{k:dict(v) for k,v in family.items()},'prompt_nll':summary['prompt_nll'],
            'target_nll':summary['target_nll'],'optimizer_states_verified':588,'optimizer_step':a.step,
            'scheduler_epoch':scheduler['last_epoch'],'scheduler_step_count':scheduler['_step_count'],
            'sampler_consumed_draws':state['sampler']['consumed_draws'],
            'scientific_decision':'Pending root comparison with earlier checkpoints; no launch or promotion.'}
with a.out.open('x') as f:
    json.dump(evidence,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in evidence.items() if k not in ['family']},sort_keys=True))
