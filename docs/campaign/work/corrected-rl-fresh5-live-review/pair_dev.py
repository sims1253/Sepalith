"""Re-score retained baseline and completed RL outputs against corrected DEV75."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PLAN = OUT.parents[1]
SOURCE = Path('/home/m0hawk/.local/state/sepalith/migration-20260906/prepared-state/snapshots/be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9/source')
sys.path.insert(0, str(SOURCE/'experiments/training'))
sys.path.insert(0, str(SOURCE/'packages/sepalith/src'))
from campaign_eval import classify
from sepalith.campaign_protocol import PromptContext

def read(path):
    assert path.stat().st_size < 4*1024*1024
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

baseline_path = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/checkpoints/SFT-primary-3000-a/evaluations/cases-step-1000.json')
baseline, baseline_pin = read(baseline_path)
assert baseline_pin['sha256'] == '1eddac3b78251e27bce4465b86637a021c4ff34569df51a297bac52403aea4ea'
panel_path = PLAN/'work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl'
panel_bytes = panel_path.read_bytes()
assert hashlib.sha256(panel_bytes).hexdigest() == '7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'
panel = [json.loads(line) for line in panel_bytes.splitlines()]
cases = {case['id']: case for case in panel}
assert len(cases) == len(panel) == 75

def score(artifact):
    assert artifact['status'] == 'complete'
    results = artifact['results']
    assert len(results) == len({r['id'] for r in results}) == 75
    assert {r['id'] for r in results} == set(cases)
    output = []
    for row in results:
        case = cases[row['id']]
        context = PromptContext.from_mapping(case['context'])
        expected = list(context.region_old) if case['operation'] == 'no_op' else case['target_body_text'].split('\n')
        classification = classify(row['raw_output'], context, expected, row['generated_ids'])
        assert row['generated_tokens'] == len(row['generated_ids']) <= 512
        assert classification['protocol_valid'] == row['protocol_valid']
        assert classification['predicted_noop'] == row['predicted_noop']
        output.append({'id': row['id'], 'family': case['family'], 'expected_noop': case['operation'] == 'no_op',
                       **classification, 'cap_hit': row['cap_hit'], 'generated_tokens': row['generated_tokens'],
                       'prompt_tokens': row['prompt_tokens'],
                       'raw_output_sha256': hashlib.sha256(row['raw_output'].encode()).hexdigest()})
    counts = {'cases': 75, 'edit_cases': 43, 'noop_cases': 32,
              'edits': sum(r['exact_region'] and not r['expected_noop'] for r in output),
              'correct_noops': sum(r['exact_region'] and r['expected_noop'] for r in output),
              'false_suggestions': sum(r['suggestion'] and r['expected_noop'] for r in output),
              'protocol_valid': sum(r['protocol_valid'] for r in output), 'caps': sum(r['cap_hit'] for r in output),
              'finish_exact': sum(r['exact_region'] and r['family'] == 'finish_block' for r in output),
              'finish_cases': sum(r['family'] == 'finish_block' for r in output)}
    return output, counts

base_rows, base_counts = score(baseline)
assert [base_counts[k] for k in ('edits', 'correct_noops', 'false_suggestions', 'finish_exact')] == [26,25,5,0]
result = {'at': datetime.now(timezone.utc).isoformat(), 'baseline_pin': baseline_pin,
          'panel_pin': {'path': str(panel_path), 'sha256': hashlib.sha256(panel_bytes).hexdigest(), 'rows': 75},
          'baseline_corrected_counts': base_counts, 'baseline_rescored_rows': base_rows,
          'comparison_policy': 'Same stored predictions, same corrected target authority, same512-token diagnostic cap. Do not compare old uncorrected NLL to new corrected NLL.'}
candidate_path = Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-corrected-theta0-fresh5-v1/archive/evaluations/cases-step-5.json')
if candidate_path.exists():
    candidate, candidate_pin = read(candidate_path)
    if candidate.get('status') == 'complete':
        current_rows, current_counts = score(candidate)
        base_by_id = {r['id']:r for r in base_rows}
        assert all(r['prompt_tokens'] == base_by_id[r['id']]['prompt_tokens'] for r in current_rows)
        improvements = [r for r in current_rows if r['exact_region'] and not base_by_id[r['id']]['exact_region']]
        regressions = [r for r in current_rows if not r['exact_region'] and base_by_id[r['id']]['exact_region']]
        result.update(status='paired_complete', candidate_pin=candidate_pin, candidate_corrected_counts=current_counts,
                      candidate_rescored_rows=current_rows, exact_improvements=improvements, exact_regressions=regressions,
                      raw_predictions_unchanged=sum(r['raw_output_sha256'] == base_by_id[r['id']]['raw_output_sha256'] for r in current_rows),
                      matched_prompt_token_counts=75, floors_pass=current_counts['edits']>=26 and current_counts['correct_noops']>=25 and current_counts['false_suggestions']<=5)
        losses = [r['loss'] for r in candidate['results']]
        result['new_corrected_target_loss_only'] = {'target_nll_sum': sum(r['target_nll_sum'] for r in losses),
            'target_tokens': sum(r['target_tokens'] for r in losses)}
        result['new_corrected_target_loss_only']['target_nll'] = result['new_corrected_target_loss_only']['target_nll_sum']/result['new_corrected_target_loss_only']['target_tokens']
    else:
        result['status'] = 'candidate_not_complete'
else:
    result['status'] = 'baseline_rescored_candidate_pending'
(OUT/'paired-dev.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('baseline_rescored_rows','candidate_rescored_rows','exact_improvements','exact_regressions')}, indent=2))
