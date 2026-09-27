"""Review the completed broad batch and check documentation after selection."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
sys.path[:0] = [str(ROOT / 'packages/sepalith/src'), str(ROOT / 'experiments/training')]
from campaign_token_audit import selected_context
from campaign_support_audit import roxygen_formal_check
from tree_sitter import Language, Parser
import tree_sitter_r

DATA = Path('/mnt/e/sepalith/campaign-20260915/data-work')
PLAN = Path(__file__).resolve().parents[2]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb', buffering=4194304) as f:
        for b in iter(lambda: f.read(4194304), b''): h.update(b)
    return h.hexdigest()

source = DATA / 'DAT-04-structured-batch-broad-v2/candidate-packets.jsonl'
assert sha(source) == '8bf29849a41ee35fa9d0a4928f07e54dd8363835ec5ba9d2964dacc45b0d6ef2'
manifest = DATA / 'DAT-02-global-split-v2.json'
assert sha(manifest) == 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
groups = {g['group_id']: g for g in json.loads(manifest.read_text())['groups']}
parser = Parser(Language(tree_sitter_r.language()))
output = DATA / 'DAT-04-structured-broad-support-packets.jsonl'
checks, excluded, counts, seen, parents = [], [], Counter(), set(), set()
with source.open(encoding='utf-8') as f, output.open('x', encoding='utf-8') as out:
    for line in f:
        p = json.loads(line); ref = p['row_ref']; result = p['result']; row_id = ref['row_id']
        assert row_id not in seen; seen.add(row_id)
        assert ref['split'] == groups[ref['group_id']]['split'] == 'train_group'
        assert result['status'] == 'converted'
        v = p['validation']
        for name in ('fresh_source_constructor', 'full_buffer_application',
                     'normalized_parent_R_parse', 'source_selection_support_check'):
            assert v[name] is True, (row_id, name)
        assert v['parent_group_split'] == 'train_group' and v['admission'] is False
        assert v['license_evidence']['fields']
        if p['family'] == 'roxygen_drafting':
            try:
                context, selection = selected_context(result)
                c = roxygen_formal_check(context.to_dict(), result['target_body'], parser)
            except ValueError as error:
                c = {'status': 'excluded', 'reason': str(error)}
            checks.append({'id': row_id, **c})
            if c['status'] != 'passed':
                excluded.append({'id': row_id, 'reason': c['reason']}); continue
        counts[p['family']] += 1; parents.add(ref['group_id'])
        out.write(line)
assert len(seen) == 7506
receipt = {
    'task': 'DAT-04', 'status': 'lead_reviewed_candidates_pending_tokenization_and_registry',
    'observed_at': datetime.now(timezone.utc).isoformat(),
    'source': str(source), 'source_sha256': sha(source),
    'output': str(output), 'output_sha256': sha(output),
    'input_count': len(seen), 'retained_count': sum(counts.values()),
    'retained_families': dict(counts), 'retained_groups': len(parents),
    'all_source_validation_fields_checked': True, 'all_parent_groups_train': True,
    'roxygen_checked_after_prediction_time_selection': checks,
    'excluded': excluded, 'exclusion_counts': dict(Counter(x['reason'] for x in excluded)),
    'review_source_sha256': sha(Path(__file__)),
    'support_helper_sha256': sha(ROOT / 'experiments/training/campaign_support_audit.py'),
    'limits': ['Formal-name check is not proof of natural-language semantics.',
               'Source-derived simulations are not observed editor trajectories.',
               'Final set remains sealed; no training admission or quality claim.'],
}
(PLAN / 'receipts/DAT-04-structured-broad-support-review.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('input_count', 'retained_count', 'retained_families', 'retained_groups', 'exclusion_counts', 'output_sha256')}))
