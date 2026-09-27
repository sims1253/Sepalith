"""Independent bounded completion geometry review; no training admission."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import tree_sitter
import tree_sitter_r

EXEC = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
DATA = Path('/mnt/e/sepalith/campaign-20260915/data-work')
sys.path[:0] = [str(EXEC / 'packages/sepalith/src'), str(EXEC / 'experiments/training')]
from campaign_token_audit import selected_context

def sha(data):
    return hashlib.sha256(data).hexdigest()

def offset(text, position):
    lines = text.split('\n')
    number, units = position['line'], position['character']
    assert 0 <= number < len(lines)
    prefix = sum(len(line) + 1 for line in lines[:number])
    # Decode the requested prefix independently of adapter code; cutting a
    # surrogate pair raises instead of silently choosing another position.
    encoded = lines[number].encode('utf-16-le')
    assert 0 <= units * 2 <= len(encoded)
    return prefix + len(encoded[:units * 2].decode('utf-16-le'))

artifact = DATA / 'DAT-04-completion-reviewed-examples-v3.json'
examples = json.loads(artifact.read_text())['real_train_review']['examples']
refs = {r['row_id']: r for r in json.loads((DATA / 'DAT-04-lead-completion-v3-audit-refs.json').read_text())}
by_file = defaultdict(list)
for example in examples:
    by_file[example['source_file']].append(example)
raw_rows = {}
for filename, group in by_file.items():
    wanted = {e['source_line']: e for e in group}
    digest = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if number in wanted:
                example = wanted[number]
                assert sha(line) == example['raw_line_sha256']
                raw_rows[example['row_id']] = json.loads(line)
    assert all(e['source_sha256'] == digest.hexdigest() for e in group)

parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_r.language()))
proof, packets = [], []
for example in examples:
    identifier = example['row_id']
    raw, ref = raw_rows[identifier], refs[identifier]
    result = example['converted_result']
    text = result['selection_source']['document_text']
    assert text == raw['prefix']
    assert sha(text.encode()) == result['context']['replacement_range']['content_sha256']
    region = result['context']['replacement_range']
    first, last = offset(text, region['start']), offset(text, region['end'])
    assert text[first:last] == '\n'.join(result['context']['region_old'])
    actual = text[:first] + '\n'.join(result['target_body']) + text[last:]
    expected = raw['prefix'] + raw['corpus_target']
    assert actual == expected
    selected, selection = selected_context(result)
    assert selected.replacement_range.to_dict() == region
    # The source constructor deliberately withholds the outer closing brace.
    # It is used only to validate the R fragment, never added to the label.
    tree = parser.parse((actual + '}').encode())
    assert not tree.root_node.has_error, identifier
    packet_ref = {k: ref[k] for k in ('row_id', 'group_id', 'split', 'file', 'line', 'source_sha256', 'raw_line_sha256')}
    packet_ref.update(package_id=example['package'], family='finish_block', source_variant=example['audit_family'])
    packets.append({'row_ref': packet_ref, 'family': 'finish_block', 'result': result})
    proof.append({'id': identifier, 'group_id': ref['group_id'], 'kind': example['kind'],
                  'independent_literal_splice': True, 'actual_eof_line': region['end']['line'],
                  'physical_eof_blank': text.endswith('\n'), 'canonical_empty_region': not result['context']['region_old'],
                  'source_selection_pass': True, 'r_fragment_with_source_boundary_pass': True,
                  'required_source_overflow': selection['required_overflow'],
                  'scientific_disposition': 'structural_candidate_pending_bulk_source_support_license_and_collision_gates'})
output = DATA / 'DAT-04-lead-completion-v3-packets.jsonl'
output.write_text(''.join(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + '\n' for p in packets))
report = {'task': 'DAT-04B', 'status': 'independently_verified_structural_candidates_not_admitted',
          'observed_at': datetime.now(timezone.utc).isoformat(), 'worker_artifact': str(artifact),
          'worker_artifact_sha256': sha(artifact.read_bytes()), 'reviewed_rows': len(proof), 'rows': proof,
          'packet_path': str(output), 'packet_sha256': sha(output.read_bytes()),
          'adapter_sha256': sha((EXEC / 'experiments/training/campaign_admission_completion.py').read_bytes()),
          'token_audit_sha256': sha((EXEC / 'experiments/training/campaign_token_audit.py').read_bytes()),
          'final_rows_opened': False, 'cuda_workload_started': False, 'admission': False}
(PLAN / 'docs/campaign/receipts/DAT-04-completion-lead-v3-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'reviewed_rows': len(proof), 'packets': str(output), 'packet_sha256': report['packet_sha256']}))
