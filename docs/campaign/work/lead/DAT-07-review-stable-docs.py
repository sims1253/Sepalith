"""Review existing dev candidates; discard unsupported derived edit labels."""
import hashlib, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
sys.path[:0] = [str(ROOT / 'packages/sepalith/src'), str(ROOT / 'experiments/training')]
from campaign_token_audit import selected_context
from campaign_support_audit import roxygen_formal_check
from tree_sitter import Language, Parser
import tree_sitter_r

DATA = Path('/mnt/e/sepalith/campaign-20260915/data-work')
PLAN = Path(__file__).resolve().parents[2]
def sha(b): return hashlib.sha256(b).hexdigest()
panel_path = DATA / 'DAT-07-dev-panel-candidates.json'
assert sha(panel_path.read_bytes()) == '9c6602f7a3d10aad255bce5df9d457ab6b23fe29b3de7f82383312183c32b66d'
panel = json.loads(panel_path.read_bytes())
manifest = DATA / 'DAT-02-global-split-v2.json'
assert sha(manifest.read_bytes()) == 'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
groups = {g['group_id']: g for g in json.loads(manifest.read_bytes())['groups']}
parser = Parser(Language(tree_sitter_r.language()))
source_path = DATA / 'DAT-07-lead-panel-records-v2.jsonl'
lines = [json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode() + b'\n' for r in panel['rows']]
with source_path.open('xb') as f:
    for line in lines: f.write(line)
source_sha = sha(source_path.read_bytes())

# Only selected dev source records are decoded; all bytes are hashed.
requested = defaultdict(dict)
for r in panel['rows']:
    o = r['origin']
    if r['family'] == 'no_op' and o['kind'] == 'existing_dev_row' or r['source_family'] == 'roxygen_drafting':
        requested[o['raw_file']][o['raw_line']] = r
raw_rows, source_checks = {}, []
for filename, wanted in requested.items():
    digest = hashlib.sha256()
    with Path(filename).open('rb', buffering=4194304) as f:
        for n, line in enumerate(f, 1):
            digest.update(line)
            if n in wanted:
                r = wanted[n]; assert sha(line) == r['origin']['raw_line_sha256']
                raw_rows[r['id']] = json.loads(line)
    assert all(r['origin']['raw_source_sha256'] == digest.hexdigest() for r in wanted.values())
    source_checks.append({'path': filename, 'sha256': digest.hexdigest(), 'selected_dev_records': len(wanted)})

out_path = DATA / 'DAT-07-lead-stable-doc-packets-v2.jsonl'
checked, excluded, families = [], [], Counter()
with out_path.open('x', encoding='utf-8') as out:
    for n, (r, raw_line) in enumerate(zip(panel['rows'], lines), 1):
        family = r['family']; rid = r['id']; o = r['origin']
        reason = None
        if family not in ('no_op', 'docs'):
            reason = 'derived_edit_has_no_visible_preceding_intent' if o['kind'] == 'source_derived_companion' else 'rewrite_label_requires_separate_visible_repair_review'
        elif family == 'docs' and r['source_family'] != 'roxygen_drafting':
            reason = 'mid_roxygen_boundary_and_support_unreviewed'
        if reason:
            excluded.append({'id': rid, 'reason': reason}); continue
        try:
            assert groups[r['group_id']]['split'] == r['parent_identity']['split'] == 'dev_group'
            src = r['source']; b = Path(src['path']).read_bytes(); assert sha(b) == src['content_sha256']
            text = b.decode('utf-8'); physical = text.replace('\r\n', '\n').split('\n')
            lo, hi = src['region_start_line'], src['region_end_line']
            assert physical[lo:hi + 1] == src['physical_region_old']
            target = src['physical_region_old'] if family == 'no_op' else r['target_body']
            post = physical[:lo] + target + physical[hi + 1:]
            post_bytes = ('\r\n' if src['source_eol'] == 'crlf' else '\n').join(post).encode()
            assert sha(post_bytes) == src['post_edit_sha256']
            if family == 'no_op': assert post_bytes == b
            if parser.parse(b).root_node.has_error or parser.parse(post_bytes).root_node.has_error:
                raise ValueError('complete_staged_source_R_parse_error')
            license = r['parent_identity']['license']
            license_bytes = Path(license['path']).read_bytes(); assert sha(license_bytes) == license['content_sha256']
            if re.search(rb'(License_restricts_use:\s*yes|FOSS:\s*no)', license_bytes, re.I):
                raise ValueError('source_license_restriction_requires_review')
            raw = raw_rows.get(rid)
            if family == 'docs':
                assert raw['region_new'] == r['target_body']
                if any(re.match(r"\s*#'\s*@(author|references|source|examples|seealso)\b", line) for line in target):
                    raise ValueError('documentation_external_facts_require_separate_support_review')
            elif raw is not None:
                assert raw['region_new'] == []
            else:
                region = '\n'.join(r['context']['region_old'])
                assert '|>' in region or re.search(r'\bna\.rm\s*=\s*TRUE\b', region)
            result = {'status': 'converted', 'context': {'schema_version': 'sepalith.prompt.prm03.v1', **r['context']}, 'operation': r['operation'],
                      'target_body': r['target_body'], 'selection_source': r['selection_source'],
                      'provenance': {'verification': 'DAT-07-lead-reviewed-source-derived-dev',
                                     'parent_identity': r['parent_identity'], 'origin': o,
                                     'source_snapshot_sha256': sha(b), 'post_edit_sha256': sha(post_bytes),
                                     'source_is_simulated': o['kind'] == 'source_derived_companion'}}
            context, selection = selected_context(result)
            support = None
            if family == 'docs':
                support = roxygen_formal_check(context.to_dict(), target, parser)
                if support['status'] != 'passed': raise ValueError(support['reason'])
            ref = {'verification': 'DAT-07-lead-panel-record', 'row_id': rid, 'group_id': r['group_id'],
                   'split': 'dev_group', 'package_id': r['package'], 'family': 'no_op' if family == 'no_op' else 'roxygen_drafting',
                   'source': r['source_family'], 'file': str(source_path), 'line': n,
                   'source_sha256': source_sha, 'raw_line_sha256': sha(raw_line)}
            out.write(json.dumps({'row_ref': ref, 'family': ref['family'], 'result': result}, ensure_ascii=False, separators=(',', ':')) + '\n')
            checked.append({'id': rid, 'group_id': r['group_id'], 'family': ref['family'], 'source_hash': sha(b),
                            'post_edit_hash': sha(post_bytes), 'formal_check': support, 'full_R_parse': True})
            families[ref['family']] += 1
        except (AssertionError, ValueError, KeyError) as error:
            excluded.append({'id': rid, 'reason': str(error) or type(error).__name__})
receipt = {'task': 'DAT-07', 'status': 'partial_source_review_pending_tokenizer_and_propagation_completion_panels',
           'observed_at': datetime.now(timezone.utc).isoformat(), 'input': str(panel_path), 'input_sha256': sha(panel_path.read_bytes()),
           'review_source_sha256': sha(Path(__file__).read_bytes()), 'source_checks': source_checks,
           'input_count': len(panel['rows']), 'retained_count': len(checked), 'retained_families': dict(families),
           'checks': checked, 'excluded': excluded, 'exclusion_counts': dict(Counter(x['reason'] for x in excluded)),
           'output': str(out_path), 'output_sha256': sha(out_path.read_bytes()),
           'development_only': True, 'admission': False, 'final_content_opened': False}
(PLAN / 'receipts/DAT-07-lead-stable-doc-review-v2.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('retained_count', 'retained_families', 'exclusion_counts', 'output_sha256')}))
