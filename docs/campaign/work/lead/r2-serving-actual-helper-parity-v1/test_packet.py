import hashlib,json
from pathlib import Path
HERE=Path(__file__).parent
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
f=json.loads((HERE/'actual-helper-fixture.json').read_text());screen=json.loads((HERE/'candidate-screen.json').read_text());remote=json.loads((HERE/'notebook/parity.json').read_text());parse=json.loads((HERE/'notebook/parse-only.json').read_text());summary=json.loads((HERE/'comparison.json').read_text())
assert len(screen['rows'])==8 and all(not x['exact_helper_names'] for x in screen['rows'])
assert sum(x['status']=='unresolved' for x in screen['rows'])==7
assert sum(x['status']=='not_applicable' for x in screen['rows'])==1
assert f['semantic_v6_status']=='semantic_supported_context_closure_root_review_required'
assert f['provenance']['split']=='train_group'
assert sha(f['preedit_text'])==f['preedit_sha256']
assert sha(f['source_after_text'])==f['source_parse_sha256']
lines=f['source_after_text'].split('\n')
for helper in f['helpers']:
 start,end=helper['analyzer_span_1based']
 assert '\n'.join(lines[start-1:end])==helper['content']
 assert sha(helper['content']) in remote['expected_context']['selected_reference_sha256']
assert remote['status']=='mismatch'
assert remote['geometry']['source_reapplication_exact'] is True
assert remote['geometry']['replacement_range']['content_sha256']==f['preedit_sha256']
assert parse['baseline']['status']==parse['reapplied']['status']=='parse_ok'
assert parse['generated_r_executed'] is False
assert summary['sample']['provider_v2_exact_helper_name_matches']==0
print('PASS 18 packet/parity/provenance/parser checks')
