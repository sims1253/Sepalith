#!/usr/bin/env python3
import collections,hashlib,json,os
from pathlib import Path
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1');WORK=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic115-hold-retrospective-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyed(p):return {x['row_id']:x for x in map(json.loads,Path(p).open())}
assert sha(BASE/'selected-01/selected-contexts.jsonl')=='3729541c9edc0eb82eb217bf4a28c6a5a8a295676c886e41277abf18fb0e98e8'
sel=keyed(BASE/'selected-01/selected-contexts.jsonl');rep={x['row_id']:x for x in json.loads((WORK/'replay-manifest.json').read_text())['results']};scope={x['row_id']:x for x in json.loads((WORK/'scoped-replay-manifest.json').read_text())['results']};rows=[];counts=collections.Counter()
for rid,x in sorted(sel.items()):
 if x['status']=='supported':continue
 reasons=x.get('reasons') or []
 if any(v.startswith('runtime_dependency_missing_reviewed_name:') for v in reasons):kind='masked_infrastructure_at_risk';resolution='v2_replay_evidence_hold_no_infrastructure';detail={'v2':rep[rid]['result'],'scoped_counterfactual':scope[rid]['result']}
 elif reasons==['mixed source EOL is rejected by the frozen parse-only provider']:kind='mixed_eol_explicit';resolution='retained_repair_queue';detail=None
 elif any(v.startswith('namespace_status:hold_namespace_conditional_directives') for v in reasons):kind='namespace_conditional_directives';resolution='retained_repair_queue';detail=None
 elif any(v.startswith('namespace_status:hold_namespace_parse_error') for v in reasons):kind='namespace_parse_error';resolution='retained_repair_queue';detail=None
 elif any(v.startswith('namespace_status:hold_namespace_malformed') for v in reasons):kind='namespace_malformed';resolution='retained_repair_queue';detail=None
 elif any(v.startswith('multiple_explicit_origins:') for v in reasons):kind='multiple_explicit_origins';resolution='retained_repair_queue';detail=None
 elif 'code_between_anchor_and_target_function' in reasons:kind='anchor_target_code';resolution='retained_repair_queue';detail=None
 elif 'complete_span_unresolved' in reasons:kind='complete_span_unresolved';resolution='retained_repair_queue';detail=None
 else:raise ValueError('unclassified:'+rid)
 counts[kind]+=1;rows.append({'row_id':rid,'class':kind,'prior_reasons':reasons,'resolution':resolution,'v2_reprocessed':rid in rep,'v2_infrastructure_failure':rep[rid]['infrastructure_failure'] if rid in rep else None,'v2_supported':(rep[rid]['result'] or {}).get('status')=='supported' if rid in rep else None,'details':detail,'training_admission':False})
assert len(rows)==115 and sum(counts.values())==115 and sum(x['v2_reprocessed'] for x in rows)==3 and not any(x['v2_infrastructure_failure'] for x in rows if x['v2_reprocessed']) and not any(x['v2_supported'] for x in rows if x['v2_reprocessed'])
out=WORK/'classification-ledger.jsonl'
with out.open('x') as f:
 for x in rows:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 f.flush();os.fsync(f.fileno())
m={'schema':'sepalith.dat10.semantic115.hold_retrospective.v1','status':'complete_review_only','denominator':115,'class_counts':dict(counts),'v2_reprocessed':3,'v2_infrastructure_failures':0,'v2_supported':0,'retained_repair_queue':115,'supported4435_preserved':True,'inputs':{'selected_sha256':'3729541c9edc0eb82eb217bf4a28c6a5a8a295676c886e41277abf18fb0e98e8','decision_ledger_sha256':'4a472985cfe9756998bed9b188caf9ff0a27adf868a9a6d80644dfec2e75525a','provider_v2_wrapper_sha256':'352a2976b235d5604af9fe2a931208fda53728a53d3cf027583aa4e237d2fcd5'},'output':{'path':str(out),'rows':115,'bytes':out.stat().st_size,'sha256':sha(out)},'interpretation':'The only three rows whose old expected-name precedence could mask infrastructure were replayed. All three completed without infrastructure failure and remained evidence holds. No existing supported row changed. All 115 stay named repair queues, not permanent exclusions.','target_or_gold_used':False,'training_admission':False}
with (WORK/'manifest.json').open('x') as f:json.dump(m,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(m,sort_keys=True))
