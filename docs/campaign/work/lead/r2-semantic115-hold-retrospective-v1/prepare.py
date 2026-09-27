#!/usr/bin/env python3
import collections,hashlib,json,os
from pathlib import Path
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1');OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic115-hold-retrospective-v1')
PINS={'selected':'3729541c9edc0eb82eb217bf4a28c6a5a8a295676c886e41277abf18fb0e98e8','ledger':'4a472985cfe9756998bed9b188caf9ff0a27adf868a9a6d80644dfec2e75525a','shard0':'f3ce8e522c524c399af4f6718eb57ac473ab8511806166a99725dd67148442c8','shard1':'d4b874cf3c877174b078ed9a59d4bb857fab241b5b72de98cd027a6e4302e60d'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyed(p):
 out={}
 for line in Path(p).open():
  x=json.loads(line);assert x['row_id'] not in out;out[x['row_id']]=x
 return out
sel=keyed(BASE/'selected-01/selected-contexts.jsonl');ledger=keyed(BASE/'final-01/decision-ledger.jsonl');assert sha(BASE/'selected-01/selected-contexts.jsonl')==PINS['selected'] and sha(BASE/'final-01/decision-ledger.jsonl')==PINS['ledger']
holds={rid:x for rid,x in sel.items() if x['status']!='supported'};assert len(holds)==115 and all(ledger[x]['status']=='hold' for x in holds)
classes=collections.Counter();at_risk=[]
for rid,x in holds.items():
 reasons=x.get('reasons') or []
 if any(v.startswith('runtime_dependency_missing_reviewed_name:') for v in reasons):kind='masked_infrastructure_at_risk';at_risk.append(rid)
 elif reasons==['mixed source EOL is rejected by the frozen parse-only provider']:kind='mixed_eol_explicit'
 elif any(v.startswith('namespace_status:hold_namespace_conditional_directives') for v in reasons):kind='namespace_conditional_directives'
 elif any(v.startswith('namespace_status:hold_namespace_parse_error') for v in reasons):kind='namespace_parse_error'
 elif any(v.startswith('namespace_status:hold_namespace_malformed') for v in reasons):kind='namespace_malformed'
 elif any(v.startswith('multiple_explicit_origins:') for v in reasons):kind='multiple_explicit_origins'
 elif 'code_between_anchor_and_target_function' in reasons:kind='anchor_target_code'
 elif 'complete_span_unresolved' in reasons:kind='complete_span_unresolved'
 else:kind='other_evidence_hold'
 classes[kind]+=1
assert len(at_risk)==3 and sum(classes.values())==115
inputs={}
for i,pin in [(0,PINS['shard0']),(1,PINS['shard1'])]:
 p=BASE/f'input-shards/shard-{i:04d}.jsonl';assert sha(p)==pin
 for line in p.open():
  x=json.loads(line)
  if x['row_id'] in at_risk:inputs[x['row_id']]=x
assert set(inputs)==set(at_risk)
(OUT/'replay-inputs').mkdir(parents=True,exist_ok=False)
entries=[]
for rid in sorted(at_risk):
 p=OUT/'replay-inputs'/f'{rid}.jsonl'
 with p.open('x') as f:f.write(json.dumps(inputs[rid],sort_keys=True,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 entries.append({'row_id':rid,'path':str(p),'sha256':sha(p),'prior_reasons':holds[rid]['reasons']})
m={'schema':'sepalith.dat10.semantic115.retrospective_input.v1','status':'complete_target_free','denominator':115,'class_counts':dict(classes),'masked_infrastructure_at_risk':entries,'supported4435_preserved':True,'training_admission':False}
with (OUT/'preparation-manifest.json').open('x') as f:json.dump(m,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(m,sort_keys=True))
