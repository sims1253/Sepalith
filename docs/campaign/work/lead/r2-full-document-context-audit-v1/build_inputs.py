#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
CENSUS=PLAN/'docs/campaign/work/lead/r2-serving-semantic-provider-v3/census-fixtures.json'
LEDGER=Path('/mnt/e/sepalith/campaign-20260915/data-work/Serving-actual-helper-parity-v1/analyzer-shard5-first128/semantic-ledger.jsonl')
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
PINS={'census':'dab7_pending','ledger':'916b45cf97d4e03c3389997d3bb6d7c838918764824e9e1dcc0012ebab7742cf','packets':'948481ce2e9cba79505d5ab20b9ae42812eb945ebe644f60735875a1c169dd9b'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(LEDGER)==PINS['ledger'] and sha(PACKETS)==PINS['packets']
c=json.loads(CENSUS.read_text()); assert c['denominator']==128 and len(c['supported_rows'])==49 and len(c['held_rows'])==79
semantic={x['row_id']:x for x in map(json.loads,LEDGER.read_text().splitlines())}
packets={}
for line in PACKETS.open():
 x=json.loads(line); rid=x.get('row_ref',{}).get('row_id')
 if rid in semantic:packets[rid]=x
rows=[]
for f in c['supported_rows']:
 rid=f['row_id'];s=semantic[rid];p=packets[rid]
 assert s['status']=='semantic_supported_context_closure_root_review_required'
 assert hashlib.sha256(f['preedit_text'].encode()).hexdigest()==f['preedit_sha256']
 raw=Path(f['source_path']).read_bytes(); assert hashlib.sha256(raw).hexdigest()==s['source_sha256']
 parsed=raw if s['target_occurrence_method']=='exact_bytes' else raw.replace(b'\r\n',b'\n')
 assert p['result']['context']['path']==f['path']
 body=p['result']['target_body']; assert body and all(type(x) is str and '\n' not in x and '\r' not in x for x in body)
 rows.append({**f,'target_body':body,'target_sha256':s['target_sha256'],'source_sha256':s['source_sha256'],'source_parse_sha256':hashlib.sha256(parsed).hexdigest(),'source_eol_mode':s['target_occurrence_method'],'helper_spans':s['context_closure']['required_helper_spans'],'package_id':p['row_ref']['package_id'],'group_id':p['row_ref']['group_id'],'split':p['row_ref']['split']})
out={'schema':'sepalith.run06.full_document_context_inputs.v1','denominator':128,'analyzer_supported':49,'analyzer_holds':79,'rows':rows,'input_pins':{'census_fixtures':{'path':str(CENSUS),'sha256':sha(CENSUS)},'semantic_ledger':{'path':str(LEDGER),'sha256':sha(LEDGER)},'candidate_packets':{'path':str(PACKETS),'sha256':sha(PACKETS)}}}
Path(__file__).with_name('inputs.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'rows':len(rows),'helpers':sum(bool(r['helper_spans']) for r in rows),'bytes':sum(len(r['preedit_text'].encode()) for r in rows)}))
