#!/usr/bin/env python3
"""Build all analyzer-supported fixtures in the frozen first-128 TRAIN census."""
import hashlib, json
from pathlib import Path
LEDGER=Path('/mnt/e/sepalith/campaign-20260915/data-work/Serving-actual-helper-parity-v1/analyzer-shard5-first128/semantic-ledger.jsonl')
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
def sha(b): return hashlib.sha256(b).hexdigest()
ledger=[json.loads(x) for x in LEDGER.read_text().splitlines()]
assert len(ledger)==128 and len({x['row_id'] for x in ledger})==128
supported={x['row_id']:x for x in ledger if x['status']=='semantic_supported_context_closure_root_review_required'}
packets={}
with PACKETS.open() as f:
    for line in f:
        x=json.loads(line); rid=x.get('row_ref',{}).get('row_id')
        if rid in supported: packets[rid]=x
assert set(packets)==set(supported)
out=[]
for rid,s in sorted(supported.items()):
    p=packets[rid]; raw=Path(s['source_path']).read_bytes(); assert sha(raw)==s['source_sha256']
    parse=raw if s['target_occurrence_method']=='exact_bytes' else raw.replace(b'\r\n',b'\n')
    target=('\n'.join(p['result']['target_body'])+'\n').encode(); assert parse.count(target)==1
    start=parse.index(target); anchor=parse[:start].count(b'\n'); before=parse[:start]+b'\n'+parse[start+len(target):]
    removed_lines=len(p['result']['target_body'])-1
    def adjusted(span):
        a,b=span
        if b-1<anchor:return [a-1,b-1]
        if a-1>anchor:return [a-1-removed_lines,b-1-removed_lines]
        raise AssertionError(f'{rid}: span overlaps removed target')
    helpers=[]
    for h in s['context_closure']['required_helper_spans']:
        helpers.append({'name':h['name'],'provider_span_0based':adjusted(h['span'])})
    out.append({'row_id':rid,'path':p['result']['context']['path'],'source_path':s['source_path'],'preedit_sha256':sha(before),'preedit_text':before.decode(),'cursor':{'line':anchor,'character':0},'target_name':s['target_definition_name'],'target_span_0based':adjusted(s['context_closure']['target_definition_span']),'analyzer_helpers':sorted(h['name'] for h in s['context_closure']['required_helper_spans'])})
holds=[{'row_id':x['row_id'],'reasons':x.get('reasons',[])} for x in ledger if x['status']=='hold_semantic_evidence']
Path(__file__).with_name('census-fixtures.json').write_text(json.dumps({'schema':'sepalith.run06.semantic_provider_census_fixtures.v1','denominator':128,'supported_rows':out,'held_rows':holds},indent=2,sort_keys=True)+'\n')
print(json.dumps({'denominator':len(ledger),'supported':len(out),'held':len(holds),'supported_with_helpers':sum(bool(x['analyzer_helpers']) for x in out)}))
