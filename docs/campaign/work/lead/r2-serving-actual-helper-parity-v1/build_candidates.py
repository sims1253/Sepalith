#!/usr/bin/env python3
"""Build TRAIN-only pre-edit fixtures from independently accepted analyzer-v6 rows."""
import hashlib,json
from pathlib import Path

RID_LEDGER=Path('/mnt/e/sepalith/campaign-20260915/data-work/Serving-actual-helper-parity-v1/analyzer-shard5-first128/semantic-ledger.jsonl')
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
semantic=[json.loads(x) for x in RID_LEDGER.read_text().splitlines()]
selected={x['row_id']:x for x in semantic if x['status']=='semantic_supported_context_closure_root_review_required' and x['context_closure']['required_helper_spans']}
packets={}
with PACKETS.open() as f:
 for line in f:
  x=json.loads(line);rid=x.get('row_ref',{}).get('row_id')
  if rid in selected:packets[rid]=x
assert set(packets)==set(selected)
out=[]
for rid,s in sorted(selected.items()):
 p=packets[rid];raw=Path(s['source_path']).read_bytes();assert sha(raw)==s['source_sha256']
 parse=raw if s['target_occurrence_method']=='exact_bytes' else raw.replace(b'\r\n',b'\n')
 target=('\n'.join(p['result']['target_body'])+'\n').encode();assert parse.count(target)==1
 start=parse.index(target);anchor=parse[:start].count(b'\n');before=parse[:start]+b'\n'+parse[start+len(target):]
 removed_lines=len(p['result']['target_body'])-1
 def adjusted(span):
  a,b=span
  if b-1<anchor:return [a-1,b-1]
  if a-1>anchor:return [a-1-removed_lines,b-1-removed_lines]
  raise AssertionError('span overlaps removed target')
 helpers=[{'name':h['name'],'analyzer_span_1based':h['span'],'provider_span_0based':adjusted(h['span']),'content':('\n'.join(parse.decode().split('\n')[h['span'][0]-1:h['span'][1]]))} for h in s['context_closure']['required_helper_spans']]
 target_span=adjusted(s['context_closure']['target_definition_span'])
 assert before.decode().split('\n')[anchor]==''
 out.append({'row_id':rid,'path':p['result']['context']['path'],'source_path':s['source_path'],'source_after_sha256':s['source_sha256'],'source_parse_sha256':sha(parse),'preedit_sha256':sha(before),'preedit_text':before.decode(),'target_sha256':s['target_sha256'],'target_lines':p['result']['target_body'],'target_name':s['target_definition_name'],'cursor':{'line':anchor,'character':0},'target_span_0based':target_span,'helpers':helpers,'provenance':{'group_id':p['row_ref']['group_id'],'package_id':p['row_ref']['package_id'],'raw_line_sha256':p['row_ref']['raw_line_sha256'],'split':p['row_ref']['split']}})
Path(__file__).with_name('candidate-fixtures.json').write_text(json.dumps({'schema':'sepalith.run06.actual_helper_candidates.v1','rows':out},indent=2,sort_keys=True)+'\n')
print(json.dumps({'rows':len(out),'ids':sorted(selected)}))
