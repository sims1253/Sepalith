#!/usr/bin/env python3
import hashlib,importlib.util,json,subprocess,tempfile
from pathlib import Path
P=Path(__file__).parent;RID='25189e2eff136821189db4db'
PROV=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-0005/ledger.jsonl')
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
def row(path,getid):
 for line in path.open():
  x=json.loads(line)
  if getid(x)==RID:return x
 raise RuntimeError('row absent')
prov=row(PROV,lambda x:x.get('row_id'));packet=row(PACKETS,lambda x:x.get('row_ref',{}).get('row_id'))
source=Path(packet['validation']['source_path']);raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==packet['validation']['source_sha256']
target=('\n'.join(packet['result']['target_body'])+'\n').encode();exact=raw.count(target);parse=raw if exact else raw.replace(b'\r\n',b'\n');assert parse.count(target)==1
name='summary.qbrms_p_significance'
with tempfile.TemporaryDirectory() as d:
 d=Path(d);copy=d/'source.R';copy.write_bytes(parse);inp=d/'input.json';out=d/'out.jsonl';inp.write_text(json.dumps({'source_groups':[{'source_path':str(copy),'source_sha256':hashlib.sha256(parse).hexdigest(),'rows':[{'row_id':RID,'target_definition_name':name}]}]}))
 done=subprocess.run(['Rscript','--vanilla',str(P/'source/semantic_scope.R'),str(inp),str(out)],capture_output=True,text=True,timeout=30)
 result=json.loads(out.read_text())
evidence={'schema':'sepalith.dat10.semantic_bytehash_exact_reproduction.v1','row_id':RID,'provenance':{'ledger':str(PROV),'ledger_sha256':hashlib.sha256(PROV.read_bytes()).hexdigest(),'status':prov['status'],'source_stat_stable':prov['source_stat_stable']},'candidate_packet':{'path':str(PACKETS),'sha256':hashlib.sha256(PACKETS.read_bytes()).hexdigest(),'raw_line_sha256':packet['row_ref']['raw_line_sha256']},'source':{'path':str(source),'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'parse_sha256':hashlib.sha256(parse).hexdigest(),'parse_bytes':len(parse),'target_occurrences':parse.count(target),'target_name':name},'r':{'exit_code':done.returncode,'result':result},'diagnosis':{'actual_hash_mismatch':result.get('parsed_source_sha256') not in (None,hashlib.sha256(parse).hexdigest()),'hash_field_missing':result.get('parsed_source_sha256') is None,'semantic_condition':'duplicate target definition','target_matches':result.get('target_matches')},'generated_or_source_r_executed':False}
(P/'exact-row-reproduction.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');print(json.dumps(evidence['diagnosis']))
