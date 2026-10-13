#!/usr/bin/env python3
import hashlib,json,subprocess,tempfile
from pathlib import Path
P=Path(__file__).parent;RID='25189e2eff136821189db4db'
PROV=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-independent-replay-v3/full-01/shards/shard-0005/ledger.jsonl');PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
def line(path,pred):
 for raw in path.open('rb'):
  if pred(json.loads(raw)):return raw
 raise RuntimeError('missing row')
prov=line(PROV,lambda x:x.get('row_id')==RID);packet=line(PACKETS,lambda x:x.get('row_ref',{}).get('row_id')==RID)
with tempfile.TemporaryDirectory() as d:
 d=Path(d);a=d/'prov.jsonl';b=d/'packet.jsonl';o=d/'out';a.write_bytes(prov);b.write_bytes(packet);digest=hashlib.sha256(packet).hexdigest()
 done=subprocess.run(['python3','-B',str(P/'source/analyze_semantics.py'),'--provenance-ledger',str(a),'--candidate-packets',str(b),'--expected-candidate-packets-sha256',digest,'--output',str(o)],capture_output=True,text=True,timeout=60)
 if done.returncode:raise RuntimeError(done.stderr)
 row=json.loads((o/'semantic-ledger.jsonl').read_text());manifest=json.loads((o/'manifest.json').read_text())
 evidence={'schema':'sepalith.dat10.semantic_bytehash_exact_pipeline.v1','row_id':RID,'exit_code':done.returncode,'status':row['status'],'reasons':row['reasons'],'scope_status':row['scope']['status'],'parsed_source_sha256':row['scope']['parsed_source_sha256'],'expected_parse_bytes_sha256':row['parse_bytes_sha256'],'hash_exact':row['scope']['parsed_source_sha256']==row['parse_bytes_sha256'],'target_matches':row['scope']['target_matches'],'manifest_rows':manifest['rows'],'exact_id_closure':manifest['exact_id_closure'],'source_or_generated_r_executed':False}
(P/'exact-pipeline-result.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');print(json.dumps(evidence))
