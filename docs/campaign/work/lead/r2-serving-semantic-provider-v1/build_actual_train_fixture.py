#!/usr/bin/env python3
import hashlib,json
from pathlib import Path

RID='227626e3a7a234b808a096b1'
SEMANTIC=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-roxy-semantic-v6/bounded-root-01/semantic-ledger.jsonl')
PACKETS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
EXPECTED_SOURCE='75b80a9d2f56e86a4264f4d786e4c7bbd777bb8b8a42b40a1077ff34232a1c22'

def one(path,key):
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if (row.get('row_id') or row.get('row_ref',{}).get('row_id'))==key:return row
    raise KeyError(key)

s=one(SEMANTIC,RID);p=one(PACKETS,RID);source=Path(s['source_path']).read_bytes()
assert hashlib.sha256(source).hexdigest()==EXPECTED_SOURCE
target=('\n'.join(p['result']['target_body'])+'\n').encode()
assert source.count(target)==1
before=source.replace(target,b'\n')
line=source[:source.index(target)].count(b'\n')
assert line==290
out={
 'schema':'sepalith.run06.semantic_provider_train_fixture.v1','row_id':RID,
 'source_after_sha256':EXPECTED_SOURCE,'preedit_sha256':hashlib.sha256(before).hexdigest(),
 'preedit_text':before.decode(),'cursor':{'line':line,'character':0},
 'path':'R/bios_fast.R','target_name':s['target_definition_name'],
 'expected_target_span':{
   'startLine':s['scope']['target_definition_span'][0]-1-len(p['result']['target_body'])+1,
   'endLine':s['scope']['target_definition_span'][1]-1-len(p['result']['target_body'])+1,
 },
 'semantic_v6_required_helpers':[x['name'] for x in s['context_closure']['required_helper_spans']],
 'contains_expected_target':False,
}
Path(__file__).with_name('actual-train-fixture.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
