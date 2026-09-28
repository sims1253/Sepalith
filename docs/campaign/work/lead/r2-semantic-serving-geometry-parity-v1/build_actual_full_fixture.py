#!/usr/bin/env python3
"""Build one review-only actual TRAIN full-source geometry fixture."""
import hashlib,importlib.util,json,sys
from pathlib import Path
RID='227626e3a7a234b808a096b1';PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
MAT=PLAN/'docs/campaign/work/lead/r2-sourcewalk-semantic-materialization-v2/materialize_semantic.py';MAT_SHA='f05708b5bc94819ed9545956774c1ef1899be8e7d6c7fdf36667d7173a8aa8dc'
SEM=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-roxy-semantic-v6/bounded-root-01/semantic-ledger.jsonl');PACK=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl')
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(MAT.read_bytes())==MAT_SHA
spec=importlib.util.spec_from_file_location('mat_v2_actual_full_fixture',MAT);m=importlib.util.module_from_spec(spec);sys.modules['mat_v2_actual_full_fixture']=m;spec.loader.exec_module(m)
def get(path):
 for line in path.open():
  row=json.loads(line)
  if row.get('row_id')==RID or row.get('row_ref',{}).get('row_id')==RID:return row
 raise KeyError(RID)
semantic,packet=get(SEM),get(PACK);raw=Path(semantic['source_path']).read_bytes();base=m.PROTOCOL.PromptContext.from_mapping(packet['result']['context']);span=m.checked_span(semantic['context_closure']['target_definition_span'],len(raw.replace(b'\r\n',b'\n').decode().split('\n')),'target')
selected,before,derived=m._expanded_context(base,raw,list(packet['result']['target_body']),span)
out={'schema':'sepalith.run06.actual_train_full_source_fixture.v1','row_id':RID,'materializer_v2_sha256':MAT_SHA,'source_after_sha256':sha(raw),'source_after':raw.decode(),'source_before':before,'source_before_sha256':sha(before.encode()),'target_lines':packet['result']['target_body'],'global_insertion_line':derived['target_start_line'],'materializer_context':selected.to_dict(),'training_admission':False}
Path(__file__).with_name('actual-full-fixture.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'row_id':RID,'line':derived['target_start_line'],'before_sha256':out['source_before_sha256']}))
