#!/usr/bin/env python3
import hashlib,importlib.util,json,sys
from pathlib import Path
RID='227626e3a7a234b808a096b1';HERE=Path(__file__).parent;MAT=HERE/'materialize_semantic.py'
def sha(b):return hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('mat_v3_parity',MAT);m=importlib.util.module_from_spec(spec);sys.modules['mat_v3_parity']=m;spec.loader.exec_module(m)
def get(path):
 with path.open() as stream:
  for line in stream:
   row=json.loads(line)
   if (row.get('row_id') or row.get('row_ref',{}).get('row_id'))==RID:return row
 raise KeyError(RID)
sem=get(Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-roxy-semantic-v6/bounded-root-01/semantic-ledger.jsonl'));packet=get(Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1/shard-0005/structured-materialization-v1/candidate-packets.jsonl'));raw=Path(sem['source_path']).read_bytes();lines=raw.replace(b'\r\n',b'\n').decode().split('\n');span=m.checked_span(sem['context_closure']['target_definition_span'],len(lines),'target');base=m.PROTOCOL.PromptContext.from_mapping(packet['result']['context']);target=packet['result']['target_body'];selected,before,derived,selection=m._production_context(base,raw,target,span);function_end=span[1]-len(target)
out={'schema':'sepalith.dat10.materializer_v3_actual_serving_parity_fixture.v1','row_id':RID,'source_after':raw.decode(),'source_after_sha256':sha(raw),'source_before':before,'source_before_sha256':sha(before.encode()),'target_lines':target,'required_source_span':{'startLine':derived['target_start_line'],'endLine':function_end},'max_source_utf16_units':m.SELECTION.DEFAULT_SELECTION_UTF16_UNITS,'python_context':selected.to_dict(),'python_prompt':m.PROTOCOL.render_prompt(selected),'selection':selection.to_dict(),'materializer_sha256':m.sha(MAT),'training_admission':False}
(HERE/'actual-parity-fixture.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'line':derived['target_start_line'],'selected_full':not selection.omissions,'prompt_sha':sha(out['python_prompt'].encode())}))
