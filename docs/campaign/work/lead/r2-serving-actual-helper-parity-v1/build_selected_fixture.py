#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
RID='227d18cf9b5ced4cba29fd2c'
here=Path(__file__).parent
all_rows=json.loads((here/'candidate-fixtures.json').read_text())['rows']
row=next(x for x in all_rows if x['row_id']==RID)
target='\n'.join(row['target_lines'])
lines=row['preedit_text'].split('\n');assert lines[row['cursor']['line']]==''
lines[row['cursor']['line']]=target
row['source_after_text']='\n'.join(lines)
assert hashlib.sha256(row['source_after_text'].encode()).hexdigest()==row['source_parse_sha256']
row['semantic_v6_status']='semantic_supported_context_closure_root_review_required'
row['semantic_v6_analyzer_sha256']='54965cb3af14a125d945d94f24975eddff040b0706ab934b5a04543a12a99a57'
(here/'actual-helper-fixture.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
print(json.dumps({'row_id':RID,'helpers':[x['name'] for x in row['helpers']],'bytes':(here/'actual-helper-fixture.json').stat().st_size}))
