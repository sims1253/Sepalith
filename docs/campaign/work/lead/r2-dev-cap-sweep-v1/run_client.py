import json,pathlib,hashlib,collections,time
import probe
w=pathlib.Path(__file__).resolve().parent
rows=json.loads((w/'dev-cases.json').read_text())
results=[]
with (w/'requests.jsonl').open('x') as f:
 for row in rows:
  caps=sorted([192,384,512],key=lambda c:hashlib.sha256((row['id']+str(c)).encode()).hexdigest())
  for cap in caps:
   record=probe.run_case('http://127.0.0.1:18414',row,'cold',1,cap,4096,5000)
   parsed=record.get('parsed_output') or {};valid=record['protocol_status']=='accepted'
   record.update(family=row['family'],package_id=row['package_id'],expected_noop=row['target_operation']=='no_op',exact_body=valid and parsed.get('body_text')==row['target_body_text'],correct_noop=valid and row['target_operation']=='no_op' and parsed.get('operation')=='no_op',false_suggestion=valid and row['target_operation']=='no_op' and parsed.get('operation')!='no_op')
   f.write(json.dumps(record)+'\n');f.flush();results.append(record)
summary={}
for cap in [192,384,512]:
 selected=[x for x in results if x['cap']==cap]
 summary[cap]={family:{'cases':len(rr),'protocol_valid':sum(x['protocol_status']=='accepted' for x in rr),'exact_body':sum(x['exact_body'] for x in rr),'correct_noop':sum(x['correct_noop'] for x in rr),'false_suggestions':sum(x['false_suggestion'] for x in rr),'timeouts':sum(x['protocol_status']=='timeout' for x in rr)} for family in sorted({x['family'] for x in selected}) for rr in [[x for x in selected if x['family']==family]]}
(w/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'requests':len(results),'summary':summary}))
