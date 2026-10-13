import json,sys,hashlib
from pathlib import Path

def summarize(run):
 arms=['ordinary-baseline','model-free-ngram','released-dspark','trained-dspark'];data={a:json.loads((run/(a+'.json')).read_text()) for a in arms};indexed={};out={}
 for a,d in data.items():
  rows=d.get('requests',[]);m={(r['row_id'],r['phase'],r['rep']):r for r in rows}
  if len(rows)!=80 or len(m)!=80:raise ValueError('expected80unique requests:'+a)
  if any(not isinstance(r.get('returned_token_ids'),list) or not isinstance(r.get('raw_text'),str) for r in rows):raise ValueError('missing actual response:'+a)
  indexed[a]=m;out[a]={'summary':d['summary'],'response_sha256':hashlib.sha256((run/(a+'.json')).read_bytes()).hexdigest()}
 base=indexed['ordinary-baseline']
 for a in arms[1:]:
  if indexed[a].keys()!=base.keys():raise ValueError('panel mismatch:'+a)
  mismatches=[]
  for k,b in base.items():
   c=indexed[a][k]
   if b['returned_token_ids']!=c['returned_token_ids'] or b['raw_text']!=c['raw_text'] or b.get('protocol_status')!=c.get('protocol_status'):
    mismatches.append({'row_id':k[0],'phase':k[1],'rep':k[2],'token_parity':b['returned_token_ids']==c['returned_token_ids'],'text_parity':b['raw_text']==c['raw_text'],'baseline_protocol':b.get('protocol_status'),'candidate_protocol':c.get('protocol_status')})
  out[a]['paired_comparison']={'requests':80,'exact_token_text_protocol_parity':not mismatches,'mismatches':mismatches}
 return {'status':'measurement_complete','arms':out,'limits':['40 TRAIN rows selected with target cap; not held-out quality','Serial cold/warm local HTTP requests; not editor/LAN latency','Natural prompts1558..2765 under4096context; no8Kstress','Root must review protocol failures and denominators before promotion']}
if __name__=='__main__':
 run=Path(sys.argv[1]);out=summarize(run);(run/'root-paired-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
