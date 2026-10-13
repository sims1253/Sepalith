#!/usr/bin/env python3
import hashlib, json, os, tempfile
from pathlib import Path

DECISIONS=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic-external-holds-audit-v1/decisions.jsonl')
PREDICTION=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic763-context-admission-v1/prepared-02/prediction-inputs.jsonl')
PINS={DECISIONS:'053d50854a5faea3feddb7cf2c62a4f47190075fb156cc1898a55cec8e6414b6',PREDICTION:'233c39c62517b6468c3335bdc0bc2e6352aed8c0359da4f9f585874ebecf1cb1'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in PINS.items():
 if sha(p)!=h:raise ValueError('input pin differs:'+str(p))
dec={x['row_id']:x for x in map(json.loads,DECISIONS.read_text().splitlines()) if x['status']=='candidate_namespace_origin_evidence'}
pred={x['row_id']:x for x in map(json.loads,PREDICTION.read_text().splitlines())}
if len(dec)!=133 or not set(dec)<=set(pred):raise ValueError('candidate closure differs')
out=Path(__file__).with_name('prediction-batch.jsonl');fd,tmp=tempfile.mkstemp(prefix='.'+out.name+'.',dir=out.parent)
with os.fdopen(fd,'w') as f:
 for rid in sorted(dec):
  d,p=dec[rid],pred[rid];ns=Path(d['namespace']['path'])
  row={'row_id':rid,'path':p['path'],'absolute_document_path':d['source']['path'],'workspace_root':str(ns.parent),'preedit_text':p['preedit_text'],'preedit_sha256':p['preedit_sha256'],'cursor':p['cursor'],'document_eol':p['document_eol'],'expected_dependencies':d['dependencies'],'expected_namespace_sha256':d['namespace']['sha256']}
  if any(any(w in k.lower() for w in ('target','gold','reward','completion')) for k in row):raise ValueError('target-shaped field')
  f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
 f.flush();os.fsync(f.fileno())
os.replace(tmp,out);d=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
print(json.dumps({'rows':len(dec),'sha256':sha(out),'output':str(out)},sort_keys=True))
