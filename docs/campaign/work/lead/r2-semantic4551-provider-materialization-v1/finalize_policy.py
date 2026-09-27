#!/usr/bin/env python3
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path('/mnt/e/sepalith/campaign-20260915/data-work/Semantic4551-provider-materialization-v1');R32=ROOT/'render-04';I16=ROOT/'render-16k-input';R16=ROOT/'render-16k';OUT=ROOT/'selected-01'
DIAG=ROOT/'diagnostic-row-01/diagnosis-supplement.json';DIAG_SHA='620ce2d0ca072404faa7a19fac5b730214290d1720cbd9d42b5d2b638ac47220';DIAG_ID='4091f84e84cb86a1f6311985'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path):
 out={}
 for line in path.read_text().splitlines():
  x=json.loads(line);rid=x['row_id']
  if rid in out:raise ValueError('duplicate:'+rid)
  out[rid]=x
 return out
r32={}
for s in ('0000','0001'):r32.update(load(R32/f'shard-{s}.jsonl'))
m16=json.loads((I16/'manifest.json').read_text());r16={}
for s in ('0000','0001'):
 p=R16/f'shard-{s}.jsonl';t=json.loads((R16/f'shard-{s}.terminal.json').read_text())
 if t['status']!='complete' or t['output']['sha256']!=sha(p):raise ValueError('16k terminal:'+s)
 r16.update(load(p))
expected=set()
for s in ('0000','0001'):expected|=set(load(I16/f'shard-{s}.jsonl'))
if set(r16)!=expected or len(r32)!=4551:raise ValueError('render closure')
if sha(DIAG)!=DIAG_SHA:raise ValueError('mixed-EOL diagnosis pin')
diagnosis=json.loads(DIAG.read_text());
if diagnosis['row_id']!=DIAG_ID or diagnosis['status']!='explicit_hold':raise ValueError('mixed-EOL diagnosis binding')
if OUT.exists():raise ValueError('fresh selected output');OUT.mkdir(parents=True)
OUT.mkdir(parents=True);tmp=OUT/'selected-contexts.jsonl.tmp';counts={};selected=[]
for rid in sorted(r32):
 a=r32[rid]
 if rid==DIAG_ID:
  if a['status']=='supported':raise ValueError('diagnosed row unexpectedly supported')
  x={**a,'reason':'mixed_eol_source_provider_unsupported','reasons':['mixed source EOL is rejected by the frozen parse-only provider'],'diagnosis_sha256':DIAG_SHA};reason='mixed_eol_explicit_hold'
 elif a['status']!='supported':x=a;reason='hold_32k'
 elif rid not in r16:x={**a,'context_size':16384};reason='reuse_full_document_16k'
 elif r16[rid]['status']=='supported':x=r16[rid];reason='selected_16k_rerun'
 else:x=a;reason='fallback_32k_after_16k_hold'
 x={**x,'policy_resolution':reason};counts[reason]=counts.get(reason,0)+1;selected.append(x)
with tmp.open('x') as f:
 for x in selected:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 f.flush();os.fsync(f.fileno())
tmp.replace(OUT/'selected-contexts.jsonl')
m={'schema':'sepalith.dat10.semantic4551.context_policy.v1','status':'complete_review_only','denominator':4551,'counts':counts,'supported':sum(x['status']=='supported' for x in selected),'holds':sum(x['status']!='supported' for x in selected),'modes':dict(__import__('collections').Counter(x.get('mode') for x in selected if x['status']=='supported')),'contexts':dict(__import__('collections').Counter(str(x.get('context_size')) for x in selected if x['status']=='supported')),'output':{'path':str(OUT/'selected-contexts.jsonl'),'sha256':sha(OUT/'selected-contexts.jsonl'),'bytes':(OUT/'selected-contexts.jsonl').stat().st_size,'rows':len(selected)},'target_used':False}
(OUT/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');print(json.dumps(m,sort_keys=True))
