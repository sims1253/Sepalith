#!/usr/bin/env python3
import collections,hashlib,json,statistics,time
from pathlib import Path
P=Path(__file__).parent
TP=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
TSHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(TP)==TSHA
from tokenizers import Tokenizer
t=Tokenizer.from_file(str(TP));t.encode_special_tokens=True
rows=json.loads((P/'rendered.json').read_text())['rows']; started=time.monotonic()
for r in rows:
 p=t.encode(r['prompt_text'],add_special_tokens=False).ids
 y=t.encode(r['target_text'],add_special_tokens=False).ids
 j=t.encode(r['prompt_text']+r['target_text'],add_special_tokens=False).ids
 assert j==p+y
 assert 0 not in p+y and 1 not in p+y
 r['prompt_tokens']=len(p);r['target_tokens']=len(y);r['sequence_tokens']=1+len(p)+len(y)+1
 r['fits_16k']=r['sequence_tokens']<=16384;r['fits_32k']=r['sequence_tokens']<=32768
 r.pop('prompt_text');r.pop('target_text')
seq=[r['sequence_tokens'] for r in rows];prompt=[r['prompt_tokens'] for r in rows];target=[r['target_tokens'] for r in rows]
def quant(x,p):return sorted(x)[min(len(x)-1,int(p*(len(x)-1)))]
def dist(x):return {'min':min(x),'median':statistics.median(x),'p90':quant(x,.90),'p95':quant(x,.95),'max':max(x),'mean':statistics.mean(x)}
helpers=[r for r in rows if r['helper_spans']]
out={'schema':'sepalith.run06.full_document_token_audit.v1','tokenizer':{'path':str(TP),'sha256':TSHA,'bos':0,'eos':1,'policy':'no tokenizer-added specials; one BOS and EOS'},'denominators':{'frozen_census':128,'analyzer_supported':49,'analyzer_holds':79,'supported_with_helpers':8},'counts':{'fits_16k':sum(r['fits_16k'] for r in rows),'fits_32k':sum(r['fits_32k'] for r in rows),'over_32k':sum(not r['fits_32k'] for r in rows),'helper_fits_16k':sum(r['fits_16k'] for r in helpers),'helper_fits_32k':sum(r['fits_32k'] for r in helpers)},'distributions':{'prompt_tokens':dist(prompt),'target_tokens':dist(target),'sequence_tokens':dist(seq)},'elapsed_seconds':time.monotonic()-started,'rows':rows}
(P/'token-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'counts':out['counts'],'sequence':out['distributions']['sequence_tokens'],'target':out['distributions']['target_tokens'],'longest':[(r['row_id'],r['sequence_tokens'],r['target_tokens'],bool(r['helper_spans'])) for r in sorted(rows,key=lambda x:x['sequence_tokens'],reverse=True)[:8]]},indent=2))
