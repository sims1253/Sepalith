#!/usr/bin/env python3
import hashlib,json,statistics
from pathlib import Path
P=Path(__file__).parent; TP=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');assert hashlib.sha256(TP.read_bytes()).hexdigest()=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
from tokenizers import Tokenizer
t=Tokenizer.from_file(str(TP)); full={r['row_id']:r for r in json.load(open(P/'token-audit.json'))['rows']};p=json.load(open(P/'provider-comparison-prompts.json'))['rows'];rows=[]
for r in p:
 if r['status']!='supported':continue
 n=len(t.encode(r['prompt_text'],add_special_tokens=False).ids); f=full[r['row_id']]['prompt_tokens']; rows.append({'row_id':r['row_id'],'provider_v3_prompt_tokens':n,'full_document_prompt_tokens':f,'full_over_v3_tokens':f-n,'ratio':f/n})
vals=[r['full_over_v3_tokens'] for r in rows];rat=[r['ratio'] for r in rows]
out={'schema':'sepalith.run06.full_document_token_tradeoff.v1','denominator_provider_supported':len(rows),'full_over_v3_tokens':{'min':min(vals),'median':statistics.median(vals),'max':max(vals),'mean':statistics.mean(vals)},'full_to_v3_ratio':{'min':min(rat),'median':statistics.median(rat),'max':max(rat),'mean':statistics.mean(rat)},'rows':rows};(P/'token-tradeoff.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
