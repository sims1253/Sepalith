#!/usr/bin/env python3
import hashlib,json,statistics
from pathlib import Path
OUT=Path(__file__).parent;AUD=OUT.parent/'r2-full-document-context-audit-v1';TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json');TSHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81';assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()==TSHA
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(TOKENIZER));inp=json.load(open(AUD/'inputs.json'));full={r['row_id']:r for r in json.load(open(AUD/'rendered.json'))['rows']};bounded={r['row_id']:r for r in json.load(open(OUT/'candidate-prompts.json'))['rows']};rows=[];counts={}
for row in inp['rows']:
 rid=row['row_id'];f=full[rid];b=bounded[rid];ft=len(tok.encode(f['prompt_text'],add_special_tokens=False).ids);assert ft==next(x for x in json.load(open(AUD/'token-audit.json'))['rows'] if x['row_id']==rid)['prompt_tokens']
 bt=len(tok.encode(b['bounded_prompt_text'],add_special_tokens=False).ids) if b['bounded_status']=='supported' else None
 expected={}
 for cap in (16384,32768):
  if 1+ft+1024<=cap:mode='full_document';pt=ft;ph=f['prompt_sha256'];reason=[]
  elif b['bounded_status']=='supported' and 1+bt+1024<=cap:mode='complete_span';pt=bt;ph=b['bounded_prompt_sha256'];reason=['full_document_context_budget']
  else:mode='unsupported';pt=bt;ph=None;reason=['complete_span_'+b['bounded_status'],*b.get('bounded_reasons',[]),] if b['bounded_status']!='supported' else ['complete_span_context_budget']
  expected[str(cap)]={'mode':mode,'prompt_tokens':pt,'prompt_sha256':ph,'reasons':reason};counts[(cap,mode)]=counts.get((cap,mode),0)+1
 rows.append({'row_id':rid,'path':row['path'],'preedit_text':row['preedit_text'],'preedit_sha256':row['preedit_sha256'],'cursor':row['cursor'],'analyzer_helpers':row['analyzer_helpers'],'bounded_semantic_status':b['bounded_status'],'full_prompt_sha256':f['prompt_sha256'],'full_prompt_tokens':ft,'bounded_prompt_sha256':b['bounded_prompt_sha256'],'bounded_prompt_tokens':bt,'expected':expected})
out={'schema':'sepalith.run06.full_document_policy_fixtures.v1','target_or_gold_fields_present':False,'tokenizer_sha256':TSHA,'generation_reserve':1024,'denominator':49,'rows':rows,'counts':{str(cap):{m:counts.get((cap,m),0) for m in ('full_document','complete_span','unsupported')} for cap in (16384,32768)}};open(OUT/'policy-fixtures.json','w').write(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out['counts']))
