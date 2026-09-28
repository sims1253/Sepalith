#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAPS=(192,384,768)
def sha_bytes(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arm(cap):
 p=ROOT/'evidence'/f'cap-{cap}'/'results.json'; raw=p.read_bytes();x=json.loads(raw)
 assert x['cap']==cap and x['status']=='complete'
 assert x['expected_ids']==x['completed_ids'] and len(x['expected_ids'])==75
 assert [r['id'] for r in x['records']]==x['expected_ids']
 counts={'protocol_valid':0,'edit_exact':0,'strict_noop_correct':0,'false_suggestions':0,'cap_hit':0,'canonical_eos':0,'noncanonical_eog':0,'transport_failed':0,'mechanical_failed':0}
 for r in x['records']:
  assert hashlib.sha256(r['raw_text'].encode()).hexdigest()==r['raw_text_sha256']
  q,e=r['quality'],r['eos']
  counts['protocol_valid']+=bool(r['protocol']['valid']);counts['edit_exact']+=bool(q['edit_exact']);counts['strict_noop_correct']+=bool(q.get('strict_noop_correct'));counts['false_suggestions']+=bool(q.get('noop_false_positive'));counts['canonical_eos']+=bool(e['canonical']);counts['noncanonical_eog']+=e['status']=='noncanonical_eog';counts['transport_failed']+=r['failure_class']=='transport';counts['mechanical_failed']+=r['failure_class']=='mechanical';counts['cap_hit']+=(not e['canonical'] and len(r['returned_token_ids'])==cap)
 assert counts==x['counts'],(cap,counts,x['counts'])
 assert sum(r['expected_noop'] for r in x['records'])==32
 assert sum(not r['expected_noop'] for r in x['records'])==43
 return p,x,counts,hashlib.sha256(raw).hexdigest()
arms={};xs={}
for c in CAPS:
 p,x,counts,h=arm(c);xs[c]=x;arms[str(c)]={'status':'complete','results':{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h},'denominators':x['denominators'],'counts':counts,'rates':{'edit_exact':counts['edit_exact']/43,'strict_noop':counts['strict_noop_correct']/32,'combined_strict_correct':(counts['edit_exact']+counts['strict_noop_correct'])/75,'protocol_valid':counts['protocol_valid']/75,'false_suggestion_per_noop':counts['false_suggestions']/32,'cap_hit':counts['cap_hit']/75}}
base=xs[192]['expected_ids']
for c in (384,768):
 assert xs[c]['expected_ids']==base
 for a,b in zip(xs[192]['records'],xs[c]['records']):
  assert a['id']==b['id'];assert a['prompt_sha256']==b['prompt_sha256'];assert a['hf_prompt_ids_sha256']==b['hf_prompt_ids_sha256'];assert a['native_prompt_ids_sha256']==b['native_prompt_ids_sha256'];assert a['target_sha256']==b['target_sha256'];assert a['expected_noop']==b['expected_noop']
paired=[]
for i,rid in enumerate(base):
 rs={c:xs[c]['records'][i] for c in CAPS}
 row={'id':rid,'family':rs[192]['family'],'expected_noop':rs[192]['expected_noop'],'prompt_sha256':rs[192]['prompt_sha256'],'target_sha256':rs[192]['target_sha256'],'caps':{}}
 for c,r in rs.items():
  row['caps'][str(c)]={'status':r['status'],'returned_tokens':len(r['returned_token_ids']),'raw_text_sha256':r['raw_text_sha256'],'returned_token_ids_sha256':r['returned_token_ids_sha256'],'canonical_eos':r['eos']['canonical'],'protocol_valid':r['protocol']['valid'],'edit_exact':r['quality']['edit_exact'],'strict_noop_correct':r['quality'].get('strict_noop_correct',False),'false_suggestion':r['quality'].get('noop_false_positive',False),'cap_hit':not r['eos']['canonical'] and len(r['returned_token_ids'])==c}
 row['changes']={'protocol_192_to_384':row['caps']['192']['protocol_valid']!=row['caps']['384']['protocol_valid'],'protocol_384_to_768':row['caps']['384']['protocol_valid']!=row['caps']['768']['protocol_valid'],'correctness_any_change':len({(row['caps'][str(c)]['edit_exact'],row['caps'][str(c)]['strict_noop_correct']) for c in CAPS})>1,'raw_384_equals_768':row['caps']['384']['raw_text_sha256']==row['caps']['768']['raw_text_sha256']}
 paired.append(row)
original=['dat07-existing-e777f718a62393c54cb5cdbe','dat07-existing-acac075364038a6979bb0631','e623a61b5a4c066358a477f2','04834fef4fe59742f13677a9']
by={r['id']:r for r in paired}
assert set(original)<=set(by)
summary={'schema':'sepalith.run06.e750-quality6-terminal-comparison.v1','status':'complete_root_review_candidate','run_id':'root-20260915T0446','expected_ids_sha256':hashlib.sha256(('\n'.join(base)+'\n').encode()).hexdigest(),'order_and_prompt_identity_exact_all_caps':True,'arms':arms,'paired':{'rows':75,'comparison_path':'paired-comparison.json','correctness_changed_cases':sum(r['changes']['correctness_any_change'] for r in paired),'protocol_recoveries_192_to_384':sum((not r['caps']['192']['protocol_valid']) and r['caps']['384']['protocol_valid'] for r in paired),'protocol_recoveries_384_to_768':sum((not r['caps']['384']['protocol_valid']) and r['caps']['768']['protocol_valid'] for r in paired),'original_cap192_hits':[by[x] for x in original]},'conclusion':['Edit exact remains29/43 and strict no-op remains26/32 at all three caps; no case gains exact correctness.','Cap384 recovers protocol termination for acac and048; cap768 additionally recovers e777. acac/e777 become protocol-valid false edits, while048 remains an inexact edit.','e623 remains capped and protocol-invalid at768.','The number of protocol-valid false suggestions rises from4 to5 to6 as two invalid no-op generations acquire terminal EOS.','Quality results are offline DEV evidence and are not production latency evidence or training data.']}
(ROOT/'paired-comparison.json').write_text(json.dumps({'schema':'sepalith.run06.e750-quality6.paired75.v1','rows':paired},indent=2,sort_keys=True)+'\n')
(ROOT/'terminal-comparison.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','arms':{k:v['counts'] for k,v in arms.items()},'correctness_changed_cases':summary['paired']['correctness_changed_cases'],'protocol_recoveries':[summary['paired']['protocol_recoveries_192_to_384'],summary['paired']['protocol_recoveries_384_to_768']]},sort_keys=True))
