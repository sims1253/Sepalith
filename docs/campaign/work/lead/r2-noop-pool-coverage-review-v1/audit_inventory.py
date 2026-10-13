import json, hashlib, collections, math, os, time
BASE='/mnt/e/sepalith/campaign-20260915/data-work'
cur=f'{BASE}/DAT10-expanded-15006-v1/combined-token-rows.jsonl'
ctx=f'{BASE}/RL11-15006-context-v1/context-sidecar.jsonl'
inv=f'{BASE}/DAT10-novel-v1/all-data-inventory-v1/all-metadata-inventory-v1.jsonl'
gate=f'{BASE}/DAT10-novel-v1/candidate-gate-v3/candidate-gate-ledger-v3.jsonl'
sup=f'{BASE}/DAT10-novel-v1/all-data-inventory-v1/support-review-noop-450-or-more.jsonl'
out=f'{BASE}/Noop-pool-coverage-review-v1/audit-census.json'

def hfile(p):
 h=hashlib.sha256(); n=0
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b); n+=len(b)
 return {'path':p,'bytes':n,'sha256':h.hexdigest()}

def bucket(n):
 return '<=512' if n<=512 else '<=1024' if n<=1024 else '<=2048' if n<=2048 else '<=4096' if n<=4096 else '>4096'
start=time.time(); current_ids=set(); current_prompts=set(); current_sources=set(); packages=set(); lenb=collections.Counter(); targets=collections.Counter(); current_rows=0; allrows=0
with open(cur) as f:
 for line in f:
  x=json.loads(line); allrows+=1
  if x.get('family')!='no_op': continue
  current_rows+=1; rid=x['id']; current_ids.add(rid); packages.add(x.get('package_id')); current_prompts.add(hashlib.sha256(x['prompt_text'].encode()).hexdigest()); lenb[bucket(len(x['input_ids']))]+=1
  targets[(x.get('target_operation'),x.get('target_body_text'),x.get('split'))]+=1
ctxstat=collections.Counter(); history=collections.Counter(); oldlines=collections.Counter(); prefixb=collections.Counter(); suffixb=collections.Counter(); cursor=collections.Counter(); ctxrows=0
with open(ctx) as f:
 for line in f:
  x=json.loads(line)
  if x['row_id'] not in current_ids: continue
  ctxrows+=1; c=x['context']; si=x.get('source_identity')
  current_sources.add(json.dumps(si,sort_keys=True,separators=(',',':')))
  history[tuple(e.get('kind') for e in c.get('history',[]))]+=1
  oldlines[len(c.get('region_old',[]))]+=1
  prefixb[bucket(len(c.get('prefix',[])))]+=1; suffixb[bucket(len(c.get('suffix_lines',[])))]+=1
  curp=c.get('cursor') or {}; cursor[(curp.get('region_line_index'),curp.get('code_point_column'))]+=1
  sg=x.get('selection_geometry') or {}; ctxstat[(sg.get('mode'),sg.get('complete_buffer_available'))]+=1
# Candidate gate noops
candidate={}; gate_reasons=collections.Counter(); candidate_part=collections.Counter(); candidate_packages=set(); candidate_groups=set(); candidate_len=collections.Counter()
with open(gate) as f:
 for line in f:
  x=json.loads(line)
  if x.get('family')!='no_op': continue
  candidate[x['row_id']]=x; candidate_part[x.get('cpt_partition')]+=1; candidate_packages.add(x.get('package_id')); candidate_groups.add(x.get('group_id'))
  gate_reasons.update(x.get('gate_reasons',[])); candidate_len[bucket(x.get('sequence_tokens',0))]+=1
# support noops
support={}; support_status=collections.Counter(); support_reasons=collections.Counter(); support_groups=set(); support_len=collections.Counter()
with open(sup) as f:
 for line in f:
  x=json.loads(line); support[x['row_id']]=x; support_groups.add(x.get('group_id')); support_status[(x.get('cpt_partition'),x.get('final_status'))]+=1; support_reasons.update(x.get('policy_reasons',[])); support_len[bucket(x.get('prompt_chars',0))]+=1
# Full inventory no-op metadata
invstat=collections.Counter(); invroute=collections.Counter(); invreasons=collections.Counter(); invgroups=set(); invids=set(); inv_train_cpt=set(); inv_prompt=collections.Counter(); invrows=0
with open(inv) as f:
 for line in f:
  x=json.loads(line)
  if x.get('family')!='no_op': continue
  invrows+=1; rid=x['row_id']; invids.add(rid); invgroups.add(x.get('group_id')); invstat[(x.get('global_registry_split'),x.get('cpt_partition'),x.get('final_status'),x.get('source_attempt_status'))]+=1; invroute[x.get('materialization_route')]+=1; invreasons.update(x.get('policy_reasons',[])); inv_prompt[bucket(x.get('prompt_chars',0))]+=1
  if x.get('global_registry_split')=='train_group' and x.get('cpt_partition')=='cpt_train': inv_train_cpt.add(rid)
eligible_not_current=inv_train_cpt-current_ids
res={
 'schema':'sepalith.dat10.noop-pool-coverage-census.v1','status':'audit_complete_no_admission','elapsed_seconds':round(time.time()-start,3),
 'inputs':{k:hfile(p) for k,p in [('current15006',cur),('context15006',ctx),('inventory',inv),('candidate_gate',gate),('support_queue',sup)]},
 'current15006':{'all_rows':allrows,'noop_rows':current_rows,'share':current_rows/allrows,'unique_ids':len(current_ids),'unique_prompt_text_sha256':len(current_prompts),'unique_packages':len(packages),'unique_source_identity_records':len(current_sources),'context_rows_joined':ctxrows,'sequence_length_buckets':dict(lenb),'target_contracts':{str(k):v for k,v in targets.items()},'history_kind_tuples':{str(k):v for k,v in history.items()},'region_old_line_counts':{str(k):v for k,v in oldlines.items()},'prefix_line_buckets':dict(prefixb),'suffix_line_buckets':dict(suffixb),'selection_geometry':{str(k):v for k,v in ctxstat.items()},'unique_cursor_positions':len(cursor)},
 'candidate_gate_noops':{'rows':len(candidate),'cpt_partition':dict(candidate_part),'unique_packages':len(candidate_packages),'unique_groups':len(candidate_groups),'gate_reasons':dict(gate_reasons),'sequence_length_buckets':dict(candidate_len),'overlap_current_ids':len(set(candidate)&current_ids)},
 'support_queue_noops':{'rows':len(support),'status':{str(k):v for k,v in support_status.items()},'unique_groups':len(support_groups),'policy_reasons':dict(support_reasons),'prompt_char_buckets':dict(support_len),'overlap_current_ids':len(set(support)&current_ids),'overlap_candidate_ids':len(set(support)&set(candidate))},
 'full_inventory_noops':{'rows':invrows,'unique_ids':len(invids),'unique_groups':len(invgroups),'statuses':{str(k):v for k,v in invstat.items()},'routes':dict(invroute),'policy_reasons':dict(invreasons),'prompt_char_buckets':dict(inv_prompt),'train_cpt_rows':len(inv_train_cpt),'train_cpt_not_current':len(eligible_not_current),'candidate_gate_overlap':len(set(candidate)&invids),'support_overlap':len(set(support)&invids),'current_overlap':len(current_ids&invids)},
 'sets':{'current_not_in_inventory_count':len(current_ids-invids),'candidate_cpt_train_not_current':len({i for i,x in candidate.items() if x.get('cpt_partition')=='cpt_train'}-current_ids),'support_cpt_train_not_current':len({i for i,x in support.items() if x.get('cpt_partition')=='cpt_train'}-current_ids)},
}
os.makedirs(os.path.dirname(out),exist_ok=True)
tmp=out+'.tmp'; open(tmp,'w').write(json.dumps(res,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out)
print(json.dumps(res,indent=2,sort_keys=True))
