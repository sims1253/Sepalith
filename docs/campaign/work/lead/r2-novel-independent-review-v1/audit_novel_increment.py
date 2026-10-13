#!/usr/bin/env python3
"""Independent metadata/TRAIN-payload audit of the frozen DAT10 v4 review packet."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, time
from pathlib import Path

V3=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/root-review-packet-v3-all-global-train.jsonl')
V4=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/expansion-increment-na-rm-v1/root-review-packet-v4-na-rm.jsonl')
GATE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/candidate-gate-v3/candidate-gate-ledger-v3.jsonl')
INC=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/expansion-increment-na-rm-v1/increment-ledger.jsonl')
GLOBAL=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
CPT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json')
ACCEPTED=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-token-rows.jsonl')
ACCEPTED_PROV=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-data-expansion-audit-v1/expanded-corrected-short-provenance.jsonl')
PINS={V3:'26e5e101a7995a32e3aec1e5d5cf545b1fb270c637d6ec49876c70bd43cf72fd',V4:'f4ee5e76a4c47f83c99137b64e4e2c95fd1e19a0c619d79cdc583f2918ba633a',GATE:'dd03d8cbb6499d540dca65720309b7ee32571bcc5cbcce90cbb645a545975574',INC:'4cd20b69bdd67cea3aec23a4ad25dc32856a1d387f1caef4be261580b0c0ae88',GLOBAL:'c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09',CPT:'6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06',ACCEPTED:'fa247ae7dbbf0b5a66538e8993d9fdd70624ce1c81ae54ae4d6f3d62b2368889',ACCEPTED_PROV:'317077b17afc38602309361396bb86d8897a6b823c5d7539f10b9733a0ad9a48'}
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
def u16(s):return len(s.encode('utf-16-le'))//2
def physical_old(x):
 sp=x['source_provenance']; c=x['context']
 v=sp.get('physical_region_old',sp.get('raw_region_old'))
 if isinstance(v,list):return v
 return c['region_old'] or ['']
def add(reasons,condition,name):
 if condition:reasons.append(name)

def main(out:Path):
 started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
 actual={str(p):sha_file(p) for p in PINS}
 bad={str(p):(PINS[p],actual[str(p)]) for p in PINS if PINS[p]!=actual[str(p)]}
 if bad:raise RuntimeError(f'input hash mismatch: {bad}')
 # Accepted pool hashes and identities; no DEV/final inputs.
 accepted_prompt=set();accepted_pair=set();accepted_target=set();accepted_ids=set()
 for r in rows(ACCEPTED):
  ph=sha_text(r['prompt_text']);th=sha_text(r['target_text'])
  accepted_prompt.add(ph);accepted_pair.add((ph,th));accepted_target.add(th);accepted_ids.add(r['id'])
 accepted_source=set()
 for p in rows(ACCEPTED_PROV):
  accepted_source.add((str(p.get('source') or ''),int(p.get('source_line') or 0),str(p.get('package_id') or '')))
 # Independent group split and CPT identity maps.
 gd=json.loads(GLOBAL.read_text());global_groups={str(g['group_id']):str(g['split']) for g in gd['groups']};del gd
 cd=json.loads(CPT.read_text());cpt_groups={str(k):str(v) for k,v in cd['groups'].items()};del cd
 gate={x['row_id']:x for x in rows(GATE) if x.get('cpt_partition')!='cpt_validation'}
 inc={x['row_id']:x for x in rows(INC)}
 # Prove v4 is byte-row prefix extension of frozen v3.
 prefix_rows=0
 with V3.open('rb') as a,V4.open('rb') as b:
  while True:
   la=a.readline()
   if not la:break
   if la!=b.readline():raise RuntimeError(f'v4 differs from v3 at row {prefix_rows+1}')
   prefix_rows+=1
 # Candidate prompt/target cardinalities used for cross-row conflict checks.
 prompt_targets=collections.defaultdict(set);pair_counts=collections.Counter();pair_first={};source_targets=collections.defaultdict(set)
 candidate=[]
 for i,x in enumerate(rows(V4)):
  r=x['row'];rid=r['id'];ph=x['prompt_sha256'];th=x['target_sha256'];sr=x['source_ref']
  prompt_targets[ph].add(th);pair_counts[(ph,th)]+=1;pair_first.setdefault((ph,th),i)
  source_targets[(str(sr.get('source') or ''),int(sr.get('line') or sr.get('source_line') or 0),str(sr.get('package_id') or sr.get('package') or ''))].add(th)
  candidate.append(x)
 decisions=collections.Counter();reasons_count=collections.Counter();families=collections.Counter();admit_family=collections.Counter();ledgers=[]
 for index,x in enumerate(candidate):
  r=x['row'];c=x['context'];sel=x['selection'];sp=x['source_provenance'];sr=x['source_ref'];rid=r['id'];family=r['family'];families[family]+=1
  ph=x['prompt_sha256'];th=x['target_sha256'];old=physical_old(x);reason=[]
  meta=gate.get(rid) if index<prefix_rows else inc.get(rid)
  add(reason,meta is None,'missing_partition_ledger')
  gid=str(sr.get('group_id') or '')
  add(reason,global_groups.get(gid)!='train_group','not_global_train')
  add(reason,cpt_groups.get(gid)=='cpt_validation','cpt_validation_reserved')
  add(reason,r.get('split')!='train' or sr.get('split')!='train_group','split_binding_mismatch')
  add(reason,not rid or rid!=sr.get('row_id') or rid!=sp.get('row_id',rid),'row_identity_mismatch')
  add(reason,rid in accepted_ids,'accepted_row_id_duplicate')
  # Exact token/text/hash closure.
  add(reason,sha_text(r['prompt_text'])!=ph,'prompt_hash_mismatch')
  add(reason,sha_text(r['target_text'])!=th,'target_hash_mismatch')
  add(reason,r['target_start']!=r['prompt_token_count']+1,'target_start_mismatch')
  add(reason,r['input_ids'][0]!=0 or r['input_ids'][-1]!=1,'bos_eos_mismatch')
  add(reason,r['input_ids'][r['target_start']:]!=r['target_body_tokens']+r['target_terminal_tokens']+[1],'target_token_tail_mismatch')
  add(reason,len(r['input_ids'])>4096 or len(r['input_ids'])-r['target_start']>1024,'length_contract_failure')
  # Prompt/target/source duplicate and contradiction checks.
  add(reason,ph in accepted_prompt,'accepted_prompt_duplicate')
  add(reason,(ph,th) in accepted_pair,'accepted_prompt_target_duplicate')
  # Retain the first exact pair and exclude only later copies. Conflicting
  # targets remain a repair for every member of the ambiguous prompt group.
  add(reason,pair_counts[(ph,th)]>1 and pair_first[(ph,th)]!=index,'within_candidate_prompt_target_duplicate')
  add(reason,len(prompt_targets[ph])>1,'within_candidate_prompt_conflicting_target')
  sk=(str(sr.get('source') or ''),int(sr.get('line') or sr.get('source_line') or 0),str(sr.get('package_id') or sr.get('package') or ''))
  add(reason,sk in accepted_source,'accepted_source_identity_duplicate')
  add(reason,len(source_targets[sk])>1,'source_identity_conflicting_target')
  # Exact selected-context and replacement geometry.
  add(reason,c.get('prefix')!=sel.get('prefix') or c.get('suffix_lines')!=sel.get('suffix'),'selection_context_mismatch')
  srgn=sel.get('region');crgn=c.get('region_old')
  add(reason,not (srgn==old and (crgn==old or (crgn==[] and old==['']))),'selected_region_mismatch')
  rr=c.get('replacement_range',{});start=rr.get('start',{});end=rr.get('end',{})
  add(reason,rr.get('content_sha256')!=sel.get('document_sha256'),'replacement_document_hash_mismatch')
  add(reason,start.get('line') is None or end.get('line') is None or end.get('line')<start.get('line'),'replacement_line_range_invalid')
  add(reason,start.get('character')!=0,'replacement_start_column_invalid')
  expected_end_line=start.get('line',0)+max(0,len(old)-1)
  expected_end_char=0 if old==[''] else u16(old[-1])
  add(reason,end.get('line')!=expected_end_line or end.get('character')!=expected_end_char,'replacement_region_extent_mismatch')
  if sp.get('target_start_line') is not None:
   add(reason,start.get('line')!=sp.get('target_start_line') or end.get('line')!=sp.get('target_end_line'),'provenance_range_mismatch')
  elif isinstance(sp.get('region_geometry'),dict):
   g=sp['region_geometry'];add(reason,start.get('line')!=g.get('start_line') or end.get('line')!=g.get('end_line') or not g.get('whole_region_verified'),'completion_range_mismatch')
  cur=c.get('cursor',{});cc=cur.get('code_point_column');uc=cur.get('utf16_column');ri=cur.get('region_line_index')
  if cc is None or uc is None:
   add(reason,not (cc is None and uc is None and ri==-1),'cursor_null_geometry_mismatch')
  else:
   add(reason,type(ri) is not int or not 0<=ri<len(old) or not 0<=cc<=len(old[ri]) or uc!=u16(old[ri][:cc]),'cursor_offset_mismatch')
  add(reason,sel.get('overflow') is not False or sel.get('required_overflow') is not False,'selection_overflow')
  # Target/evidence and leakage checks.
  op=r.get('target_operation');body=r.get('target_body_text','');oldtext='\n'.join(old)
  add(reason,(op=='no_op')!=(body=='[NO_EDIT]'),'operation_target_contradiction')
  add(reason,op=='replace' and body==oldtext,'unchanged_replacement')
  add(reason,op=='replace' and len(body)>=12 and body in r['prompt_text'],'target_text_in_prompt')
  if family=='finish_block':
   add(reason,sp.get('target_body_sha256')!=sha_text(body),'completion_target_evidence_mismatch')
   add(reason,sp.get('r_fragment',{}).get('passed') is not True or sp.get('finish_splice',{}).get('literal_source_splice_verified') is not True,'completion_structural_evidence_missing')
   add(reason,any(a.get('semantic_target_matches') is False for a in sp.get('generative_alternatives',[])),'completion_alternative_contradiction')
  elif family=='na_rm_propagation':
   add(reason,'na.rm = TRUE' not in body or 'na.rm = TRUE' in oldtext,'na_rm_transformation_contradiction')
  elif family=='pipe_rewrite': add(reason,'%>%' not in oldtext or '|>' not in body,'pipe_transformation_contradiction')
  elif family=='format_propagation': add(reason,''.join(oldtext.split())!=''.join(body.split()),'format_semantic_change')
  elif family=='roxygen_drafting': add(reason,old!=[''] or not body.startswith("#'"),'roxygen_geometry_or_target_contradiction')
  elif family=='no_op': add(reason,op!='no_op','noop_operation_contradiction')
  # Upstream gate reasons remain exclusions/repairs, not silently discarded.
  if index<prefix_rows:
   for z in (meta or {}).get('gate_reasons',[]): reason.append('upstream_'+z)
  # Direct, named license evidence is required for this independent admission subset.
  direct=(sr.get('license_evidence_present') is True and sr.get('license_status')=='direct_row_evidence' and bool(sr.get('license')))
  if not direct:reason.append('license_or_provenance_evidence_incomplete')
  if sel.get('support_revalidation_required') is True:reason.append('context_support_revalidation_required')
  # Classify permanent duplicate exclusions separately; all other problems are repair queues.
  reason=sorted(set(reason)); permanent=sorted(z for z in reason if 'duplicate' in z and 'conflicting' not in z)
  repair=sorted(z for z in reason if z not in permanent)
  decision='admissible' if not reason else ('exclude_duplicate' if permanent and not repair else 'repair_required')
  decisions[decision]+=1
  if decision=='admissible':admit_family[family]+=1
  for z in reason:reasons_count[z]+=1
  ledgers.append({'index':index,'row_id':rid,'group_id':gid,'family':family,'decision':decision,'reasons':reason,'permanent_exclusion_reasons':permanent,'repair_reasons':repair,'prompt_sha256':ph,'target_sha256':th})
 with (out/'audit-ledger.jsonl').open('w') as f:
  for x in ledgers:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 with (out/'admissible-ids.jsonl').open('w') as f:
  for x in ledgers:
   if x['decision']=='admissible':f.write(json.dumps({'row_id':x['row_id'],'index':x['index'],'family':x['family']},sort_keys=True)+'\n')
 queues=collections.defaultdict(list)
 for x in ledgers:
  for z in x['reasons']:queues[z].append(x['row_id'])
 qdoc={'schema':'sepalith.dat10.novel-independent-repair-queues.v1','denominator':len(ledgers),'queues':{k:{'rows':len(v),'row_ids':v} for k,v in sorted(queues.items())}}
 (out/'repair-queues.json').write_text(json.dumps(qdoc,indent=2)+'\n')
 duplicate_rows=sum(any('duplicate' in z and 'conflicting' not in z for z in x['reasons']) for x in ledgers)
 repair_rows=sum(bool(x['repair_reasons']) for x in ledgers)
 duplicate_and_repair=sum(bool(x['permanent_exclusion_reasons']) and bool(x['repair_reasons']) for x in ledgers)
 replay_path=out/'source-line-replay.json';replay=json.loads(replay_path.read_text()) if replay_path.exists() else None
 replay_pass=bool(replay and replay.get('status')=='pass' and replay.get('requested_rows')==len(ledgers) and replay.get('matched_rows')==len(ledgers))
 summary={'schema':'sepalith.dat10.novel-independent-review.v1','status':'independent_candidate_review_complete_no_producer_mutation','inputs':{str(p):{'sha256':actual[str(p)],'bytes':p.stat().st_size} for p in PINS},'denominators':{'v3_prefix_rows':prefix_rows,'na_rm_increment_rows':len(candidate)-prefix_rows,'review_rows':len(candidate),'accepted_comparison_rows':len(accepted_ids),'admissible_rows':decisions['admissible'],'permanent_duplicate_rows':duplicate_rows,'repair_queue_rows':repair_rows,'duplicate_and_repair_overlap':duplicate_and_repair},'family_rows':dict(families),'decision_rows':dict(decisions),'admissible_by_family':dict(admit_family),'reason_rows':dict(reasons_count),'checks':{'v4_exact_v3_prefix':prefix_rows==5109,'row_ids_unique':len({x['row_id'] for x in ledgers})==len(ledgers),'global_split_registry_checked':True,'cpt_validation_payload_opened':False,'token_hash_geometry_checked':True,'prompt_target_dedup_against_11505':True,'raw_source_line_replay':replay_pass},'elapsed_seconds':time.monotonic()-started,'artifacts':{}}
 for name in ['audit-ledger.jsonl','admissible-ids.jsonl','repair-queues.json']:
  p=out/name;summary['artifacts'][name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha_file(p)}
 if replay is not None:summary['artifacts']['source-line-replay.json']={'path':str(replay_path),'bytes':replay_path.stat().st_size,'sha256':sha_file(replay_path)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'decisions':dict(decisions),'admissible_by_family':dict(admit_family),'top_reasons':reasons_count.most_common(20),'elapsed':summary['elapsed_seconds']},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);main(a.parse_args().output)
