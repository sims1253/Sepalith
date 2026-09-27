#!/usr/bin/env python3
from __future__ import annotations
import collections,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-license-alias-recovery-v1')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 validator=load('alias_output_audit_validator',HERE/'source/campaign_cpt_data.py');aliases=load('alias_output_audit_aliases',HERE/'source/license_aliases.py')
 progress=json.loads((OUT/'progress.json').read_text());counts=collections.Counter();licenses=collections.Counter();docs=set();rowids=set();producer=collections.Counter();groups=[]
 for folder in sorted((OUT/'groups').iterdir()):
  if not folder.is_dir():continue
  receipt=json.loads((folder/'receipt.json').read_text());producer[receipt.get('materializer_sha256','initial_c9c3b0980679')]+=1
  for name,record in receipt['artifacts'].items():
   p=folder/name
   if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError(f'artifact mismatch: {p}')
  rows=[json.loads(x) for x in (folder/'cpt_train.jsonl').open()];checked=None
  if rows:
   checked=validator.validate_materialized_rows(rows,max_sequence_tokens=2048,require_complete_documents=True)
   if checked['payload_tokens']!=receipt['counts']['payload_tokens']:raise ValueError(f'validator count mismatch: {folder}')
  for row in rows:
   if row['row_id'] in rowids:raise ValueError('duplicate recovery row ID')
   rowids.add(row['row_id'])
  for row in map(json.loads,(folder/'documents.jsonl').open()):
   if row['document_id'] in docs:raise ValueError('duplicate recovery document')
   docs.add(row['document_id']);licenses[row['original_license']]+=1
   if aliases.normalize_recognized_alias(row['original_license'])!=row['canonical_license_for_existing_family']:raise ValueError('alias lineage mismatch')
   if not row['provisional_pending_terminal_main_dedup']:raise ValueError('document incorrectly marked final')
  counts.update(receipt['counts']);groups.append({'group':folder.name,'receipt_sha256':sha(folder/'receipt.json'),'counts':receipt['counts'],'validator_pass':checked is not None or receipt['counts'].get('rows',0)==0})
 if progress['recoverable_exclusion_records_at_frontier']!=counts['documents']+counts['dedup_excluded']+counts['repair']:raise ValueError('frontier accounting mismatch')
 if progress['counts']!=dict(counts):raise ValueError('progress aggregate mismatch')
 if progress['admitted'] or not progress['terminal_rescan_required']:raise ValueError('growing-main provisional gates differ')
 result={'schema':'sepalith.cpt.license-alias-recovery-audit.v1','status':'pass_provisional_pending_terminal_main_dedup','output':str(OUT),'progress_sha256':sha(OUT/'progress.json'),'captured_main_groups':progress['captured_main_groups'],'candidate_records':progress['recoverable_exclusion_records_at_frontier'],'counts':dict(counts),'original_license_documents':dict(sorted(licenses.items())),'unique_documents':len(docs),'unique_rows':len(rowids),'producer_group_counts':dict(producer),'existing_recovery_vs_new_main_conflicts':progress['existing_recovery_vs_new_main_conflicts'],'groups':groups,'admitted':False}
 (HERE/'output-audit-second-frontier.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='groups'},sort_keys=True))
if __name__=='__main__':main()
