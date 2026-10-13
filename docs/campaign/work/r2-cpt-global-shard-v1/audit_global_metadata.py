#!/usr/bin/env python3
"""Supplement the unchanged full source-byte validator with global-order exclusions."""
import collections,hashlib,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True;os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_global_shard as b
def main():
 out=b.HERE/'shard';m=json.loads((out/'manifest.json').read_text());v=json.loads((b.HERE/'source-reconstruction-validation.json').read_text());assert v['status']=='PASS' and v['manifest_sha256']==b.sha(out/'manifest.json')
 order=json.loads(b.ORDER.read_text())['packages'];parts=json.loads((b.BASE/'cpt-train-group-partition.json').read_text())['groups']
 broad=set();used=set();reserved=set();parents=set(json.loads((b.BASE/'known-nontrain-parent-hashes.json').read_text()))
 for r in map(json.loads,(b.BASE/'broader-shard-v1-2k/documents.jsonl').open()):broad.add(r['sha256']);used.add(r['group_id'])
 for r in map(json.loads,(b.BASE/'profile-shard-v1/documents.jsonl').open()):
  if r['cpt_partition']=='cpt_validation':reserved.add(r['sha256'])
 receipts=[json.loads(x) for x in (out/'package-receipts.jsonl').open()];assert [r['seeded_index'] for r in receipts]==list(range(m['last_seeded_index']+1))
 for r in receipts:assert (r['package'],r['group_id'])==(order[r['seeded_index']]['name'],order[r['seeded_index']]['group_id'])
 processed={r['group_id'] for r in receipts};seen=set();groups=collections.Counter();initials=collections.Counter();licenses=collections.Counter();documents=0;raw_bytes=0
 for r in map(json.loads,(out/'documents.jsonl').open()):
  assert r['group_id'] in processed and r['group_id'] not in used and parts[r['group_id']]=='cpt_train' and r['split']=='train_group'
  assert r['sha256']==r['document_id'] and r['sha256'] not in broad and r['sha256'] not in reserved and r['sha256'] not in seen
  assert not parents.intersection(r[k] for k in ['sha256','sha1','git_blob_sha1'])
  assert b.c.allowed_license(r['license']);seen.add(r['sha256']);groups[r['group_id']]+=r['source_code_tokens'];initials[r['package'][0].upper()]+=r['source_code_tokens'];licenses[r['license']]+=1;documents+=1;raw_bytes+=r['source_utf8_bytes']
 assert dict(groups)==m['group_code_token_counts'];assert max(groups.values(),default=0)<=m['group_code_token_cap']
 assert documents==m['counts']['cpt_train']['documents'] and raw_bytes==m['counts']['cpt_train']['raw_bytes']
 assert sum(groups.values())==m['counts']['cpt_train']['code_tokens'];assert sum(r['emitted_code_tokens'] for r in receipts)==sum(groups.values())
 result={'status':'PASS','documents':documents,'new_groups':len(groups),'code_tokens':sum(groups.values()),'existing_broad_group_overlap':0,'existing_broad_document_overlap':0,'reserved_validation_document_overlap':0,'reserved_validation_group_overlap':0,'known_nontrain_parent_hash_matches':0,'within_shard_exact_duplicates':0,'processed_seeded_package_prefix':len(receipts),'fully_processed_packages':sum(r['package_complete'] for r in receipts),'last_package_partial':bool(receipts and not receipts[-1]['package_complete']),'eligible_remaining_order_size':len(order),'group_fraction_of_remaining_order':len(groups)/len(order),'whole_corpus_coverage_claimed':False,'package_initials':sorted(initials),'code_tokens_by_package_initial':dict(initials),'maximum_group_code_tokens':max(groups.values(),default=0),'maximum_group_fraction':max(groups.values(),default=0)/max(1,sum(groups.values())),'source_reconstruction_validator':v,'manifest_sha256':b.sha(out/'manifest.json'),'training_admitted':False,'limits':['Exact-byte duplication and known non-TRAIN parent metadata only; no complete heldout near-duplicate inventory.','R syntax/execution not checked; this is raw source CPT.','Separate shard; later use requires root data-transition and budget admission.']}
 (b.HERE/'global-shard-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
