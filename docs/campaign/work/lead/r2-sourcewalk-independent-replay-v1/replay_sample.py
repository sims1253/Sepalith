#!/usr/bin/env python3
"""Independent bounded source/provenance replay for frozen source-walk rows.

Reads exact raw scenario lines, normalized source bytes, global/protected split
registries, and DESCRIPTION license bytes. Outputs hashes and statuses only.
"""
from __future__ import annotations

import argparse, collections, hashlib, importlib.util, json, os, time
from pathlib import Path
from typing import Any

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/source-walk-shards-v1')
GLOBAL=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
CPT=PLAN/'docs/campaign/work/r2-corpus-preparation-v1/cpt-train-group-partition.json'
HOLDS=Path('/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-SFT-admission-review-v1/mechanical-hold-ledger.jsonl')
STRICT=PLAN/'docs/campaign/work/lead/r2-roxy8597-root-fixes-v1/audit_authoritative_stream.py'
STRICT_SHA='7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37'
GLOBAL_SHA='c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
CPT_SHA='6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06'
DEFAULT_SHARDS=(5,10,15,20,25,30,35,40)

def sha_bytes(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb',buffering=4<<20) as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def atomic_json(p:Path,o:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 q.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
 with q.open('rb') as f:os.fsync(f.fileno())
 q.replace(p)
 d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)

def load_strict():
 if sha_file(STRICT)!=STRICT_SHA:raise RuntimeError('strict_validator_hash_mismatch')
 spec=importlib.util.spec_from_file_location('pinned_strict_protocol',STRICT)
 if spec is None or spec.loader is None:raise RuntimeError('strict_validator_import_failed')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def choose(shards:tuple[int,...],per_family:int,max_packages:int):
 holds={json.loads(x)['row_id'] for x in HOLDS.open()}
 selected=[]; packages=set(); shard_pins=[]
 for shard in shards:
  d=BASE/f'shard-{shard:04d}'/'structured-materialization-v1'
  tm=json.loads((d/'token-audit-manifest.json').read_text()); p=Path(tm['token_rows']['path'])
  digest=hashlib.sha256(); rows=[]
  with p.open('rb') as f:
   for line_no,raw in enumerate(f,1):
    digest.update(raw); o=json.loads(raw); r=o['row']
    if r['id'] not in holds: rows.append((line_no,o))
  if digest.hexdigest()!=tm['token_rows']['sha256'] or len(rows)+sum(1 for _ in ()) > tm['token_rows']['rows']:
   # Row count is checked separately because held rows are deliberately skipped.
   if digest.hexdigest()!=tm['token_rows']['sha256']:raise RuntimeError(f'token_hash_mismatch:{shard}')
  original_rows=sum(1 for _ in p.open('rb'))
  if original_rows!=tm['token_rows']['rows']:raise RuntimeError(f'token_count_mismatch:{shard}')
  for fam in ('roxygen_drafting','no_op'):
   candidates=[x for x in rows if x[1]['row']['family']==fam]
   if not candidates:continue
   package=next((x[1]['row']['package_id'] for x in candidates if x[1]['row']['package_id'] in packages),None)
   if package is None:
    package=candidates[0][1]['row']['package_id']
    if len(packages)>=max_packages:continue
    packages.add(package)
   selected.extend([(shard,*x) for x in candidates if x[1]['row']['package_id']==package][:per_family])
  shard_pins.append({'shard':shard,'token_rows':{'path':str(p),'rows':original_rows,'sha256':digest.hexdigest()},'token_manifest_sha256':sha_file(d/'token-audit-manifest.json')})
 return selected,packages,shard_pins

def reconstructed_source(raw:dict[str,Any],family:str)->tuple[bytes,bytes]:
 before=list(raw.get('prefix',[]))+list(raw.get('region_old',[]))+list(raw.get('suffix',[]))
 after=before if family=='no_op' else list(raw.get('prefix',[]))+list(raw.get('region_new',[]))+list(raw.get('suffix',[]))
 return ('\n'.join(before)+'\n').encode(), ('\n'.join(after)+'\n').encode()

def decide(e:dict[str,Any])->str:
 common=('raw_line_hash_match','raw_source_file_hash_match','raw_metadata_match','global_train','protected_disjoint','normalized_source_hash_match','normalized_source_reconstruction_match','source_parse_ok','license_hash_match','license_fields_match','strict_protocol_ok')
 if not all(e.get(x) is True for x in common):return 'hold_independent_provenance_failure'
 if e['family']=='no_op':
  if e.get('noop_family_predicate')!='exact_unchanged_source_supported_kind':return 'hold_noop_family_predicate'
  if e.get('reference_analyzer')!='complete_not_applicable_noop':return 'hold_analyzer_incomplete'
  return 'provenance_supported_candidate_root_review_required'
 if e.get('reference_analyzer')!='complete':return 'provenance_pass_semantic_analyzer_queued'
 if e.get('unsupported_claim_analyzer')!='complete':return 'provenance_pass_unsupported_claim_analysis_queued'
 return 'provenance_and_analyzers_supported_candidate_root_review_required'

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--shards',default=','.join(map(str,DEFAULT_SHARDS)));ap.add_argument('--per-family',type=int,default=3);ap.add_argument('--max-packages',type=int,default=20);a=ap.parse_args()
 if a.out.exists() and any(a.out.iterdir()):raise ValueError('output_must_be_fresh')
 a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic(); strict=load_strict()
 if sha_file(GLOBAL)!=GLOBAL_SHA or sha_file(CPT)!=CPT_SHA:raise RuntimeError('registry_hash_mismatch')
 groups={x['group_id']:x for x in json.loads(GLOBAL.read_text())['groups']};cpt=json.loads(CPT.read_text())['groups']
 selected,packages,shard_pins=choose(tuple(map(int,a.shards.split(','))),a.per_family,a.max_packages)
 ids={x[2]['row']['id'] for x in selected}; packets={}; packet_pins=[]
 for shard in sorted({x[0] for x in selected}):
  d=BASE/f'shard-{shard:04d}'/'structured-materialization-v1'; manifest=json.loads((d/'manifest.json').read_text()); p=Path(manifest['outputs']['candidate_packets']['path']);h=hashlib.sha256();n=0
  with p.open('rb') as f:
   for raw in f:
    h.update(raw);n+=1;o=json.loads(raw);rid=o.get('row_ref',{}).get('row_id')
    if rid in ids:packets[rid]=o
  if h.hexdigest()!=manifest['outputs']['candidate_packets']['sha256'] or n!=manifest['outputs']['candidate_packets']['rows']:raise RuntimeError(f'packet_pin_mismatch:{shard}')
  packet_pins.append({'shard':shard,'path':str(p),'rows':n,'sha256':h.hexdigest()})
 if set(packets)!=ids:raise RuntimeError('packet_join_incomplete')

 # One full pass per raw scenario file provides both exact source-file and line hashes.
 wanted=collections.defaultdict(lambda:collections.defaultdict(list))
 for shard,line_no,item in selected:
  ref=item['source_ref'];wanted[Path(ref['file'])][int(ref['line'])].append((item['row']['id'],ref))
 raw_by_id={};raw_file_meta=[]
 for path,line_map in wanted.items():
  before=path.stat();h=hashlib.sha256();found=set()
  with path.open('rb',buffering=4<<20) as f:
   for n,raw_line in enumerate(f,1):
    h.update(raw_line)
    if n in line_map:
     found.add(n)
     for rid,ref in line_map[n]:raw_by_id[rid]=(raw_line,json.loads(raw_line))
  after=path.stat(); expected={ref['source_sha256'] for vals in line_map.values() for _,ref in vals}
  raw_file_meta.append({'path':str(path),'bytes':after.st_size,'sha256':h.hexdigest(),'expected_sha256':sorted(expected),'stat_stable':(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'requested_lines':len(line_map),'found_lines':len(found)})

 # Parse source only; never execute R.
 from tree_sitter import Language,Parser
 import tree_sitter_r
 parser=Parser(Language(tree_sitter_r.language()))
 source_cache={};license_cache={};results=[]
 for shard,line_no,item in selected:
  r=item['row'];rid=r['id'];packet=packets[rid];ref=item['source_ref'];raw_bytes,raw=raw_by_id[rid];v=packet['validation'];source_path=Path(v['source_path']);license_ev=v['license_evidence'];license_path=Path(license_ev['path'])
  if source_path not in source_cache:
   b=source_path.read_bytes();source_cache[source_path]=(b,sha_bytes(b),not parser.parse(b).root_node.has_error)
  source_bytes,source_sha,parse_ok=source_cache[source_path]
  if license_path not in license_cache:
   b=license_path.read_bytes();text=b.decode('utf-8','replace').splitlines();license_cache[license_path]=(sha_bytes(b),text)
  license_sha,license_lines=license_cache[license_path];before_src,after_src=reconstructed_source(raw,r['family'])
  raw_match=sha_bytes(raw_bytes)==ref['raw_line_sha256'] or sha_bytes(raw_bytes.rstrip(b'\r\n'))==ref['raw_line_sha256']
  token_errors=strict.validate_token_row(r,full_text=True)
  group=groups.get(ref['group_id']);partition=cpt.get(ref['group_id'])
  raw_meta=raw.get('family')==r['family'] and raw.get('package')==r['package_id'] and raw.get('path')==packet['result']['context']['path']
  # Scenario rows may contain a source window rather than the complete file.
  # Require the reconstructed after-window to occur in the independently
  # reopened normalized source; equality is the whole-file special case.
  candidate_window=after_src.rstrip(b'\n')
  window_occurrences=source_bytes.count(candidate_window) if candidate_window else 0
  recon_match=window_occurrences==1
  e={'row_id':rid,'family':r['family'],'shard':shard,'package_id':r['package_id'],'group_id':ref['group_id'],'raw_source_path':ref['file'],'raw_source_line':ref['line'],'raw_source_sha256':ref['source_sha256'],'raw_line_sha256':ref['raw_line_sha256'],'raw_line_hash_match':raw_match,'raw_source_file_hash_match':next(x for x in raw_file_meta if x['path']==str(Path(ref['file'])))['sha256']==ref['source_sha256'],'raw_metadata_match':raw_meta,'global_split':group.get('split') if group else None,'global_train':bool(group and group.get('split')=='train_group'),'protected_disjoint':bool(group and group.get('split')=='train_group' and partition!='cpt_validation'),'cpt_partition':partition or 'absent_not_disqualifying','normalized_source_path':str(source_path),'normalized_source_sha256':source_sha,'normalized_source_hash_match':source_sha==v['source_sha256'],'normalized_source_reconstruction_match':recon_match,'normalized_window_occurrences':window_occurrences,'before_snapshot_sha256':sha_bytes(before_src),'after_snapshot_sha256':sha_bytes(after_src),'source_parse_ok':parse_ok,'license_path':str(license_path),'license_sha256':license_sha,'license_hash_match':license_sha==license_ev['sha256'],'license_field_sha256':[sha_bytes(x.encode()) for x in license_ev['fields']],'license_fields_match':all(x in license_lines for x in license_ev['fields']) and not any(x in ('License_restricts_use: yes','License_is_FOSS: no') for x in license_lines),'strict_protocol_ok':not token_errors,'strict_protocol_errors':token_errors,'reference_analyzer':'complete_not_applicable_noop' if r['family']=='no_op' else 'pending_full_source_semantic_follow_on','unsupported_claim_analyzer':'complete_not_applicable_noop' if r['family']=='no_op' else 'pending_full_source_semantic_follow_on','source_text_written':False,'target_text_written':False}
  if r['family']=='no_op':
   kind=raw.get('kind');exact_unchanged=(before_src==after_src and raw.get('region_new')==[] and r['target_operation']=='no_op' and r['target_body_text']=='[NO_EDIT]')
   e['noop_source_kind']=kind;e['noop_family_predicate']='exact_unchanged_source_supported_kind' if exact_unchanged and kind in ('after_close_brace','blank_between') else 'failed'
  e['status']=decide(e);results.append(e)

 # Source-backed negative controls mutate evidence from an actually replayed row.
 base=dict(next(r for r in results if r['status']!='hold_independent_provenance_failure'));controls={}
 for name,field in [('raw_hash','raw_line_hash_match'),('global_split','global_train'),('license_hash','license_hash_match'),('source_hash','normalized_source_hash_match'),('protocol','strict_protocol_ok')]:
  x=dict(base);x[field]=False;controls[name]={'status':decide(x),'passed':decide(x)=='hold_independent_provenance_failure'}
 x=dict(next(r for r in results if r['family']=='roxygen_drafting' and r['status']=='provenance_pass_semantic_analyzer_queued'));x['reference_analyzer']='';controls['missing_roxy_analyzer']={'status':decide(x),'passed':decide(x)=='provenance_pass_semantic_analyzer_queued'}
 if not all(x['passed'] for x in controls.values()):
  atomic_json(a.out/'failed-controls.json',{'controls':controls,'base_statuses':collections.Counter(x['status'] for x in results),'first_rows':results[:2]})
  raise AssertionError('negative_control_failed')
 counts=collections.Counter(x['status'] for x in results);families=collections.Counter(x['family'] for x in results);shards=collections.Counter(x['shard'] for x in results)
 ledger=a.out/'sample-ledger.jsonl';
 with ledger.open('w') as f:
  for x in sorted(results,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 manifest={'schema':'sepalith.dat10.sourcewalk_independent_replay_sample.v1','status':'complete_sample_review_only','rows':len(results),'families':dict(families),'packages':len(packages),'shards':dict(shards),'status_counts':dict(counts),'source_files':raw_file_meta,'normalized_source_files':len(source_cache),'license_files':len(license_cache),'strict_validator':{'path':str(STRICT),'sha256':STRICT_SHA},'registry_pins':{'global':{'path':str(GLOBAL),'sha256':GLOBAL_SHA},'cpt':{'path':str(CPT),'sha256':CPT_SHA}},'shard_token_pins':shard_pins,'packet_pins':packet_pins,'negative_controls':controls,'outputs':{'ledger':{'path':str(ledger),'rows':len(results),'bytes':ledger.stat().st_size,'sha256':sha_file(ledger)}},'elapsed_seconds':time.monotonic()-started,'cpu_threads_max':2,'cuda':False,'source_or_target_text_written':False,'training_admission':False,'full_stream_launched':False}
 atomic_json(a.out/'manifest.json',manifest);print(json.dumps({'rows':len(results),'families':dict(families),'packages':len(packages),'status_counts':dict(counts),'elapsed_seconds':manifest['elapsed_seconds']},sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
