#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, json, os, subprocess
from pathlib import Path

PLAN = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
PACKET = PLAN/'docs/campaign/work/lead/r2-roxy-loop-scope-review-v1'
INPUT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Roxy-NSE-semantic-review-v1/r-input.json')
OUT = Path('/mnt/e/sepalith/campaign-20260915/data-work/Roxy-loop-scope-review-v1')
EXPECTED = 2273
POSITIVE = {'217b833f864559b916e600ba','0aaa68a5550a99eafa7ab6bd','1e6aba2099887b2541954c88'}
MIXED = {'1b17cd3e61434aa90c93a742': 'i'}
NEGATIVE = {'1f9a49ae2e8d3b01765e6588','1e27038e2dcc95d5b8e440b7'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def main():
 resume_raw = OUT.exists() and (OUT/'r-scope-output.jsonl').is_file() and (OUT/'r-stderr.log').is_file()
 if OUT.exists() and not resume_raw: raise FileExistsError('fresh output or complete raw replay required')
 OUT.mkdir(parents=True,exist_ok=True)
 data=json.loads(INPUT.read_text())
 rows=[r for g in data['source_groups'] for r in g['rows']]
 if len(rows)!=EXPECTED or len({r['row_id'] for r in rows})!=EXPECTED: raise ValueError('input parity')
 raw=OUT/'r-scope-output.jsonl'
 if not resume_raw:
  for group in data['source_groups']:
   if sha(Path(group['source_path']))!=group['source_sha256']: raise ValueError('source hash '+group['source_path'])
  env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
  result=subprocess.run(['Rscript','--vanilla',str(PACKET/'loop_scope_audit.R'),str(INPUT),str(raw)],text=True,capture_output=True,timeout=1200,env=env)
  (OUT/'r-stderr.log').write_text(result.stderr)
  if result.returncode: raise RuntimeError('R failed '+str(result.returncode))
 elif (OUT/'r-stderr.log').stat().st_size:
  raise ValueError('cannot resume raw replay with R stderr')
 audited=[json.loads(x) for x in raw.read_text().splitlines() if x]
 if len(audited)!=EXPECTED or {x['row_id'] for x in audited}!={x['row_id'] for x in rows}: raise ValueError('output parity')
 audited.sort(key=lambda x:x['row_id'])
 recovered=[x for x in audited if x['status']=='recoverable_lexically_bound_residuals']
 byid={x['row_id']:x for x in audited}
 if not POSITIVE <= {x['row_id'] for x in recovered}: raise ValueError('known loop/lexical controls missed')
 if any(byid[x]['status']=='recoverable_lexically_bound_residuals' for x in NEGATIVE): raise ValueError('known global recovered')
 for row_id,name in MIXED.items():
  detail=byid[row_id]['evidence']['residuals'][name]
  if not detail['all_occurrences_structurally_bound'] or byid[row_id]['status']=='recoverable_lexically_bound_residuals':
   raise ValueError('mixed bound/unbound control mishandled')
 with (OUT/'scope-ledger.jsonl').open('x') as f:
  for x in audited: f.write(canon(x)+'\n')
 for name,selected,status in [('recoverable-ids.json',recovered,'review_only_not_training_admission'),('retained-hold-ids.json',[x for x in audited if x not in recovered],'further_evidence_required')]:
  (OUT/name).write_text(json.dumps({'schema':'sepalith.dat10.roxy-loop-scope-ids.v1','status':status,'count':len(selected),'ids':[x['row_id'] for x in selected]},indent=2,sort_keys=True)+'\n')
 counts=collections.Counter(x['status'] for x in audited)
 classes=collections.Counter()
 types=collections.Counter()
 for x in audited:
  for d in (x.get('evidence') or {}).get('residuals',{}).values():
   classes.update(d.get('class_counts',{}))
   for o in d.get('occurrence_evidence',[]):
    if o.get('binding_type'): types[o['binding_type']]+=1
 summary={'schema':'sepalith.dat10.roxy-loop-scope-review.v1','status':'complete_review_only_root_admission_required','input':{'path':str(INPUT),'sha256':sha(INPUT),'rows':EXPECTED},'rows_audited':EXPECTED,'source_files_verified':len(data['source_groups']),'source_hashes_exact':True,'resumed_postprocessing_from_complete_raw_replay':resume_raw,'status_counts':dict(sorted(counts.items())),'recoverable_count':len(recovered),'retained_count':EXPECTED-len(recovered),'occurrence_class_counts':dict(sorted(classes.items())),'binding_type_counts':dict(sorted(types.items())),'known_positive_controls':sorted(POSITIVE),'known_mixed_bound_and_unbound_controls':MIXED,'known_true_global_controls':sorted(NEGATIVE),'policy':{'names_only_allowlist':False,'for_sequence_evaluated_before_binder':True,'for_body_and_nested_closures_inherit_binder':True,'function_body_inherits_formals':True,'assignment_or_post_loop_definite_binding_inferred':False,'codetools_global_crosscheck_required':True,'training_admission':False},'artifacts':{}}
 for p in [raw,OUT/'scope-ledger.jsonl',OUT/'recoverable-ids.json',OUT/'retained-hold-ids.json',OUT/'r-stderr.log']:
  summary['artifacts'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(canon(summary))
if __name__=='__main__': main()
