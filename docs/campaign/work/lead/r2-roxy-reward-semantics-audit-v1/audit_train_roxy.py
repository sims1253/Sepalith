#!/usr/bin/env python3
import collections,hashlib,json,os,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from roxy_objective_reward import extract_visible_signature,check_roxygen_candidate
ROWS=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl')
CONTEXT=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
rows={}
with ROWS.open() as f:
 for line in f:
  r=json.loads(line)
  if r.get('family')=='roxygen_drafting':rows[r['id']]=r
contexts={}; leakage=[]
with CONTEXT.open() as f:
 for line in f:
  c=json.loads(line)
  if c.get('row_id') in rows:
   contexts[c['row_id']]=c
   if c.get('context_has_target_or_reward_keys') is not False:leakage.append(c['row_id'])
assert len(rows)==len(contexts)==815 and not leakage
errors=collections.Counter();supported=[];unresolved=[];samples=[];generic_pass=0
for row_id in sorted(rows):
 r=rows[row_id]; c=contexts[row_id]['context']; suffix=c.get('suffix_lines')
 try:sig=extract_visible_signature(suffix)
 except ValueError as e:
  unresolved.append({'id':row_id,'reason':str(e)});continue
 check=check_roxygen_candidate(r['target_body_text'],sig)
 errors.update(error.split(':',1)[0] for error in check['errors'])
 generic = "#' Documentation for visible function"
 for parameter in sig.parameters:
  generic += f"\n#' @param {parameter} Value for {parameter}."
 if check_roxygen_candidate(generic, sig)['objective_pass']: generic_pass += 1
 item={'id':row_id,'package_id':r['package_id'],'function_name':sig.function_name,'parameters':list(sig.parameters),'target_sha256':hashlib.sha256(r['target_body_text'].encode()).hexdigest(),'gold_objective_pass':check['objective_pass'],'gold_errors':check['errors']}
 if check['objective_pass']:supported.append(item)
 if len(samples)<16:samples.append(item)
ids=[x['id'] for x in supported]
(HERE/'objective-supported-row-ids.json').write_text(json.dumps(ids,indent=2)+'\n')
result={'schema':'sepalith.rl11.roxy-reward-semantics-audit.v1','status':'cpu_train_only_audit_complete','inputs':{'rows':str(ROWS),'rows_sha256':sha(ROWS),'contexts':str(CONTEXT),'contexts_sha256':sha(CONTEXT)},'denominators':{'eligible15006_rows':15006,'roxygen_rows':len(rows),'contexts_joined':len(contexts),'context_leakage_flags':len(leakage),'visible_signature_resolved':len(rows)-len(unresolved),'visible_signature_unresolved':len(unresolved),'gold_objective_supported':len(supported),'gold_objective_not_supported':len(rows)-len(unresolved)-len(supported)},'gold_control_error_counts_by_category':dict(errors),'synthetic_generic_docs_objective_pass':generic_pass,'supported_ids_sha256':hashlib.sha256(canonical(ids)).hexdigest(),'unresolved':unresolved,'samples':samples,'interpretation':{'current_reward':'Exact target earns1.2. A protocol-valid, syntax-valid, nonexact roxygen edit earns0.0; semantic correctness beyond exact is not measured.','safe_finding':'Visible function signatures support objective invented/duplicate/missing parameter and roxygen-structure checks on only rows whose TRAIN gold passes the same checker.','positive_credit':'Not supported. Parameter coverage and parseable roxygen can be satisfied by generic or false descriptions and do not prove behavior, return value, side effects, or useful documentation.','proposed_use':'Keep exact reward unchanged. After root review, use the checker only for narrow negative diagnostics/penalties on the gold-controlled subset; passing alternatives remain reward0 and are reported separately.','future10017':'Not admitted here. Data owner context selection remains necessary; full-file anchor match alone is not semantic support.'},'scope':{'train_targets_used_as_gold_controls':True,'dev_derived_fixtures_or_labels_used':False,'generated_r_executed':False,'cuda':False,'notebook':False}}
(HERE/'train-roxy-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result['denominators'],sort_keys=True));print(json.dumps(dict(errors),sort_keys=True))
