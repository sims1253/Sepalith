#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path

HOLD_SHA='82424614fc26b1bcc5fdc003cb187c20ec999354960fc8fbb02741d20a4bfb4d'
SEMANTIC_SHA='1bc01b04a64ca800ffd6be3b74e23860e93dbcfb3f9a358702338ffd9980282a'
PREDICTION_SHA='233c39c62517b6468c3335bdc0bc2e6352aed8c0359da4f9f585874ebecf1cb1'
EXPECTED=140
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(path,key='row_id'):
 out={}
 with Path(path).open() as f:
  for line in f:
   x=json.loads(line);rid=x[key];require(rid not in out,'duplicate:'+rid);out[rid]=x
 return out
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def main():
 p=argparse.ArgumentParser();p.add_argument('--holds',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--helper',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
 require(sha(a.holds)==HOLD_SHA and sha(a.semantic)==SEMANTIC_SHA and sha(a.prediction)==PREDICTION_SHA,'input pin differs')
 holds={rid:x for rid,x in read(a.holds).items() if x['reason']=='external_namespace_import_provider_unavailable'};semantic=read(a.semantic);prediction=read(a.prediction);require(len(holds)==EXPECTED and set(holds)<=set(semantic)&set(prediction),'external hold closure differs')
 namespaces={}
 for rid in sorted(holds):
  row=semantic[rid];ns=row['namespace'];require(ns['status']=='namespace_inventory_complete' and ns['present'] is True and ns['stat_stable'] is True and ns['sha256']==ns['parsed_sha256'],'upstream namespace gate differs:'+rid)
  path=ns['path'];record=namespaces.setdefault(path,{'id':hashlib.sha256(path.encode()).hexdigest()[:24],'path':path,'expected_sha256':ns['sha256']});require(record['expected_sha256']==ns['sha256'],'namespace path has multiple hashes')
 a.output.mkdir(parents=True,exist_ok=False);request=a.output/'namespace-request.json';raw=a.output/'namespace-evidence.jsonl';request.write_text(json.dumps({'namespaces':sorted(namespaces.values(),key=lambda x:x['id'])},indent=2,sort_keys=True)+'\n')
 result=subprocess.run(['Rscript',str(a.helper),str(request),str(raw)],capture_output=True,text=True,timeout=120);require(result.returncode==0,'namespace parse infrastructure failure')
 evidence=read(raw,key='id');by_path={x['path']:evidence[x['id']] for x in namespaces.values()};decisions=[];counts=collections.Counter();dep_counts=collections.Counter();origins=collections.Counter()
 for rid in sorted(holds):
  hold=holds[rid];sem=semantic[rid];pred=prediction[rid];ev=by_path[sem['namespace']['path']]
  require(ev['parsed_sha256']==sem['namespace']['sha256'] and ev['stat_stable'] is True and ev['status']=='namespace_evidence_complete','new namespace evidence differs:'+rid)
  explicit=collections.defaultdict(set);wildcards=set()
  for directive in ev['directives']:
   if directive['kind']=='importFrom':
    for symbol in directive['symbols']:explicit[symbol].add(directive['package'])
   else:wildcards.add(directive['package'])
  dependencies=hold['external_import_dependencies'];mapping={};reasons=[]
  direct=set(sem['reference_resolution']['direct_references']);visited=set(sem['reference_resolution']['visited_dependency_names']);visible=[]
  for dep in dependencies:
   packages=sorted(explicit.get(dep,set()));dep_counts[dep]+=1
   if len(packages)==1:
    mapping[dep]={'status':'exact_importFrom_origin','package':packages[0]};origins[packages[0]]+=1
   elif len(packages)>1:
    mapping[dep]={'status':'ambiguous_multiple_importFrom_origins','packages':packages};reasons.append('multiple_explicit_origins:'+dep)
   elif wildcards:
    mapping[dep]={'status':'unresolved_under_wildcard_imports','packages':sorted(wildcards)};reasons.append('wildcard_origin_unresolved:'+dep)
   else:
    mapping[dep]={'status':'namespace_origin_absent'};reasons.append('namespace_origin_absent:'+dep)
   if dep in direct or dep in visited:visible.append(dep)
   else:reasons.append('dependency_not_visible_in_analyzer_reference_inventory:'+dep)
  raw_mode=hold['policy_mode'];policy_supported=raw_mode in {'full_document','complete_span'}
  exact=not reasons and len(visible)==len(dependencies)
  if exact and policy_supported:status='candidate_namespace_origin_evidence'
  elif exact:status='hold_prediction_policy_unsupported_after_namespace_repair';reasons.append('prediction_policy_unsupported')
  else:status='hold_namespace_evidence_insufficient'
  counts[status]+=1
  decisions.append({'row_id':rid,'status':status,'reason_codes':sorted(reasons),'dependencies':dependencies,'dependency_origin':mapping,'all_dependencies_visible_in_preedit_reference_inventory':len(visible)==len(dependencies),'target_or_gold_used':False,'prediction_policy_mode':raw_mode,'namespace':{'path':sem['namespace']['path'],'sha256':ev['parsed_sha256'],'directives_sha256':hashlib.sha256(json.dumps(ev['directives'],sort_keys=True,separators=(',',':')).encode()).hexdigest()},'source':{'path':sem['source_path'],'sha256':sem['source_sha256'],'preedit_sha256':pred['preedit_sha256']}})
 with (a.output/'decisions.jsonl').open('x') as f:
  for x in decisions:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 candidates=[x['row_id'] for x in decisions if x['status']=='candidate_namespace_origin_evidence'];held=[x['row_id'] for x in decisions if x['status']!='candidate_namespace_origin_evidence']
 manifest={'schema':'sepalith.dat10.semantic_external_holds_audit.v1','status':'complete_review_only','training_admission':False,'denominator':EXPECTED,'counts':dict(counts),'candidate_ids':candidates,'candidate_ids_sha256':hashlib.sha256(('\n'.join(candidates)+'\n').encode()).hexdigest(),'held_ids':held,'dependencies':dict(dep_counts.most_common()),'origin_package_counts':dict(origins.most_common()),'namespace_files':len(namespaces),'inputs':{'holds_sha256':HOLD_SHA,'semantic_sha256':SEMANTIC_SHA,'prediction_sha256':PREDICTION_SHA},'outputs':{'decisions.jsonl':{'rows':len(decisions),'sha256':sha(a.output/'decisions.jsonl')},'namespace-evidence.jsonl':{'rows':len(evidence),'sha256':sha(raw)}},'evidence_contract':{'provided':'exact parse-only importFrom(package,symbol) directive and source/NAMESPACE hashes','already_visible':'dependency call/reference shape in pre-edit source','not_claimed':'external implementation, runtime behavior, full external function signature, or semantic correctness','selection_target_or_gold_used':False},'elapsed_seconds':time.monotonic()-started}
 atomic(a.output/'manifest.json',manifest);print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
