#!/usr/bin/env python3
"""V2 union audit: terminal-manifest binding and input/candidate partition."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import audit_expanded_union_v1 as V1
SCHEMA='sepalith.sft11.expanded-union-audit.v2';INPUTS={'accepted_current_20191':20191,'finalized_semantic10948':10948,'eventual_semantic9534':9534,'noop4100':4100}
class AuditError(RuntimeError):pass
def req(v,m):
 if not v:raise AuditError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pin(x,label):
 req(isinstance(x,dict),label+':pin');p=Path(x.get('path',''));req(p.is_file(),label+':missing');req(sha(p)==x.get('sha256'),label+':sha');req(type(x.get('rows'))is int and x['rows']>=0,label+':rows');return p
def refs(value,base):
 out=set()
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str) and type(x.get('rows'))is int:
    p=Path(x['path']);out.add((str((base/p).resolve()) if not p.is_absolute() else str(p.resolve()),x['sha256'],x['rows']))
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(value);return out
def manifest_pin(x,known,label):
 p=pin(x,label);req((str(p.resolve()),x['sha256'],x['rows']) in known,label+':not_in_terminal_manifest');return p
def ids(path,label):
 out=[]
 with path.open() as stream:
  for n,line in enumerate(stream,1):
   if line.strip():
    x=json.loads(line);rid=x.get('row_id') or x.get('id');req(isinstance(rid,str) and rid,f'{label}:{n}:id');out.append(rid)
 req(len(out)==len(set(out)),label+':duplicate_id');return out
def validate(cohort):
 name=cohort.get('name');req(name in INPUTS,name+':name');req(cohort.get('input_rows')==INPUTS[name],name+':input_denominator')
 if cohort.get('terminal_manifest') is None:return None
 mp=pin(cohort['terminal_manifest'],name+':manifest');known=refs(json.loads(mp.read_text()),mp.parent);tokens=cohort.get('token_rows');provs=cohort.get('provenance_rows');req(isinstance(tokens,list) and isinstance(provs,list) and len(tokens)==len(provs)>0,name+':candidate_files')
 token_paths=[manifest_pin(x,known,name+':token') for x in tokens];[manifest_pin(x,known,name+':provenance') for x in provs]
 candidate=sum(x['rows'] for x in tokens);req(candidate==sum(x['rows'] for x in provs)==cohort.get('candidate_rows'),name+':candidate_denominator')
 ledger=cohort.get('decision_ledger');lp=manifest_pin(ledger,known,name+':ledger');req(ledger['rows']==cohort['input_rows'],name+':ledger_denominator')
 classes=cohort.get('decision_status_classes');req(set(classes or {})=={'candidate','hold','exact_duplicate'},name+':classes');reverse={s:k for k,v in classes.items() for s in v};req(len(reverse)==sum(map(len,classes.values())),name+':overlap_classes');part={k:[] for k in classes}
 with lp.open() as stream:
  for n,line in enumerate(stream,1):
   if line.strip():
    x=json.loads(line);rid=x.get('row_id') or x.get('id');status=x.get('status');req(isinstance(rid,str) and rid and status in reverse,f'{name}:ledger:{n}');part[reverse[status]].append(rid)
 allids=sum(part.values(),[]);req(len(allids)==INPUTS[name] and len(set(allids))==len(allids),name+':partition_ids');token_ids=sum((ids(p,name+':token') for p in token_paths),[]);req(token_ids==part['candidate'],name+':candidate_id_partition')
 counts={k:len(v) for k,v in part.items()};req(cohort.get('partition_counts')==counts and sum(counts.values())==INPUTS[name],name+':partition_counts');return candidate,part
def audit(spec):
 req(spec.get('schema')==SCHEMA,'spec_schema');cs=spec.get('cohorts');req(isinstance(cs,list) and [x.get('name') for x in cs]==list(INPUTS),'cohort_order');bound={c['name']:validate(c) for c in cs}
 translated=copy.deepcopy(spec);translated['schema']=V1.SCHEMA
 for c in translated['cohorts']:
  value=bound[c['name']];c['expected_rows']=INPUTS[c['name']] if value is None else value[0]
  if value is None:c.update(manifest=None,token_rows=None,provenance_rows=None)
  else:c['manifest']=c.pop('terminal_manifest')
  for k in ('input_rows','candidate_rows','decision_ledger','decision_status_classes','partition_counts'):c.pop(k,None)
 old=V1.EXPECTED_COHORTS;V1.EXPECTED_COHORTS={c['name']:c['expected_rows'] for c in translated['cohorts']}
 try:rows,report=V1.audit(translated)
 finally:V1.EXPECTED_COHORTS=old
 audited={x['row_id'] for x in rows};partitions={}
 for name,value in bound.items():
  if value is None:continue
  candidate,part=value;req(set(part['candidate'])<=audited,name+':audited_candidates');partitions[name]={k:len(v) for k,v in part.items()}
  for status in ('hold','exact_duplicate'):
   rows.extend({'row_id':rid,'cohort':name,'status':'source_'+status,'reasons':['terminal_materializer_partition'],'silent_drop':False} for rid in part[status])
 report.update(schema='sepalith.sft11.expanded-union-audit.result.v2',input_rows=sum(INPUTS.values()),candidate_rows=sum(v[0] for v in bound.values() if v),input_denominators=INPUTS,source_partition_counts=partitions,unbound_input_cohorts=[n for n,v in bound.items() if v is None],admission_ready=False,training_admission=False);rows.sort(key=lambda x:x['row_id']);return rows,report
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();req(sha(a.spec)==a.spec_sha256,'spec_sha');rows,report=audit(json.loads(a.spec.read_text()));V1.write_atomic(a.output,rows,report);print(V1.canonical(report))
if __name__=='__main__':main()
