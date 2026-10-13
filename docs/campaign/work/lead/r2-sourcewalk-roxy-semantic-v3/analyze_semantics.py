#!/usr/bin/env python3
"""Conservative semantic/context closure for provenance-passing roxygen rows."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path
from typing import Any

HERE=Path(__file__).parent
REVIEW=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-roxy-recovery-union-review-v2')
SAFE_IDS=REVIEW/'safe-review-union-ids.json';SAFE_SHA='b7700a8b67e1bb2bace0411a2c14a8697516e7c5cd23dbcd85e5e2e570e48962'
SOURCE_LEDGER=REVIEW/'source-set-ledger.jsonl';SOURCE_LEDGER_SHA='7f58405c012cebcdc0f75dd4d896c0db177483ef1c2a5b75ad6240dd7f53f328'
SCOPE_REFERENCE=HERE/'reviewed_scope_reference.R';SCOPE_REFERENCE_SHA='e02b4bd66438c93c472c84622e3d244a6c9efb71275a121058f8b2d6a11c0f7b'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path:Path,value:Any):
 q=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with q.open('x') as f:f.write(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 q.replace(path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def stable(path:Path):
 a=path.stat();b=path.read_bytes();z=path.stat();ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns)
 return b,ident(a)==ident(z) and len(b)==a.st_size
def packet_map(paths:list[Path],ids:set[str]):
 out={};pins=[]
 for path in paths:
  digest=hashlib.sha256()
  with path.open('rb') as f:
   for line in f:
    digest.update(line);x=json.loads(line);rid=x.get('row_ref',{}).get('row_id')
    if rid in ids:
     if rid in out:raise RuntimeError('duplicate packet row')
     out[rid]=x
  pins.append({'path':str(path),'sha256':digest.hexdigest()})
 return out,pins
def target_name(source:str,target:str)->str|None:
 pos=source.find(target)
 if pos<0:return None
 tail=source[pos+len(target):]
 match=re.match(r'\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=)\s*function\s*\(',tail)
 return match.group(1) if match else None
def doc_params(target:str)->set[str]:
 out=set()
 for line in target.splitlines():
  match=re.match(r"\s*#'\s*@param\s+([^\s]+)",line)
  if match:
   for name in match.group(1).split(','):
    if name.strip():out.add(name.strip())
 return out
def namespace_symbols(package_root:Path)->tuple[set[str],dict]:
 path=package_root/'NAMESPACE'
 if not path.is_file():return set(),{'path':str(path),'present':False}
 raw,ok=stable(path);text=raw.decode('utf-8','replace');names=set()
 for body in re.findall(r'importFrom\s*\(([^)]*)\)',text,re.S):
  parts=[x.strip().strip('"\'') for x in body.split(',')]
  names.update(x for x in parts[1:] if re.fullmatch(r'[A-Za-z.][A-Za-z0-9._]*',x))
 return names,{'path':str(path),'present':True,'sha256':hashlib.sha256(raw).hexdigest(),'stat_stable':ok}
def decide(item:dict,s:dict|None)->tuple[list[str],dict,dict]:
 reasons=[];resolution={};closure={'target_definition_span':s.get('target_definition_span') if s else None,'required_helper_spans':[],'stable_source_order':True,'target_roxygen_excluded_from_prompt':True,'full_target_preserved':True,'no_length_drop':True}
 if item['target_occurrences']!=1:reasons.append('target_not_exactly_once_in_reopened_source')
 if not item['target_definition_name']:reasons.append('target_not_adjacent_to_named_function_definition')
 if s is None:reasons.append('scope_inventory_unavailable')
 elif s['status']!='scope_inventory_complete':reasons.append(s['status'])
 else:
  formals=set(s['function_formals'])-{'...'};documented=set(item['documented_params'])
  if documented-formals:reasons.append('invented_documented_formal:'+','.join(sorted(documented-formals)))
  if formals-documented:reasons.append('undocumented_function_formal:'+','.join(sorted(formals-documented)))
  refs=set(s['variable_references'])|set(s['call_heads']);top=set(s['top_level_definitions']);resolved=set(s['base_bound'])|top|set(item['imported_symbols'])|formals|{item['target_definition_name']};unresolved=refs-resolved
  if unresolved:reasons.append(('reviewed_recovery_binding_required_for_new_source_identity:' if item['prior_reviewed_recovery'] else 'unresolved_global_or_nse_requires_occurrence_evidence:')+','.join(sorted(unresolved)))
  spans=s.get('top_level_definition_spans',{});helpers=[]
  for name in sorted((refs&top)-{item['target_definition_name']}):
   span=spans.get(name)
   if s.get('top_level_definition_counts',{}).get(name)!=1:reasons.append('helper_definition_not_unique:'+name)
   elif not isinstance(span,list) or len(span)!=2 or not all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in span):reasons.append('helper_definition_span_unavailable:'+name)
   else:helpers.append({'name':name,'span':span})
  helpers.sort(key=lambda x:(x['span'][0],x['name']));closure['required_helper_spans']=helpers
  resolution={'references':sorted(refs),'resolved':sorted(refs&resolved),'unresolved':sorted(unresolved),'helper_spans':helpers}
 return reasons,resolution,closure
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--provenance-ledger',type=Path,required=True);ap.add_argument('--candidate-packets',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 a.output.mkdir(parents=True)
 started=time.monotonic();rows=[]
 with a.provenance_ledger.open() as f:
  for line in f:
   x=json.loads(line)
   if x.get('family')=='roxygen_drafting' and x.get('status')=='provenance_pass_semantic_analyzer_queued':rows.append(x)
 ids={x['row_id'] for x in rows};packets,packet_pins=packet_map(a.candidate_packets,ids)
 if set(packets)!=ids:raise RuntimeError(f'packet join incomplete {len(packets)}/{len(ids)}')
 if sha(SAFE_IDS)!=SAFE_SHA or sha(SOURCE_LEDGER)!=SOURCE_LEDGER_SHA or sha(SCOPE_REFERENCE)!=SCOPE_REFERENCE_SHA:raise RuntimeError('reviewed evidence pin mismatch')
 reviewed=json.loads(SAFE_IDS.read_text());reviewed_ids=set(reviewed.get('ids',reviewed) if isinstance(reviewed,dict) else reviewed)
 prepared=[];groups=collections.defaultdict(list);source_pins={}
 for row in rows:
  rid=row['row_id'];p=packets[rid];source_path=Path(p['validation']['source_path']);raw,ok=stable(source_path)
  if not ok or hashlib.sha256(raw).hexdigest()!=p['validation']['source_sha256']:raise RuntimeError(f'source pin failed:{rid}')
  text=raw.decode('utf-8');target='\n'.join(p['result']['target_body'])+'\n';occ=text.count(target);name=target_name(text,target)
  root=Path(p['validation']['license_evidence']['path']).parent;imports,ns=namespace_symbols(root)
  item={'row_id':rid,'source_path':str(source_path),'source_sha256':hashlib.sha256(raw).hexdigest(),'target_sha256':hashlib.sha256(target.encode()).hexdigest(),'target_bytes':len(target.encode()),'target_lines':len(p['result']['target_body']),'target_occurrences':occ,'target_definition_name':name,'documented_params':sorted(doc_params(target)),'imported_symbols':sorted(imports),'namespace':ns,'prior_reviewed_recovery':rid in reviewed_ids}
  prepared.append(item)
  if occ==1 and name:groups[str(source_path)].append({'row_id':rid,'target_definition_name':name})
  source_pins[str(source_path)]={'source_path':str(source_path),'source_sha256':item['source_sha256'],'rows':groups[str(source_path)]}
 with tempfile.TemporaryDirectory(prefix='sepalith-roxy-semantic-') as d:
  inp=Path(d)/'input.json';out=Path(d)/'out.jsonl';inp.write_text(json.dumps({'source_groups':list(source_pins.values())}))
  done=subprocess.run(['Rscript','--vanilla',str(HERE/'semantic_scope.R'),str(inp),str(out)],capture_output=True,text=True,timeout=300)
  if done.returncode:raise RuntimeError('R scope infrastructure failure:'+done.stderr[-500:])
  scope={x['row_id']:x for x in map(json.loads,out.read_text().splitlines())}
 results=[]
 for item in prepared:
  rid=item['row_id'];s=scope.get(rid);reasons,resolution,closure=decide(item,s);item['reference_resolution']=resolution
  status='semantic_supported_context_closure_root_review_required' if not reasons else 'hold_semantic_evidence'
  results.append({**item,'scope':s,'status':status,'reasons':reasons,'context_closure':closure})
 ledger=a.output/'semantic-ledger.jsonl'
 with ledger.open('x') as f:
  for x in sorted(results,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 manifest={'schema':'sepalith.dat10.sourcewalk_roxy_semantic.v3','status':'complete_review_only','rows':len(results),'status_counts':dict(collections.Counter(x['status'] for x in results)),'reason_counts':dict(collections.Counter(r for x in results for r in x['reasons'])),'inputs':{'provenance_ledger':{'path':str(a.provenance_ledger),'sha256':sha(a.provenance_ledger)},'candidate_packets':packet_pins},'reviewed_bindings':{'safe_ids_sha256':SAFE_SHA,'source_ledger_sha256':SOURCE_LEDGER_SHA,'scope_reference_sha256':SCOPE_REFERENCE_SHA},'output':{'path':str(ledger),'sha256':sha(ledger),'bytes':ledger.stat().st_size,'rows':len(results)},'elapsed_seconds':time.monotonic()-started,'training_admission':False}
 atomic(a.output/'manifest.json',manifest);print(json.dumps({'rows':len(results),'status_counts':manifest['status_counts']}))
if __name__=='__main__':main()
