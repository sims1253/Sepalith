#!/usr/bin/env python3
"""Fail-closed semantic and recursive context closure for roxygen rows."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path
from typing import Any
HERE=Path(__file__).parent
REVIEW=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-roxy-recovery-union-review-v2')
SAFE_IDS=REVIEW/'safe-review-union-ids.json';SAFE_SHA='b7700a8b67e1bb2bace0411a2c14a8697516e7c5cd23dbcd85e5e2e570e48962'
SOURCE_LEDGER=REVIEW/'source-set-ledger.jsonl';SOURCE_LEDGER_SHA='7f58405c012cebcdc0f75dd4d896c0db177483ef1c2a5b75ad6240dd7f53f328'
SCOPE_REFERENCE=HERE/'reviewed_scope_reference.R';SCOPE_REFERENCE_SHA='e02b4bd66438c93c472c84622e3d244c6c9efb71275a121058f8b2d6a11c0f7b';SCOPE_HELPER=HERE/'semantic_scope.R';NAMESPACE_HELPER=HERE/'namespace_scope.R'
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
 a=path.stat();raw=path.read_bytes();z=path.stat();ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns)
 return raw,ident(a)==ident(z) and len(raw)==a.st_size
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
 positions=[m.start() for m in re.finditer(re.escape(target),source)]
 if len(positions)!=1:return None
 match=re.match(r'\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=|<<-)\s*function\s*\(',source[positions[0]+len(target):])
 return match.group(1) if match else None
def doc_params(target:str)->set[str]:
 out=set()
 for line in target.splitlines():
  match=re.match(r"\s*#'\s*@param\s+([^\s]+)",line)
  if match:out.update(x.strip() for x in match.group(1).split(',') if x.strip())
 return out
def occurrence(source:bytes,target:bytes)->tuple[int,str,bytes]:
 exact=source.count(target)
 if exact:return exact,'exact_bytes',source
 if b'\r\n' in source and b'\r' not in source.replace(b'\r\n',b''):
  normalized=source.replace(b'\r\n',b'\n');count=normalized.count(target)
  if count:return count,'uniform_crlf_to_lf',normalized
 return 0,'no_match',source
def namespace_inventories(package_roots:set[Path])->dict[str,tuple[set[str],dict]]:
 result={};pending=[]
 for number,package_root in enumerate(sorted(package_roots,key=str)):
  path=package_root/'NAMESPACE';key=str(package_root)
  if not path.is_file():result[key]=(set(),{'path':str(path),'present':False,'stat_stable':True,'status':'not_present'});continue
  raw,ok=stable(path);digest=hashlib.sha256(raw).hexdigest();pending.append({'id':str(number),'key':key,'path':path,'raw':raw,'sha256':digest,'stat_stable':ok})
 if not pending:return result
 with tempfile.TemporaryDirectory(prefix='sepalith-namespace-semantic-') as d:
  root=Path(d);groups=[]
  for item in pending:
   copy=root/f"namespace-{item['id']}.txt";copy.write_bytes(item['raw']);groups.append({'id':item['id'],'path':str(copy)})
  inp=root/'input.json';out=root/'out.jsonl';inp.write_text(json.dumps({'namespaces':groups}));done=subprocess.run(['Rscript','--vanilla',str(NAMESPACE_HELPER),str(inp),str(out)],capture_output=True,text=True,timeout=300)
  if done.returncode:raise RuntimeError('R namespace infrastructure failure:'+done.stderr[-500:])
  parsed={x['id']:x for x in map(json.loads,out.read_text().splitlines())}
 if set(parsed)!={x['id'] for x in pending}:raise RuntimeError('R namespace inventory ID closure failed')
 for item in pending:
  evidence=parsed[item['id']]
  if evidence.get('parsed_sha256')!=item['sha256']:raise RuntimeError('R namespace parsed byte hash mismatch:'+item['key'])
  detail={'path':str(item['path']),'present':True,'sha256':item['sha256'],'stat_stable':item['stat_stable'],'status':evidence.get('status'),'parsed_sha256':evidence.get('parsed_sha256'),'malformed':evidence.get('malformed',[]),'error':evidence.get('error')}
  result[item['key']]=(set(evidence.get('imported_symbols',[])),detail)
 return result
def namespace_ok(evidence:dict)->bool:return evidence.get('stat_stable') is True and evidence.get('status') in {'not_present','namespace_inventory_complete'}
def valid_span(span,lines,name):
 return (isinstance(span,list) and len(span)==2 and all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in span) and span[0]<=span[1]<=len(lines) and re.search(r'(?<![A-Za-z0-9._])'+re.escape(name)+r'\s*(?:<-|=|<<-)\s*function\b','\n'.join(lines[span[0]-1:span[1]])) is not None)
def decide(item:dict,s:dict|None,source_text:str)->tuple[list[str],dict,dict]:
 reasons=[];closure={'target_definition_span':s.get('target_definition_span') if s else None,'required_helper_spans':[],'stable_source_order':True,'target_roxygen_excluded_from_prompt':True,'full_target_preserved':True,'no_length_drop':True};resolution={}
 if item['target_occurrences']!=1:reasons.append('target_not_exactly_once_in_reopened_source')
 if not item['target_definition_name']:reasons.append('target_not_adjacent_to_named_function_definition')
 if s is None:reasons.append('scope_inventory_unavailable');return reasons,resolution,closure
 if s.get('status')!='scope_inventory_complete':reasons.append(s.get('status','scope_status_missing'));return reasons,resolution,closure
 lines=source_text.splitlines();target=item['target_definition_name'];span=s.get('target_definition_span')
 if not valid_span(span,lines,target):reasons.append('target_definition_span_missing_or_not_source_backed')
 formals=set(s.get('function_formals',[]));documented=set(item['documented_params'])
 if documented-formals:reasons.append('invented_documented_formal:'+','.join(sorted(documented-formals)))
 if formals-documented:reasons.append('undocumented_function_formal:'+','.join(sorted(formals-documented)))
 top=set(s.get('top_level_definitions',[]));spans=s.get('top_level_definition_spans',{});counts=s.get('top_level_definition_counts',{});deps=s.get('function_dependencies',{})
 base=set(s.get('base_bound',[]));imports=set(item['imported_symbols']);direct=set(s.get('variable_references',[]))|set(s.get('call_heads',[]))
 # A formal is lexical to one function. Carry its owning function through the
 # work queue so a target formal can never satisfy a helper's free variable.
 queue=[(name,target,formals) for name in sorted(direct)];visited=set();unresolved=set();helpers=[];helper_names=set()
 while queue:
  name,owner,owner_formals=queue.pop(0);ref=(owner,name)
  if ref in visited:continue
  visited.add(ref)
  if name in owner_formals or name==target:continue
  # Package/source-local bindings shadow imports and base bindings in R's
  # lexical lookup. Resolve them first and include their source spans.
  if name not in top:
   if name in imports or name in base:continue
   unresolved.add(name);continue
  helper_span=spans.get(name)
  if counts.get(name)!=1:reasons.append('helper_definition_not_unique:'+name);continue
  if not valid_span(helper_span,lines,name):reasons.append('helper_span_missing_or_not_source_backed:'+name);continue
  if name not in helper_names:helpers.append({'name':name,'span':helper_span});helper_names.add(name)
  dep=deps.get(name)
  if not isinstance(dep,dict) or not dep.get('codetools_ok'):reasons.append('helper_dependency_inventory_incomplete:'+name);continue
  helper_formals=set(dep.get('formals',[]));base.update(dep.get('base_bound',[]))
  queue.extend((child,name,helper_formals) for child in sorted(set(dep.get('variables',[]))|set(dep.get('call_heads',[]))))
 if unresolved:reasons.append(('reviewed_recovery_binding_required_for_new_source_identity:' if item['prior_reviewed_recovery'] else 'unresolved_global_or_nse_requires_occurrence_evidence:')+','.join(sorted(unresolved)))
 helpers.sort(key=lambda x:(x['span'][0],x['name']));closure['required_helper_spans']=helpers
 resolution={'direct_references':sorted(direct),'visited_dependency_references':[{'owner':owner,'name':name} for owner,name in sorted(visited)],'visited_dependency_names':sorted({name for _,name in visited}),'unresolved':sorted(unresolved),'helper_spans':helpers}
 return reasons,resolution,closure
def register_source_variant(source_pins, groups, source_path, parse_bytes, item, occurrence_count, name):
 # A filename can yield different parse reconstructions for different targets.
 key=(str(source_path),item['parse_bytes_sha256'])
 if hashlib.sha256(parse_bytes).hexdigest()!=key[1]:raise RuntimeError('parse variant bytes/hash differ')
 if occurrence_count==1 and name:groups[key].append({'row_id':item['row_id'],'target_definition_name':name})
 source_pins[key]={'parse_bytes':parse_bytes,'parse_bytes_sha256':key[1],'rows':groups[key]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--provenance-ledger',type=Path,required=True);ap.add_argument('--candidate-packets',type=Path,action='append',required=True);ap.add_argument('--expected-candidate-packets-sha256',action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 if len(a.candidate_packets)!=len(a.expected_candidate_packets_sha256):raise ValueError('one expected packet SHA per packet path required')
 a.output.mkdir(parents=True);started=time.monotonic();rows=[]
 with a.provenance_ledger.open() as f:
  for line in f:
   x=json.loads(line)
   if x.get('family')=='roxygen_drafting' and x.get('status')=='provenance_pass_semantic_analyzer_queued':rows.append(x)
 ids={x['row_id'] for x in rows};packets,packet_pins=packet_map(a.candidate_packets,ids)
 for observed,expected in zip(packet_pins,a.expected_candidate_packets_sha256):
  if observed['sha256']!=expected:raise RuntimeError('candidate packet hash differs from provenance binding')
 if set(packets)!=ids:raise RuntimeError(f'packet join incomplete {len(packets)}/{len(ids)}')
 if sha(SAFE_IDS)!=SAFE_SHA or sha(SOURCE_LEDGER)!=SOURCE_LEDGER_SHA or sha(SCOPE_REFERENCE)!=SCOPE_REFERENCE_SHA:raise RuntimeError('reviewed evidence pin mismatch')
 reviewed_ids=set(json.loads(SAFE_IDS.read_text())['ids']);prepared=[];groups=collections.defaultdict(list);source_pins={}
 package_roots={Path(packets[rid]['validation']['license_evidence']['path']).parent for rid in ids};namespace_cache=namespace_inventories(package_roots)
 for row in rows:
  rid=row['row_id'];p=packets[rid];source_path=Path(p['validation']['source_path']);raw,ok=stable(source_path)
  if not ok or hashlib.sha256(raw).hexdigest()!=p['validation']['source_sha256']:raise RuntimeError(f'source pin failed:{rid}')
  target='\n'.join(p['result']['target_body'])+'\n';occ,method,parse_bytes=occurrence(raw,target.encode());text=parse_bytes.decode('utf-8');name=target_name(text,target);package_root=Path(p['validation']['license_evidence']['path']).parent;imports,ns=namespace_cache[str(package_root)]
  item={'row_id':rid,'source_path':str(source_path),'source_sha256':hashlib.sha256(raw).hexdigest(),'parse_bytes_sha256':hashlib.sha256(parse_bytes).hexdigest(),'source_text':text,'target_sha256':hashlib.sha256(target.encode()).hexdigest(),'target_bytes':len(target.encode()),'target_lines':len(p['result']['target_body']),'target_occurrences':occ,'target_occurrence_method':method,'target_definition_name':name,'documented_params':sorted(doc_params(target)),'imported_symbols':sorted(imports),'namespace':ns,'prior_reviewed_recovery':rid in reviewed_ids};prepared.append(item)
  register_source_variant(source_pins,groups,source_path,parse_bytes,item,occ,name)
 with tempfile.TemporaryDirectory(prefix='sepalith-roxy-semantic-') as d:
  root=Path(d);r_groups=[]
  for number,value in enumerate(source_pins.values()):
   copy=root/f'source-{number:06d}.R';copy.write_bytes(value['parse_bytes']);r_groups.append({'source_path':str(copy),'source_sha256':value['parse_bytes_sha256'],'rows':value['rows']})
  inp=root/'input.json';out=root/'out.jsonl';inp.write_text(json.dumps({'source_groups':r_groups}));done=subprocess.run(['Rscript','--vanilla',str(SCOPE_HELPER),str(inp),str(out)],capture_output=True,text=True,timeout=300)
  if done.returncode:raise RuntimeError('R scope infrastructure failure:'+done.stderr[-500:])
  scope={x['row_id']:x for x in map(json.loads,out.read_text().splitlines())}
 results=[]
 for item in prepared:
  rid=item['row_id'];s=scope.get(rid)
  if s and s.get('parsed_source_sha256')!=item['parse_bytes_sha256']:raise RuntimeError(f'R parsed byte hash mismatch:{rid}')
  if not namespace_ok(item['namespace']):reasons=['namespace_stat_changed'];resolution={};closure={'target_definition_span':None,'required_helper_spans':[],'stable_source_order':False,'target_roxygen_excluded_from_prompt':True,'full_target_preserved':True,'no_length_drop':True}
  else:reasons,resolution,closure=decide(item,s,item['source_text'])
  item.pop('source_text');item['reference_resolution']=resolution;status='semantic_supported_context_closure_root_review_required' if not reasons else 'hold_semantic_evidence';results.append({**item,'scope':s,'status':status,'reasons':reasons,'context_closure':closure})
 if {x['row_id'] for x in results}!=ids or len(results)!=len(ids):raise RuntimeError('analyzer exact ID closure failed')
 ledger=a.output/'semantic-ledger.jsonl'
 with ledger.open('x') as f:
  for x in sorted(results,key=lambda y:y['row_id']):f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 manifest={'schema':'sepalith.dat10.sourcewalk_roxy_semantic.v6','status':'complete_review_only','rows':len(results),'status_counts':dict(collections.Counter(x['status'] for x in results)),'reason_counts':dict(collections.Counter(r for x in results for r in x['reasons'])),'inputs':{'provenance_ledger':{'path':str(a.provenance_ledger),'sha256':sha(a.provenance_ledger)},'candidate_packets':packet_pins},'code':{'analyzer_sha256':sha(Path(__file__).resolve()),'scope_helper_sha256':sha(SCOPE_HELPER),'namespace_helper_sha256':sha(NAMESPACE_HELPER)},'reviewed_bindings':{'safe_ids_sha256':SAFE_SHA,'source_ledger_sha256':SOURCE_LEDGER_SHA,'scope_reference_sha256':SCOPE_REFERENCE_SHA},'output':{'path':str(ledger),'sha256':sha(ledger),'bytes':ledger.stat().st_size,'rows':len(results)},'exact_id_closure':True,'elapsed_seconds':time.monotonic()-started,'training_admission':False}
 atomic(a.output/'manifest.json',manifest);print(json.dumps({'rows':len(results),'status_counts':manifest['status_counts']}))
if __name__=='__main__':main()
