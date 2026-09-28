#!/usr/bin/env python3
"""Validate shard bindings and reconstruct target-free pre-edit prediction inputs."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path

HERE=Path(__file__).parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
B=load('semantic763_base',HERE/'source/base_materializer.py')
SUPPORTED='semantic_supported_context_closure_root_review_required'
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
class Writer:
 def __init__(self,path):self.path=path;self.tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');self.f=self.tmp.open('x');self.n=0
 def write(self,x):self.f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');self.n+=1
 def close(self):self.f.flush();os.fsync(self.f.fileno());self.f.close();self.tmp.replace(self.path)
def main():
 ap=argparse.ArgumentParser()
 for name in ('semantic_manifest','semantic_ledger','provenance_ledger','candidate_packets'):ap.add_argument('--'+name.replace('_','-'),type=Path,required=True);ap.add_argument('--expected-'+name.replace('_','-')+'-sha256',required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh_output_required')
 a.output.mkdir(parents=True)
 for name in ('semantic_manifest','semantic_ledger','provenance_ledger','candidate_packets'):
  p=getattr(a,name);expected=getattr(a,'expected_'+name+'_sha256')
  if B.sha(p)!=expected:raise ValueError('input_hash_mismatch:'+name)
 manifest=json.loads(a.semantic_manifest.read_text());code=manifest.get('code',{});out=manifest.get('output',{})
 expected_code={'analyzer_sha256':B.SEMANTIC_SHA,'scope_helper_sha256':B.SEMANTIC_SCOPE_SHA,'namespace_helper_sha256':B.SEMANTIC_NAMESPACE_SHA}
 if manifest.get('status')!='complete_review_only' or manifest.get('exact_id_closure') is not True or code!=expected_code or out.get('sha256')!=B.sha(a.semantic_ledger):raise ValueError('semantic_manifest_closure_invalid')
 sem,semsha,semn=B.exact_rows(a.semantic_ledger);prov,provsha,provn=B.exact_rows(a.provenance_ledger);packets,packetsha,packetn=B.exact_rows(a.candidate_packets)
 supported=sorted(r for r,x in sem.items() if x.get('status')==SUPPORTED and x.get('reasons')==[])
 pred=Writer(a.output/'prediction-inputs.jsonl');train=Writer(a.output/'training-sidecar.jsonl');holds=Writer(a.output/'preparation-holds.jsonl');source_cache={}
 for rid in supported:
  try:
   semantic=sem[rid];provenance=prov[rid];packet=packets[rid];identity=B.validate_identity(semantic,provenance,packet)
   source_path=Path(semantic['source_path']);cache_key=(str(source_path),semantic['source_sha256'])
   if cache_key in source_cache:
    raw,identity_stat=source_cache[cache_key];now=source_path.stat();current=(now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)
    if current!=identity_stat:raise B.Hold('source_changed_after_cached_read')
   else:
    before=source_path.stat();raw=source_path.read_bytes();after=source_path.stat();identity_stat=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=identity_stat or len(raw)!=before.st_size:raise B.Hold('source_stat_unstable')
    source_cache[cache_key]=(raw,identity_stat)
   if sha_bytes(raw)!=semantic['source_sha256']:raise B.Hold('source_pin_failed')
   if semantic.get('target_occurrence_method')=='exact_bytes':parse_bytes=raw
   elif semantic.get('target_occurrence_method')=='uniform_crlf_to_lf' and b'\r\n' in raw and b'\r' not in raw.replace(b'\r\n',b''):parse_bytes=raw.replace(b'\r\n',b'\n')
   else:raise B.Hold('unsupported_or_unproven_eol_normalization')
   source_lf=parse_bytes.decode('utf-8');parse_sha=sha_bytes(parse_bytes)
   target_lines=list(packet.get('result',{}).get('target_body',[]))
   if not target_lines or any(not isinstance(x,str) or '\n' in x or '\r' in x for x in target_lines):raise B.Hold('target_body_invalid')
   target=('\n'.join(target_lines)+'\n').encode()
   if semantic.get('parse_bytes_sha256')!=parse_sha or semantic.get('scope',{}).get('parsed_source_sha256')!=parse_sha:raise B.Hold('semantic_parse_byte_identity_failed')
   if not B.SEMANTIC.namespace_ok(semantic.get('namespace',{})):raise B.Hold('semantic_namespace_evidence_invalid')
   if sha_bytes(target)!=semantic.get('target_sha256') or len(target)!=semantic.get('target_bytes') or parse_bytes.count(target)!=1:raise B.Hold('target_hash_size_or_occurrence_failed')
   item={k:semantic[k] for k in ('target_occurrences','target_definition_name','documented_params','imported_symbols','prior_reviewed_recovery')}
   reasons,resolution,closure=B.SEMANTIC.decide(item,semantic.get('scope'),source_lf)
   if reasons or resolution.get('unresolved') or closure!=semantic.get('context_closure'):raise B.Hold('semantic_recheck_failed:'+','.join(reasons))
   target_start=parse_bytes.index(target);target_start_line=parse_bytes[:target_start].count(b'\n');target_span=B.checked_span(closure['target_definition_span'],len(source_lf.split('\n')),'target_definition')
   base=B.PROTOCOL.PromptContext.from_mapping(packet['result']['context'])
   B._validate_candidate_window(base,packet,target_lines,source_lf,target_start_line)
   selected,full_before,derived,_=B._production_context(base,raw,target_lines,target_span)
   if B._apply_global(full_before,selected,target_lines)!=raw.decode('utf-8'):raise B.Hold('full_source_reapplication_failed')
   used=set(semantic.get('reference_resolution',{}).get('visited_dependency_names',[]));imports=set(semantic.get('imported_symbols',[]));external=sorted(used&imports)
   relative_path=provenance.get('normalized_relative_source_path')
   if not isinstance(relative_path,str) or not relative_path or relative_path.startswith('/') or '\\' in relative_path:raise B.Hold('normalized_relative_source_path_invalid')
   pred.write({'schema':'sepalith.dat10.semantic763.prediction_input.v1','row_id':rid,'path':relative_path,'preedit_text':full_before,'preedit_sha256':sha_bytes(full_before.encode()),'cursor':selected.replacement_range.start.to_dict(),'document_eol':selected.document_eol,'required_helper_names':[x['name'] for x in closure['required_helper_spans']],'external_import_dependencies':external})
   train.write({'schema':'sepalith.dat10.semantic763.training_sidecar.v1','row_id':rid,'identity':identity,'target_lines':target_lines,'target_sha256':semantic['target_sha256'],'postedit_source_sha256':sha_bytes(raw),'external_import_dependencies':external,'required_helper_spans':closure['required_helper_spans']})
  except Exception as e:holds.write({'row_id':rid,'stage':'preparation','reason':type(e).__name__+':'+str(e),'silent_drop':False})
 for w in (pred,train,holds):w.close()
 summary={'schema':'sepalith.dat10.semantic763.preparation_manifest.v1','status':'complete_with_explicit_holds' if holds.n else 'complete','semantic_rows':semn,'semantic_supported':len(supported),'prediction_inputs':pred.n,'training_sidecar_rows':train.n,'preparation_holds':holds.n,'exact_supported_id_accounting':pred.n+holds.n==len(supported),'inputs':{'semantic_manifest':{'sha256':B.sha(a.semantic_manifest)},'semantic_ledger':{'sha256':semsha,'rows':semn},'provenance_ledger':{'sha256':provsha,'rows':provn},'candidate_packets':{'sha256':packetsha,'rows':packetn}},'outputs':{x.name:{'sha256':B.sha(x),'bytes':x.stat().st_size,'rows':n} for x,n in [(a.output/'prediction-inputs.jsonl',pred.n),(a.output/'training-sidecar.jsonl',train.n),(a.output/'preparation-holds.jsonl',holds.n)]}}
 (a.output/'preparation-manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
