#!/usr/bin/env python3
"""Materialize root-accepted semantic-v6 TRAIN closures into PRM03 rows."""
from __future__ import annotations
import argparse,collections,dataclasses,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from typing import Any

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
SEMANTIC_PATH=PLAN/'docs/campaign/work/lead/r2-sourcewalk-roxy-semantic-v6/analyze_semantics.py'
SEMANTIC_SHA='54965cb3af14a125d945d94f24975eddff040b0706ab934b5a04543a12a99a57'
SEMANTIC_SCOPE_SHA='64d984139c2e5baf289f4caf8070903ff278f652a5a205f8141bc43627e84f61'
SEMANTIC_NAMESPACE_SHA='81196a5851f630b3af723d80914f488a03b58dcc8bb95256bfb131b47038587d'
PROTOCOL_PATH=PLAN/'docs/campaign/work/lead/r2-full-weight-edit-sft-preparation-v1/source/packages/sepalith/src/sepalith/campaign_protocol.py'
PROTOCOL_SHA='5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156'
STRICT_PATH=PLAN/'docs/campaign/work/lead/r2-roxy8597-root-fixes-v1/audit_authoritative_stream.py'
STRICT_SHA='7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37'
GEOMETRY_PATH=Path(__file__).with_name('semantic_context_geometry_v2.py')
SELECTION_PATH=Path(__file__).with_name('campaign_selection.py')
SELECTION_SHA='354a31c8f1f88cd9f1680c5e1e8dd917be3212776716a7481c80c367a545729e'
GEOMETRY_SHA='f38c3a96b5a8b34b420b289c00cf5de3f0c4fb51fba09ea3e634a61299339b6d'
TOKENIZER_PATH=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
NORMALIZED_ROOT=Path('/mnt/h/sepalith/normalized')

class Hold(ValueError):pass

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def load(name:str,path:Path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError('module_load_failed:'+str(path))
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
if sha(SEMANTIC_PATH)!=SEMANTIC_SHA or sha(PROTOCOL_PATH)!=PROTOCOL_SHA or sha(STRICT_PATH)!=STRICT_SHA or sha(GEOMETRY_PATH)!=GEOMETRY_SHA or sha(SELECTION_PATH)!=SELECTION_SHA:raise RuntimeError('frozen_source_pin_mismatch')
SEMANTIC=load('semantic_v6_materialization_binding',SEMANTIC_PATH)
PROTOCOL=load('campaign_protocol_semantic_materialization',PROTOCOL_PATH)
STRICT=load('strict_protocol_semantic_materialization',STRICT_PATH)
GEOMETRY=load('semantic_context_geometry_materialization_v3',GEOMETRY_PATH)
SELECTION=load('campaign_selection_materialization_v3',SELECTION_PATH)

class TokenizerAdapter:
 def __init__(self,path:Path=TOKENIZER_PATH):
  if sha(path)!=TOKENIZER_SHA:raise RuntimeError('tokenizer_pin_mismatch')
  from tokenizers import Tokenizer
  self.backend=Tokenizer.from_file(str(path));self.backend.encode_special_tokens=True
 def encode(self,text:str,*,add_special_tokens:bool=False,split_special_tokens:bool=True)->list[int]:
  if add_special_tokens is not False or split_special_tokens is not True:raise ValueError('tokenizer_policy_violation')
  return self.backend.encode(text,add_special_tokens=False).ids

def exact_rows(path:Path)->tuple[dict[str,dict],str,int]:
 digest=hashlib.sha256();rows={};count=0
 with path.open('rb') as f:
  for line in f:
   digest.update(line)
   if not line.strip():continue
   row=json.loads(line);rid=str(row.get('row_id') or row.get('row_ref',{}).get('row_id') or '')
   if not rid or rid in rows:raise ValueError('missing_or_duplicate_row_id:'+rid)
   rows[rid]=row;count+=1
 return rows,digest.hexdigest(),count

def checked_span(span:Any,line_count:int,label:str)->tuple[int,int]:
 if not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span):raise Hold(label+':invalid_span_type')
 start,end=span
 if start<1 or end<start or end>line_count:raise Hold(label+':span_out_of_range')
 return start-1,end

def render_runs(lines:list[str],spans:list[tuple[int,int]])->list[str]:
 out=[];last=None
 for start,end in spans:
  if last is not None and start>last:out.append('')
  out.extend(lines[start:end]);last=end
 return out

def length_bucket(sequence:int)->str:
 if sequence<=4096:return 'le_4096'
 if sequence<=8192:return '4097_8192'
 if sequence<=16384:return '8193_16384'
 if sequence<=32768:return '16385_32768'
 return 'gt_32768'

def validate_identity(semantic:dict,provenance:dict,packet:dict)->dict:
 rid=semantic['row_id'];ref=packet.get('row_ref',{});validation=packet.get('validation',{})
 checks={
  'provenance_status':provenance.get('status')=='provenance_pass_semantic_analyzer_queued',
  'provenance_family':provenance.get('family')=='roxygen_drafting',
  'license_positive':provenance.get('license_decision',{}).get('ok') is True and provenance.get('license_decision',{}).get('reason')=='positive_reviewed_allowed_license',
  'stable_source_and_description':provenance.get('source_stat_stable') is True and provenance.get('description_stat_stable') is True,
  'row_id_join':ref.get('row_id')==rid,
  'family_join':packet.get('family')=='roxygen_drafting' and ref.get('family')=='roxygen_drafting',
  'train_split':ref.get('split')=='train_group' and validation.get('parent_group_split')=='train_group',
  'train_source':ref.get('source')=='scenario_roxygen_drafting' and str(ref.get('file','')).startswith('/mnt/h/sepalith/datasets/scenarios_v1/'),
  'group_and_line_identity':isinstance(ref.get('group_id'),str) and bool(ref.get('group_id')) and type(ref.get('line')) is int and ref.get('line')>0 and isinstance(ref.get('raw_line_sha256'),str) and len(ref.get('raw_line_sha256'))==64,
  'package_join':provenance.get('package_id')==ref.get('package_id'),
  'source_path_join':semantic.get('source_path')==validation.get('source_path'),
  'normalized_source_path':Path(str(validation.get('source_path',''))).is_relative_to(NORMALIZED_ROOT),
  'source_hash_join':semantic.get('source_sha256')==validation.get('source_sha256'),
  'full_buffer_application':validation.get('full_buffer_application') is True,
  'normalized_parent_R_parse':validation.get('normalized_parent_R_parse') is True,
 }
 failed=sorted(k for k,v in checks.items() if not v)
 if failed:raise Hold('identity_or_train_membership:'+','.join(failed))
 return {'row_id':rid,'family':'roxygen_drafting','package_id':ref['package_id'],'group_id':ref.get('group_id'),'source_file':ref.get('file'),'source_line':ref.get('line'),'raw_line_sha256':ref.get('raw_line_sha256'),'source_path':semantic['source_path'],'source_sha256':semantic['source_sha256'],'split':'train_group','checks':checks}

def _selection_source(packet:dict)->tuple[str,str]:
 value=packet.get('result',{}).get('selection_source')
 if not isinstance(value,dict) or not {'text','content_sha256'}<=set(value):raise Hold('selection_source_contract_invalid')
 text=value.get('text');digest=value.get('content_sha256')
 if not isinstance(text,str) or sha_bytes(text.encode())!=digest:raise Hold('selection_source_hash_mismatch')
 return text,digest

def _validate_candidate_window(base:Any,packet:dict,target_lines:list[str],source_lf:str,target_start_line:int)->dict:
 selection_text,selection_sha=_selection_source(packet)
 try:editable=GEOMETRY.context_buffer_text(base.to_dict());after=GEOMETRY.apply_zero_width(base.to_dict(),target_lines)
 except ValueError as e:raise Hold('candidate_geometry_invalid:'+str(e)) from e
 if editable!=selection_text or base.replacement_range.content_sha256!=selection_sha:raise Hold('candidate_buffer_selection_source_mismatch')
 after_lf=after.replace('\r\n','\n');count=source_lf.count(after_lf)
 if count!=1:raise Hold('candidate_applied_window_not_exactly_once_in_source')
 offset=source_lf.index(after_lf)
 if offset and source_lf[offset-1]!='\n':raise Hold('candidate_window_not_line_aligned')
 global_line=source_lf[:offset].count('\n')+base.replacement_range.start.line
 if global_line!=target_start_line:raise Hold('candidate_local_to_global_anchor_mismatch')
 return {'selection_sha256':selection_sha,'selection_bytes':len(selection_text.encode()),'local_anchor_line':base.replacement_range.start.line,'mapped_global_anchor_line':global_line,'applied_window_sha256':sha_bytes(after.encode())}

def _apply_global(before:str,context:Any,target_lines:list[str])->str:
 sep=GEOMETRY.eol_text(context.document_eol);lines=before.split(sep);rr=context.replacement_range
 if not rr.is_empty or rr.start.character!=0 or rr.end.character!=0 or rr.start.line>=len(lines):raise Hold('global_zero_width_range_invalid')
 if lines[rr.start.line]!='':raise Hold('global_anchor_is_not_physical_blank')
 lines[rr.start.line]=sep.join(target_lines);return sep.join(lines)

def _production_context(base:Any,raw:bytes,target_lines:list[str],target_span:tuple[int,int])->tuple[Any,str,dict,Any]:
 try:derived=GEOMETRY.derive_full_before(raw,target_lines,[target_span[0]+1,target_span[1]])
 except (ValueError,UnicodeDecodeError) as e:raise Hold('full_document_identity_derivation_failed:'+str(e)) from e
 full_before=derived['before_text'];full_lines=SELECTION.logical_lines(full_before);anchor=derived['target_start_line'];target_count=len(target_lines)
 function_start=target_span[0]+1-target_count;function_end=target_span[1]-target_count
 if function_start<=anchor or function_end<function_start or function_end>=len(full_lines):raise Hold('target_function_preedit_span_invalid')
 selection=SELECTION.select_source_window(lines=full_lines,region_start_line=anchor,region_end_line=anchor,
  scope=SELECTION.SelectionScope(anchor,function_end),max_utf16_units=SELECTION.DEFAULT_SELECTION_UTF16_UNITS,document_sha256=derived['before_sha256'])
 if tuple(selection.region)!=('',):raise Hold('production_selection_anchor_not_empty')
 rr=PROTOCOL.ReplacementRange(uri=base.replacement_range.uri,document_version=base.replacement_range.document_version,
  content_sha256=derived['before_sha256'],start=PROTOCOL.Position(anchor,0),end=PROTOCOL.Position(anchor,0))
 selected=dataclasses.replace(base,prefix=tuple(selection.prefix),suffix_lines=tuple(selection.suffix),region_old=(),cursor=PROTOCOL.Cursor(-1,None,None),replacement_range=rr,document_eol=derived['document_eol'],selected_references=())
 if _apply_global(full_before,selected,target_lines)!=raw.decode('utf-8'):raise Hold('global_application_does_not_reconstruct_source')
 return selected,full_before,derived,selection

def materialize_row(semantic:dict,provenance:dict,packet:dict,tokenizer:TokenizerAdapter,source_cache:dict|None=None)->tuple[dict,dict]:
 if semantic.get('status')!='semantic_supported_context_closure_root_review_required' or semantic.get('reasons')!=[]:raise Hold('semantic_status_not_supported')
 identity=validate_identity(semantic,provenance,packet)
 source_path=Path(semantic['source_path']);cache_key=(str(source_path),semantic['source_sha256'])
 if source_cache is not None and cache_key in source_cache:
  raw,identity_stat=source_cache[cache_key];now=source_path.stat();current=(now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)
  if current!=identity_stat:raise Hold('source_changed_after_cached_read')
 else:
  before_stat=source_path.stat();raw=source_path.read_bytes();after_stat=source_path.stat();identity_stat=(after_stat.st_dev,after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns)
  if (before_stat.st_dev,before_stat.st_ino,before_stat.st_size,before_stat.st_mtime_ns)!=identity_stat or len(raw)!=before_stat.st_size:raise Hold('source_stat_unstable')
  if source_cache is not None:source_cache[cache_key]=(raw,identity_stat)
 if sha_bytes(raw)!=semantic['source_sha256']:raise Hold('source_pin_failed')
 if semantic.get('target_occurrence_method')=='exact_bytes':parse_bytes=raw
 elif semantic.get('target_occurrence_method')=='uniform_crlf_to_lf' and b'\r\n' in raw and b'\r' not in raw.replace(b'\r\n',b''):parse_bytes=raw.replace(b'\r\n',b'\n')
 else:raise Hold('unsupported_or_unproven_eol_normalization')
 try:source_lf=parse_bytes.decode('utf-8')
 except UnicodeDecodeError as e:raise Hold('source_not_utf8') from e
 target_lines=list(packet.get('result',{}).get('target_body',[]))
 if not target_lines or any(not isinstance(x,str) or '\n' in x or '\r' in x for x in target_lines):raise Hold('target_body_invalid')
 target=('\n'.join(target_lines)+'\n').encode()
 if semantic.get('parse_bytes_sha256')!=sha_bytes(parse_bytes) or semantic.get('scope',{}).get('parsed_source_sha256')!=sha_bytes(parse_bytes):raise Hold('semantic_parse_byte_identity_failed')
 if not SEMANTIC.namespace_ok(semantic.get('namespace',{})):raise Hold('semantic_namespace_evidence_invalid')
 if sha_bytes(target)!=semantic.get('target_sha256') or len(target)!=semantic.get('target_bytes'):raise Hold('target_hash_or_size_join_failed')
 if parse_bytes.count(target)!=1:raise Hold('target_not_exactly_once')
 target_start=parse_bytes.index(target);target_start_line=parse_bytes[:target_start].count(b'\n');target_end_line=target_start_line+len(target_lines);lines=source_lf.split('\n')
 item={k:semantic[k] for k in ('target_occurrences','target_definition_name','documented_params','imported_symbols','prior_reviewed_recovery')}
 reasons,resolution,closure=SEMANTIC.decide(item,semantic.get('scope'),source_lf)
 if reasons or resolution.get('unresolved') or closure!=semantic.get('context_closure'):raise Hold('semantic_v6_recheck_failed:'+','.join(reasons))
 target_span=checked_span(closure['target_definition_span'],len(lines),'target_definition')
 helpers=list(closure['required_helper_spans'])
 if helpers:raise Hold('helper_evidence_production_wiring_unavailable:'+','.join(sorted(str(x.get('name')) for x in helpers)))
 base=PROTOCOL.PromptContext.from_mapping(packet['result']['context'])
 if packet['result'].get('operation')!='replace' or base.region_old or not base.replacement_range.is_empty:raise Hold('roxygen_insertion_geometry_required')
 if base.selected_references:raise Hold('candidate_typed_evidence_production_wiring_unavailable')
 candidate_audit=_validate_candidate_window(base,packet,target_lines,source_lf,target_start_line)
 selected,full_before,derived,selection=_production_context(base,raw,target_lines,target_span)
 target_fragment='\n'.join(lines[target_span[0]:target_span[1]]);view_lines=[*selected.prefix,'',*selected.suffix_lines];view_wire='\n'.join(view_lines);view_document=GEOMETRY.eol_text(selected.document_eol).join(view_lines)
 if view_wire.count(target_fragment)!=1:raise Hold('target_function_not_once_in_production_source_view')
 row=PROTOCOL.build_training_row(selected,operation='replace',region_new=target_lines,tokenizer=tokenizer,row_id=identity['row_id'],family='roxygen_drafting',package_id=identity['package_id'],split='train')
 strict_errors=STRICT.validate_token_row(row,full_text=True)
 if strict_errors:raise Hold('strict_protocol_validation_failed:'+','.join(strict_errors))
 target_body='\n'.join(target_lines)
 if target_body in row['prompt_text'].replace('\r\n','\n'):raise Hold('target_roxygen_leaked_into_prompt')
 parsed=PROTOCOL.parse_output(row['target_text'],selected);plan=PROTOCOL.make_application_plan(parsed,selected,active_uri=selected.replacement_range.uri,active_document_version=selected.replacement_range.document_version,active_content_sha256=sha_bytes(full_before.encode()))
 if plan.replacement_text!=target_body.replace('\n',GEOMETRY.eol_text(selected.document_eol)):raise Hold('application_plan_replacement_mismatch')
 if _apply_global(full_before,selected,target_lines)!=raw.decode('utf-8'):raise Hold('application_plan_source_reconstruction_failed')
 selection_data=selection.to_dict();full_view=not selection.omissions
 profile={'schema':'sepalith.dat10.sourcewalk_semantic_materialization_profile.v3','row_id':identity['row_id'],'status':'materialized_review_only_root_admission_required','identity':identity,'semantic':{'target_sha256':semantic['target_sha256'],'target_definition_name':semantic['target_definition_name'],'target_definition_span':semantic['scope']['target_definition_span'],'required_helper_spans':[],'reference_resolution':resolution,'target_roxygen_source_interval':[target_start_line+1,target_end_line],'target_roxygen_excluded_from_prompt':True,'complete_target_preserved':True},'candidate_audit':candidate_audit,'geometry':{'mode':'production_global_identity_bounded_contiguous_view','operation':'replace','application_scope':'full_active_document','full_preedit_sha256':sha_bytes(full_before.encode()),'full_preedit_bytes':len(full_before.encode()),'global_replacement_range':selected.replacement_range.to_dict(),'selected_view_document_sha256':sha_bytes(view_document.encode()),'selected_view_wire_sha256':sha_bytes(view_wire.encode()),'selected_view_document_bytes':len(view_document.encode()),'selected_view_is_full_document':full_view,'selection_policy_id':selection.policy_id,'selection':selection_data,'application_exact':True,'prompt_sha256':sha_bytes(row['prompt_text'].encode()),'target_text_sha256':sha_bytes(row['target_text'].encode()),'target_body_sha256':sha_bytes(row['target_body_text'].encode()),'prompt_tokens':row['prompt_token_count'],'target_body_tokens':row['target_body_token_count'],'target_tokens':row['target_token_count'],'sequence_tokens':len(row['input_ids']),'sequence_bucket':length_bucket(len(row['input_ids'])),'target_gt_1024':row['target_body_token_count']>1024},'protocol':{'renderer_id':row['renderer_id'],'tokenizer_json_sha256':row['tokenizer_json_sha256'],'bos_token_id':row['bos_token_id'],'eos_token_id':row['eos_token_id'],'terminal_tokens':row['target_terminal_tokens'],'strict_validator_errors':[]},'training_admission':False}
 return row,profile

def atomic_json(path:Path,value:Any):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with tmp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
class AtomicJsonl:
 def __init__(self,path:Path):self.path=path;self.tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');self.stream=self.tmp.open('x');self.rows=0
 def write(self,row:dict):self.stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');self.rows+=1
 def close(self):self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.tmp.replace(self.path)
def fsync_dir(path:Path):fd=os.open(path,os.O_RDONLY);os.fsync(fd);os.close(fd)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--semantic-manifest',type=Path,required=True);ap.add_argument('--expected-semantic-manifest-sha256',required=True);ap.add_argument('--semantic-ledger',type=Path,required=True);ap.add_argument('--expected-semantic-sha256',required=True);ap.add_argument('--provenance-ledger',type=Path,required=True);ap.add_argument('--expected-provenance-sha256',required=True);ap.add_argument('--candidate-packets',type=Path,required=True);ap.add_argument('--expected-packets-sha256',required=True);ap.add_argument('--accepted-ids',type=Path,required=True);ap.add_argument('--expected-accepted-ids-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh_output_required')
 for path,expected in ((a.semantic_manifest,a.expected_semantic_manifest_sha256),(a.semantic_ledger,a.expected_semantic_sha256),(a.provenance_ledger,a.expected_provenance_sha256),(a.candidate_packets,a.expected_packets_sha256),(a.accepted_ids,a.expected_accepted_ids_sha256)):
  if sha(path)!=expected:raise ValueError('input_hash_mismatch:'+str(path))
 semantic_manifest=json.loads(a.semantic_manifest.read_text());code=semantic_manifest.get('code',{});output=semantic_manifest.get('output',{})
 if semantic_manifest.get('status')!='complete_review_only' or semantic_manifest.get('exact_id_closure') is not True or code!={'analyzer_sha256':SEMANTIC_SHA,'scope_helper_sha256':SEMANTIC_SCOPE_SHA,'namespace_helper_sha256':SEMANTIC_NAMESPACE_SHA} or output.get('sha256')!=a.expected_semantic_sha256:raise ValueError('semantic_v6_manifest_binding_invalid')
 accepted_raw=json.loads(a.accepted_ids.read_text());accepted_list=accepted_raw['ids'] if isinstance(accepted_raw,dict) else accepted_raw
 if not isinstance(accepted_list,list) or any(not isinstance(x,str) or not x for x in accepted_list) or len(set(accepted_list))!=len(accepted_list):raise ValueError('accepted_ids_invalid_or_duplicate')
 accepted=set(accepted_list)
 semantic,semantic_sha,_=exact_rows(a.semantic_ledger);provenance,provenance_sha,_=exact_rows(a.provenance_ledger);packets,packets_sha,_=exact_rows(a.candidate_packets)
 if not accepted or not accepted<=set(semantic) or not accepted<=set(provenance) or not accepted<=set(packets):raise ValueError('accepted_id_join_incomplete')
 a.output.mkdir(parents=True);started=time.monotonic();tokenizer=TokenizerAdapter();row_out=AtomicJsonl(a.output/'training-rows.jsonl');profile_out=AtomicJsonl(a.output/'profiles.jsonl');hold_out=AtomicJsonl(a.output/'holds.jsonl');buckets=collections.Counter();target_long=0;source_cache={}
 for rid in sorted(accepted):
  try:
   row,profile=materialize_row(semantic[rid],provenance[rid],packets[rid],tokenizer,source_cache);row_out.write(row);profile_out.write(profile);buckets[profile['geometry']['sequence_bucket']]+=1;target_long+=profile['geometry']['target_gt_1024']
  except Exception as e:hold_out.write({'schema':'sepalith.dat10.sourcewalk_semantic_materialization_hold.v3','row_id':rid,'reason':type(e).__name__+':'+str(e),'target_truncation':'none','silent_drop':False})
 for stream in (row_out,profile_out,hold_out):stream.close()
 outputs={name:{'path':str(a.output/name),'bytes':(a.output/name).stat().st_size,'sha256':sha(a.output/name),'rows':count} for name,count in [('training-rows.jsonl',row_out.rows),('profiles.jsonl',profile_out.rows),('holds.jsonl',hold_out.rows)]}
 manifest={'schema':'sepalith.dat10.sourcewalk_semantic_materialization.v3','status':'complete_review_only_root_admission_required' if not hold_out.rows and row_out.rows==len(accepted) else 'complete_with_explicit_holds_root_review_required','accepted_scope_rows':len(accepted),'materialized_rows':row_out.rows,'hold_rows':hold_out.rows,'exact_denominator_closure':row_out.rows+hold_out.rows==len(accepted),'length_buckets':dict(buckets),'target_gt_1024':target_long,'inputs':{'semantic_manifest':{'path':str(a.semantic_manifest),'sha256':sha(a.semantic_manifest)},'semantic_ledger':{'path':str(a.semantic_ledger),'sha256':semantic_sha},'provenance_ledger':{'path':str(a.provenance_ledger),'sha256':provenance_sha},'candidate_packets':{'path':str(a.candidate_packets),'sha256':packets_sha},'accepted_ids':{'path':str(a.accepted_ids),'sha256':sha(a.accepted_ids),'rows':len(accepted)},'semantic_v6_analyzer_sha256':SEMANTIC_SHA,'campaign_protocol_sha256':PROTOCOL_SHA,'strict_protocol_validator_sha256':STRICT_SHA,'semantic_context_geometry_v2_sha256':GEOMETRY_SHA,'campaign_selection_sha256':SELECTION_SHA,'tokenizer':{'path':str(TOKENIZER_PATH),'sha256':TOKENIZER_SHA}},'outputs':outputs,'policy':{'target_truncation':'none','context_length_exclusion':'none','long_rows':'materialize and report length bucket','target_roxygen_in_prompt':False,'required_spans':'target function pinned in production-equivalent contiguous source selection; helper-dependent rows held until production evidence provider wiring','training_admission':False},'elapsed_seconds':time.monotonic()-started,'training_admission':False}
 atomic_json(a.output/'manifest.json',manifest);fsync_dir(a.output);print(json.dumps({'materialized':row_out.rows,'holds':hold_out.rows,'length_buckets':dict(buckets)},sort_keys=True))
if __name__=='__main__':main()
