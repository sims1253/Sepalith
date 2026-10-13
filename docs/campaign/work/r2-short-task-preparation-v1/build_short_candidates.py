#!/usr/bin/env python3
"""Bounded, unadmitted task candidates from exact allowlisted TRAIN source only."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,sys,time
from collections import Counter
from pathlib import Path
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));sys.dont_write_bytecode=True
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['RAYON_NUM_THREADS']='2';os.environ['CUDA_VISIBLE_DEVICES']=''
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
EXEC=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912')
OUT=Path(__file__).resolve().parent
DOCUMENTS=PLAN/'work/r2-corpus-preparation-v1/profile-shard-v1/documents.jsonl'
DOCUMENTS_SHA='674d3bf6e2da08b53a0d0fa6d7ae1977c5bc6940938ebe2aaf6c7c1643ff6d68'
PARTITION=PLAN/'work/r2-corpus-preparation-v1/cpt-train-group-partition.json'
PARTITION_SHA='6553ab5d84429d07a9094df28ea029b94088b48bbd0fbf706451f443ccd66b06'
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
POLICY='r2_literal_short_tail_and_completed_pipe_v1'

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
sys.path.insert(0,str(EXEC/'packages/sepalith/src'))
from sepalith.campaign_protocol import PromptContext,Cursor,Position,ReplacementRange,build_training_row,render_prompt,utf16_length,parse_output
from sepalith.campaign_selection import select_source_window
rp=load('r2_short_pinned_parser',PLAN/'work/r2-development-diagnosis-v1/reward-candidate/pinned_r_parser.py')
parse_r=rp.create_r_parser()
finish=load('r2_short_existing_finish',EXEC/'experiments/synthetic-data/finish_block.py')
scenario=load('r2_short_existing_scenarios',EXEC/'experiments/synthetic-data/scenarios.py')
adapter=load('r2_short_structured',EXEC/'experiments/training/campaign_admission_structured.py')
sys.path.insert(0,'/home/m0hawk/Documents/Sepalith/.venv-sft/lib/python3.10/site-packages')
from tokenizers import Tokenizer
assert sha(TOKENIZER.read_bytes())==TOKENIZER_SHA
_backend=Tokenizer.from_file(str(TOKENIZER));_backend.encode_special_tokens=True
class Encoder:
 def encode(self,text,*,add_special_tokens=False,split_special_tokens=True):
  assert add_special_tokens is False and split_special_tokens is True
  return _backend.encode(text,add_special_tokens=False).ids
ENCODER=Encoder()

def build_context(document,path,start_line,end_line,old,history=(),version=0,require_full_prefix=False):
 lines=document.split('\n');digest=sha(document.encode())
 selected=select_source_window(lines=lines,region_start_line=start_line,region_end_line=end_line,max_utf16_units=6000,document_sha256=digest)
 if selected.overflow or (require_full_prefix and selected.omissions):raise ValueError('selection_drops_required_function_prefix')
 end=Position(end_line,utf16_length(old[-1]) if old else 0)
 rr=ReplacementRange('file:///r2-train/'+path,version,digest,Position(start_line,0),end)
 cursor=Cursor(len(old)-1,len(old[-1]),utf16_length(old[-1])) if old else Cursor(-1,None,None)
 ctx=PromptContext(path=path,prefix=selected.prefix,selected_references=(),history=tuple(history),diagnostics=(),retrieval=(),scope_mode='off',scope_lines=(),suffix_lines=selected.suffix,region_old=tuple(old),cursor=cursor,replacement_range=rr,document_eol='lf')
 return ctx,selected.to_dict()

def packet(meta,context,selection,operation,new,family,derivation,before,after,extra):
 key={'policy':POLICY,'source':meta['sha256'],'group_id':meta['group_id'],'derivation':derivation,'before':sha(before.encode()),'range':context.replacement_range.to_dict(),'target':list(new)}
 rid='r2-short-'+sha(canonical(key).encode())[:24]
 row=build_training_row(context,operation=operation,region_new=new,tokenizer=ENCODER,row_id=rid,family=family,package_id=meta['package'],split='train')
 if row['target_token_count']+1>192:raise ValueError('target_over_192_including_EOS')
 if row['prompt_token_count']+1>3000 or len(row['input_ids'])>4096:raise ValueError('prompt_or_total_over_limit')
 assert row['input_ids'][0]==0 and row['input_ids'][-1]==1
 assert not any(i in (0,1) for i in row['input_ids'][1:-1])
 assert _backend.decode(row['input_ids'][1:-1],skip_special_tokens=False)==row['prompt_text']+row['target_text']
 pred=parse_output(row['target_text'],context);assert pred.status=='accepted'
 assert parse_r(after)
 provenance={'source_path':meta['path'],'source_sha256':meta['sha256'],'source_sha1':meta['sha1'],'source_git_blob_sha1':meta['git_blob_sha1'],'source_document_id':meta['document_id'],'source_profile_manifest_sha256':DOCUMENTS_SHA,'CPT_partition_sha256':PARTITION_SHA,'group_id':meta['group_id'],'package_id':meta['package'],'version':meta['version'],'source_split':'train_group','cpt_partition':'cpt_train','source_role':'literal_source_slice' if family=='finish_block' else 'canonical_pipe_rewrite_simulation','source_is_observed_user_edit':False,'derivation':derivation,'policy':POLICY,'selection_source':{'document_text':before,'content_sha256':sha(before.encode()),'document_version':context.replacement_range.document_version,'availability':'full_snapshot','lineage':'source_derived_simulated_editor_buffer'},'gold_applied_document_text':after,'gold_applied_document_sha256':sha(after.encode()),'before_R_parse':parse_r(before),'after_R_parse':True,'selection':selection,**extra}
 return {'schema':'r2.short_task_candidate.v1','candidate_status':'source_and_token_checks_passed_not_training_admitted','admitted_for_training':False,'row':row,'context':context.to_dict(),'source_provenance':provenance}

def tail_candidates(meta,src):
 tree=finish.parser.parse(src)
 for name,fn,roxy in finish.collect_functions(src,tree):
  body=next((x for x in fn.children if x.type=='braced_expression'),None)
  if body is None:continue
  stmts=[x for x in body.named_children if x.type!='comment']
  if len(stmts)<2:continue
  last=stmts[-1]
  called=scenario.callee_name(src,last) if last.type=='call' else None
  if last.type!='identifier' and called!='return':continue
  cut=src.rfind(b'\n',0,last.start_byte)+1
  if cut<=stmts[-2].end_byte or cut<=body.start_byte:continue
  parent=fn.parent
  if parent is None or parent.type!='binary_operator':continue
  start=parent.start_byte;end=body.end_byte
  target=src[cut:end].decode('utf-8');before=src[start:cut].decode('utf-8');after=src[start:end].decode('utf-8')
  if not before.endswith('\n') or before+target!=after:continue
  visible={scenario.node_text(src,n).decode('utf-8') for n in scenario.traverse(fn) if n.type=='identifier' and n.end_byte<=cut}
  used={scenario.node_text(src,n).decode('utf-8') for n in scenario.traverse(last) if n.type=='identifier'}-{'return'}
  if not used or used-visible:continue
  rel='R/'+Path(meta['path']).name
  ctx,selected=build_context(before,rel,len(before.split('\n'))-1,len(before.split('\n'))-1,[],require_full_prefix=True)
  yield packet(meta,ctx,selected,'replace',target.split('\n'),'finish_block','literal_visible_tail_return',before,after,{'function_name':name,'function_source_byte_start':start,'function_source_byte_end':end,'target_source_byte_start':cut,'target_source_byte_end':end,'literal_splice_verified':True,'closing_brace_in_target':target.rstrip().endswith('}'),'visible_statement_count':len(stmts)-1,'tail_statement_count':1,'tail_identifier_count':len(used),'unseen_tail_identifiers':sorted(used-visible),'inferability':'visible-identifier tail-return proxy; not proof of unique intended implementation','pair_id':None})

def eligible_pipe_sites(bundle):
 sites=[]
 for node in scenario.traverse(bundle.tree.root_node):
  kids=node.children
  if node.type=='binary_operator' and len(kids)>=3 and kids[1].type=='special' and scenario.node_text(bundle.src,kids[1])==b'%>%' and scenario._pipe_ok(bundle,kids[1],kids[0],kids[2]):
   sites.append((kids[1].start_byte,*bundle.rowcol(kids[1].start_byte)))
 return sorted(sites)

def pipe_pair(meta,src):
 rel='R/'+Path(meta['path']).name;b=scenario.Bundle(meta['package'],rel,src);sites=eligible_pipe_sites(b)
 if len(sites)!=2 or sites[0][1]==sites[1][1]:return []
 raw=scenario.extract_pipe(b,random.Random(3407),cap=1)
 if len(raw)!=1:return []
 ex=raw[0];scenario.validate_example(ex)
 _,patch=adapter._parse_event_patch(ex,rel)
 lines=src.decode('utf-8').split('\n');after_event=adapter._apply_event(lines,patch)
 target_line=sites[1][1];old=ex['region_old'];new=ex['region_new'];assert after_event[target_line:target_line+len(old)]==old
 post=after_event[:target_line]+new+after_event[target_line+len(old):]
 original='\n'.join(lines);current='\n'.join(after_event);complete='\n'.join(post)
 if not all(parse_r(x) for x in [original,current,complete]):return []
 if eligible_pipe_sites(scenario.Bundle(meta['package'],rel,complete.encode())):return []
 uri='file:///r2-train/'+rel
 hist,history_meta=adapter._event_history(ex['event_diff'],patch,{},before_hash=sha(original.encode()),after_hash=sha(current.encode()),event_uri=uri,before_version=0)
 pairid='r2-pipe-pair-'+sha((meta['sha256']+str(target_line)).encode())[:20]
 ctx,selection=build_context(current,rel,target_line,target_line,old,hist,version=1)
 first=packet(meta,ctx,selection,'replace',new,'pipe_rewrite','canonical_pipe_pending_last_site',current,complete,{'pair_id':pairid,'original_source_R_parse':True,'original_source_sha256':meta['sha256'],'history_replay':history_meta,'canonical_eligible_sites_before':2,'canonical_eligible_sites_after':0})
 event=scenario.event_diff_for(rel,target_line+1,old[0],new[0]);_,patch2=adapter._parse_event_patch({'event_diff':event},rel)
 replay=adapter._apply_event(after_event,patch2);assert replay==post
 hist2,meta2=adapter._event_history(event,patch2,{},before_hash=sha(current.encode()),after_hash=sha(complete.encode()),event_uri=uri,before_version=1)
 ctx2,selection2=build_context(complete,rel,target_line,target_line,new,hist+hist2,version=2)
 second=packet(meta,ctx2,selection2,'no_op',new,'no_op','canonical_pipe_post_completion_noop',complete,complete,{'pair_id':pairid,'parent_candidate_id':first['row']['id'],'history_replay':[history_meta,meta2],'remaining_eligible_sites':0,'detector':'full-file count of canonical scenarios._pipe_ok eligible %>% AST nodes; not merely extract_pipe()==[]','detector_scope':'pipe_rewrite only','full_source_original_hash_verified':True})
 return [first,second]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-finish',type=int,default=512);ap.add_argument('--max-pipe-pairs',type=int,default=32);args=ap.parse_args()
 assert 1<=args.max_finish<=1024 and 0<=args.max_pipe_pairs<=64
 started=time.monotonic();assert sha(DOCUMENTS.read_bytes())==DOCUMENTS_SHA;assert sha(PARTITION.read_bytes())==PARTITION_SHA
 partition=json.loads(PARTITION.read_text());assert sum(x=='cpt_validation' for x in partition['groups'].values())==556
 all_docs=[json.loads(s) for s in DOCUMENTS.read_text().splitlines()]
 # Exclude validation BEFORE reading any source path.
 docs=[r for r in all_docs if r['cpt_partition']=='cpt_train' and partition['groups'].get(r['group_id'])=='cpt_train' and r['split']=='train_group'];assert len(docs)==1055
 docs.sort(key=lambda x:sha((POLICY+'\0'+x['sha256']).encode()))
 counts=Counter();rejected=Counter();rows=[];source_observations=[];package_finish=Counter();seen=set();pair_count=0;failures=[]
 for meta in docs:
  path=Path(meta['path']);before_stat=path.stat();src=path.read_bytes();after_stat=path.stat()
  assert (before_stat.st_size,before_stat.st_mtime_ns,before_stat.st_ino)==(after_stat.st_size,after_stat.st_mtime_ns,after_stat.st_ino)
  assert sha(src)==meta['sha256'] and len(src)==meta['source_utf8_bytes'];counts['source_files_verified']+=1
  source_observations.append({'path':str(path),'sha256':meta['sha256'],'bytes':len(src),'group_id':meta['group_id'],'package':meta['package']})
  if len(src)>400000:rejected['canonical_max_file_bytes']+=1;continue
  try:text=src.decode('utf-8')
  except UnicodeDecodeError:rejected['non_utf8']+=1;continue
  if '\r' in text:rejected['CR_or_CRLF_initial_slice_not_supported']+=1;continue
  if not parse_r(text):rejected['original_R_parse_error']+=1;continue
  counts['original_R_parse_valid']+=1
  if counts['finish_rows']<args.max_finish and package_finish[meta['package']]<3:
   try:
    for candidate in tail_candidates(meta,src):
     if candidate['row']['id'] in seen:continue
     if counts['finish_rows']>=args.max_finish or package_finish[meta['package']]>=3:break
     seen.add(candidate['row']['id']);rows.append(candidate);counts['finish_rows']+=1;package_finish[meta['package']]+=1
   except (ValueError,AssertionError) as error:rejected[str(error)[:120] or type(error).__name__]+=1
  if pair_count<args.max_pipe_pairs:
   try:
    pair=pipe_pair(meta,src)
    if pair:rows.extend(pair);pair_count+=1;counts['pipe_edits']+=1;counts['post_completion_noops']+=1
   except (ValueError,AssertionError) as error:rejected['pipe:'+str(error)[:120] or type(error).__name__]+=1
 with (OUT/'candidate-packets.jsonl').open('x') as f:
  for r in rows:f.write(canonical(r)+'\n')
 (OUT/'source-observations.json').write_text(json.dumps(source_observations,indent=2)+'\n')
 summary={'status':'candidate_source_token_parse_checks_complete_not_admitted','policy':POLICY,'source_documents_manifest':{'path':str(DOCUMENTS),'sha256':DOCUMENTS_SHA},'partition':{'path':str(PARTITION),'sha256':PARTITION_SHA,'excluded_validation_groups':556,'profile_validation_documents_not_read':len(all_docs)-len(docs)},'counts':counts,'rejections':rejected,'rows':len(rows),'families':dict(Counter(r['row']['family'] for r in rows)),'groups':len({r['source_provenance']['group_id'] for r in rows}),'packages':len({r['row']['package_id'] for r in rows}),'max_target_tokens_including_EOS':max((r['row']['target_token_count']+1 for r in rows),default=0),'max_prompt_tokens_including_BOS':max((r['row']['prompt_token_count']+1 for r in rows),default=0),'all_targets_include_EOS_within192':all(r['row']['target_token_count']+1<=192 for r in rows),'CPT_validation_group_overlap':len({r['source_provenance']['group_id'] for r in rows}&{k for k,v in partition['groups'].items() if v=='cpt_validation'}),'training_admitted':False,'model_framework_imported':any(x in sys.modules for x in ['torch','transformers']),'seconds':time.monotonic()-started,'limitations':['Initial source pool is frozen profile cpt_train1055 documents, not all broader raw corpus.','Tail-return lexical visibility is a grounding proxy, not proof of unique intended semantics.','Paired no-ops cover only canonical pipe rewrite with exactly two eligible source sites on separate lines.','Before finish buffer is an intentionally truncated source-derived function; full gold buffer is an exact source function slice.','No R evaluation, target invention, DEV/final data or original source writes.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
