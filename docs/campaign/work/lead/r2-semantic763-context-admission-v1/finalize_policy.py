#!/usr/bin/env python3
"""Apply target-free prompt policy, then join complete targets for review-only token rows."""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).parent
def load(name,path):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=load('semantic763_finalize_base',HERE/'source/base_materializer.py')
RESERVE=1024
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read_rows(path,key='row_id'):
 rows={}
 with path.open() as f:
  for line in f:
   if not line.strip():continue
   x=json.loads(line);rid=x[key]
   if rid in rows:raise ValueError('duplicate_row_id:'+rid)
   rows[rid]=x
 return rows
class Writer:
 def __init__(self,path):self.path=path;self.tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');self.f=self.tmp.open('x');self.n=0
 def write(self,x):self.f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');self.n+=1
 def close(self):self.f.flush();os.fsync(self.f.fileno());self.f.close();self.tmp.replace(self.path)
def decision(rendered,tokenizer,cap):
 ft=len(tokenizer.encode(rendered['full_prompt']));bt=None
 if 1+ft+RESERVE<=cap:return 'full_document',rendered['full_context'],rendered['full_prompt'],ft,[]
 if rendered['bounded_status']=='supported':
  bt=len(tokenizer.encode(rendered['bounded_prompt']))
  if 1+bt+RESERVE<=cap:return 'complete_span',rendered['bounded_context'],rendered['bounded_prompt'],bt,['full_document_context_budget']
  return 'unsupported',None,None,bt,['complete_span_context_budget']
 return 'unsupported',None,None,None,['complete_span_'+rendered['bounded_status'],*rendered['bounded_reasons']]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--rendered',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('fresh_output_required')
 a.output.mkdir(parents=True);started=time.monotonic();prep=json.loads((a.prepared/'preparation-manifest.json').read_text())
 pred=read_rows(a.prepared/'prediction-inputs.jsonl');side=read_rows(a.prepared/'training-sidecar.jsonl');render=read_rows(a.rendered)
 if set(pred)!=set(side) or set(pred)!=set(render) or len(pred)+prep['preparation_holds']!=prep['semantic_supported']:raise ValueError('prepared_or_rendered_id_closure_failed')
 tokenizer=B.TokenizerAdapter();profiles=[]
 for cap in (16384,32768):
  for rid in sorted(pred):
   r=render[rid]
   if r['preedit_sha256']!=pred[rid]['preedit_sha256']:raise ValueError('rendered_preedit_join_failed:'+rid)
   mode,context,prompt,prompt_tokens,reasons=decision(r,tokenizer,cap)
   evidence_hold=None
   if pred[rid]['external_import_dependencies']:evidence_hold='external_namespace_import_provider_unavailable'
   elif mode=='unsupported':evidence_hold='prediction_policy_unsupported'
   profiles.append({'row_id':rid,'context_size':cap,'raw_policy_mode':mode,'selected_context':context,'prompt_text':prompt,'prompt_tokens':prompt_tokens,'policy_reasons':reasons,'evidence_hold':evidence_hold,'external_import_dependencies':pred[rid]['external_import_dependencies'],'helper_count':len(pred[rid]['required_helper_names']),'preedit_sha256':pred[rid]['preedit_sha256']})
 # Detect cohort-local prompt contradictions and exact duplicates after evidence gates.
 for cap in (16384,32768):
  eligible=[x for x in profiles if x['context_size']==cap and x['evidence_hold'] is None]
  groups=collections.defaultdict(list)
  for x in eligible:groups[sha_bytes(x['prompt_text'].encode())].append(x)
  for _,items in groups.items():
   targets=collections.defaultdict(list)
   for x in items:targets[side[x['row_id']]['target_sha256']].append(x)
   if len(targets)>1:
    for x in items:x['evidence_hold']='prompt_target_contradiction'
   else:
    for x in sorted(items,key=lambda y:y['row_id'])[1:]:x['evidence_hold']='duplicate_prompt_target'
 outputs={};summary={}
 for cap in (16384,32768):
  tokenout=Writer(a.output/f'candidate-tokenrows-{cap}.jsonl');profileout=Writer(a.output/f'profiles-{cap}.jsonl');holdout=Writer(a.output/f'holds-{cap}.jsonl');counts=collections.Counter();helpers=collections.Counter();target_over=0
  for x in sorted((z for z in profiles if z['context_size']==cap),key=lambda z:z['row_id']):
   rid=x['row_id'];s=side[rid];counts['raw_'+x['raw_policy_mode']]+=1
   if x['helper_count']:helpers['raw_'+x['raw_policy_mode']]+=1
   if x['evidence_hold']:
    holdout.write({'schema':'sepalith.dat10.semantic763.context_hold.v1','row_id':rid,'context_size':cap,'reason':x['evidence_hold'],'policy_mode':x['raw_policy_mode'],'policy_reasons':x['policy_reasons'],'external_import_dependencies':x['external_import_dependencies'],'helper_count':x['helper_count'],'silent_drop':False});counts['hold_'+x['evidence_hold']]+=1;continue
   context=B.PROTOCOL.PromptContext.from_mapping(x['selected_context']);target_lines=s['target_lines']
   if context.replacement_range.content_sha256!=x['preedit_sha256'] or B.sha_bytes(x['prompt_text'].encode())!=B.sha_bytes(B.PROTOCOL.render_prompt(context).encode()):raise ValueError('context_identity_or_render_mismatch:'+rid)
   row=B.PROTOCOL.build_training_row(context,operation='replace',region_new=target_lines,tokenizer=tokenizer,row_id=rid,family='roxygen_drafting',package_id=s['identity']['package_id'],split='train')
   errors=B.STRICT.validate_token_row(row,full_text=True)
   if errors:raise ValueError('strict_protocol_failed:'+rid+':'+','.join(errors))
   if B._apply_global(pred[rid]['preedit_text'],context,target_lines) and sha_bytes(B._apply_global(pred[rid]['preedit_text'],context,target_lines).encode())!=s['postedit_source_sha256']:raise ValueError('postedit_reconstruction_failed:'+rid)
   if row['prompt_text']!=x['prompt_text']:raise ValueError('prompt_rebuild_mismatch:'+rid)
   if row['target_token_count']>RESERVE:
    target_over+=1;holdout.write({'schema':'sepalith.dat10.semantic763.context_hold.v1','row_id':rid,'context_size':cap,'reason':'complete_target_exceeds_generation_reserve','target_tokens':row['target_token_count'],'silent_drop':False});counts['hold_complete_target_exceeds_generation_reserve']+=1;continue
   if len(row['input_ids'])>cap:raise ValueError('selected_sequence_exceeds_context:'+rid)
   tokenout.write(row);profileout.write({'schema':'sepalith.dat10.semantic763.context_profile.v1','row_id':rid,'context_size':cap,'mode':x['raw_policy_mode'],'prompt_tokens':x['prompt_tokens'],'target_tokens':row['target_token_count'],'sequence_tokens':len(row['input_ids']),'prompt_sha256':sha_bytes(row['prompt_text'].encode()),'target_sha256':s['target_sha256'],'preedit_sha256':x['preedit_sha256'],'postedit_source_sha256':s['postedit_source_sha256'],'helper_count':x['helper_count'],'required_helper_spans':s['required_helper_spans'],'external_import_dependencies':[],'training_admission':False});counts['candidate']+=1
   if x['helper_count']:helpers['candidate']+=1
  for w in (tokenout,profileout,holdout):w.close()
  outputs[str(cap)]={name:{'path':str(path),'sha256':B.sha(path),'bytes':path.stat().st_size,'rows':n} for name,path,n in [('tokenrows',tokenout.path,tokenout.n),('profiles',profileout.path,profileout.n),('holds',holdout.path,holdout.n)]}
  summary[str(cap)]={'counts':dict(counts),'helper_counts':dict(helpers),'candidate_rows':tokenout.n,'hold_rows':holdout.n,'preparation_holds':prep['preparation_holds'],'exact_763_accounting':tokenout.n+holdout.n+prep['preparation_holds']==prep['semantic_supported'],'target_over_reserve':target_over}
 manifest={'schema':'sepalith.dat10.semantic763.context_admission.v1','status':'complete_review_only_root_admission_required','semantic_supported_denominator':prep['semantic_supported'],'selection_target_or_gold_used':False,'generation_reserve':RESERVE,'profiles':summary,'inputs':{'preparation_manifest':{'sha256':B.sha(a.prepared/'preparation-manifest.json')},'prediction_inputs':{'sha256':B.sha(a.prepared/'prediction-inputs.jsonl'),'rows':len(pred)},'training_sidecar':{'sha256':B.sha(a.prepared/'training-sidecar.jsonl'),'rows':len(side)},'rendered_prediction':{'sha256':B.sha(a.rendered),'rows':len(render)},'tokenizer':{'path':str(B.TOKENIZER_PATH),'sha256':B.TOKENIZER_SHA}},'outputs':outputs,'cross_corpus_dedup':'pending_root_union_review; cohort-local prompt/target duplicate and contradiction gates applied','training_admission':False,'elapsed_seconds':time.monotonic()-started}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
