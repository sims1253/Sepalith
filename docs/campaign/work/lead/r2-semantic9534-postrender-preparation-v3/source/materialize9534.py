#!/usr/bin/env python3
import argparse,collections,hashlib,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');E=Path('/mnt/e/sepalith/campaign-20260915/data-work');BASE=PLAN/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py';INTEGRATE=PLAN/'docs/campaign/work/lead/r2-semantic763-dedup-integration-v1/integrate.py';PREV=PLAN/'docs/campaign/work/lead/r2-semantic4551-provider-materialization-v1/materialize4551.py';INPUTS=E/'Semantic9535-provider-preparation-v1/render-inputs-v1'
INPUT_MANIFEST_SHA256='db1dd374d1d029695c9b8daef59151272d0db0a912276f8073b24b0df4f8e9a0';CURRENT_MANIFEST_SHA256='c44dadde27b2ace8908a505d1d3f691ec0e7bcb464c8411ee875d036b0b8fc60';SEMANTIC10948_MANIFEST_SHA256='c86e3bc3a402f4f568295a0dcd15fca6dfe6046ce6e7fb60844e25d505eb370f';SEMANTIC10948_TOKENROWS_SHA256='17d02115590ea61921b8ee9cee3b9c0de044cea065330fd9a35a266b3db03069';SEMANTIC10948_PROVENANCE_SHA256='b8d54654bc35d3166ef6b96cfe7cd4f78bcc0bd03b42d6eeaf5dcebe4f92a329';PROVIDER_ROWS=9534;GEOMETRY_HOLDS=1;SUPPORTED_DENOMINATOR=9535;CURRENT_ROWS=20191;SEMANTIC10948_ROWS=10682;COMBINED_ROWS=30873
PINS={'source_terminal':'0f2e2d321dce5ba4355a1eb7b304e97a2bfc093da27e46106dda3d41d0a63f78','rows4435':'4c3d7898c803d565a001c14f56bab24a1ad24019d1bfb4f0e79e644a516133cc','prov4435':'138f7de823e23442ca79a3e9016718e2d9c791d7b875e094f16eafdbbedf9acd','row_recovered':'6f2c2d76e0acfb70b1dc374da4e48c965303cc0f5e1cbdd3aed707806b754a06','prov_recovered':'a6929e93055659bd0f5d37b14e19c3e5ba5e6fd972689ca5a1850c856584a6cf','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'}
SOURCE_PINS={BASE:'2edfe6c4761271889e2b2f008fcccd00b8bf2adaf712c18cc2d588eb9840136f',INTEGRATE:'5f15ea1aa3f83890b812ddb9a564936c51bb4607f855fd1db5130a724209cce6',PREV:'af4a095daff1441d074c8fe74f3a6c58a1ff2defe9d923444375c2d29622bb7e'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def validate_provider_input_manifest(path):
 m=json.loads(Path(path).read_text());assert m['schema']=='sepalith.dat10.semantic9535.provider_inputs.v1' and m['status']=='complete_target_free_review_only' and m['training_admission'] is False and m['target_or_gold_copied_to_provider_inputs'] is False and m['exact_supported_denominator_closure'] is True
 assert sha(path)==INPUT_MANIFEST_SHA256
 assert m['provider_rows']==PROVIDER_ROWS and m['geometry_preparation_holds']==GEOMETRY_HOLDS
 assert m['upstream']['semantic_supported']==SUPPORTED_DENOMINATOR
 assert sum(int(x['provider_rows']) for x in m['shards'])==PROVIDER_ROWS
 assert [int(x['shard']) for x in m['shards']]==list(range(27,41))
 return m
if any(sha(path)!=expected for path,expected in SOURCE_PINS.items()):raise RuntimeError('external_source_pin_mismatch')
def loadmod(n,p):spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
B=loadmod('s10948_base',BASE);I=loadmod('s10948_integrate',INTEGRATE);M=loadmod('s10948_previous',PREV)
def keyed(paths,key='row_id'):
 out={}
 for p in paths:
  for l in Path(p).open():
   x=json.loads(l);rid=x[key]
   if rid in out:raise ValueError('duplicate:'+rid)
   out[rid]=x
 return out
def augment(path,prov,eh,ph,existing,by_pair,by_prompt,by_source_target,by_path_target,tok):M.augment(path,prov,eh,ph,existing,by_pair,by_prompt,by_source_target,by_path_target,tok)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selected',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--dedup-binding',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();started=time.monotonic();assert sha(a.tokenizer)==PINS['tokenizer']
 input_manifest_path=INPUTS/'manifest.json';input_manifest=validate_provider_input_manifest(input_manifest_path)
 side_paths=[];input_paths=[]
 for bound in input_manifest['shards']:
  shard=int(bound['shard']);side=INPUTS/bound['sidecar']['path'];inp=INPUTS/bound['provider']['path'];assert sha(side)==bound['sidecar']['sha256'] and sha(inp)==bound['provider']['sha256'];side_paths.append(side);input_paths.append(inp)
 selected=keyed([a.selected]);side=keyed(side_paths);inputs=keyed(input_paths);denominator=input_manifest['provider_rows'];assert len(selected)==len(side)==len(inputs)==denominator and set(selected)==set(side)==set(inputs)
 tok=B.TokenizerAdapter(a.tokenizer);existing,by_pair,by_prompt,by_source_target,by_path_target=I.load_existing(E/'DAT10-finish-source-repair-v3/train-token-rows.jsonl',E/'RL11-15006-context-v1/context-sidecar.jsonl')
 augment(E/'Semantic763-dedup-integration-v1/final-02/candidate-tokenrows.jsonl',E/'Semantic763-dedup-integration-v1/final-02/candidate-provenance.jsonl',M.PINS['rows616'],M.PINS['prov616'],existing,by_pair,by_prompt,by_source_target,by_path_target,tok);augment(E/'Semantic-provider-integration-v1/final-01/candidate-tokenrows.jsonl',E/'Semantic-provider-integration-v1/final-01/candidate-provenance.jsonl',M.PINS['rows133'],M.PINS['prov133'],existing,by_pair,by_prompt,by_source_target,by_path_target,tok);augment(E/'Semantic4551-provider-materialization-v1/final-01/candidate-tokenrows.jsonl',E/'Semantic4551-provider-materialization-v1/final-01/candidate-provenance.jsonl',PINS['rows4435'],PINS['prov4435'],existing,by_pair,by_prompt,by_source_target,by_path_target,tok);augment(E/'Semantic4551-hold-recovery-v1/final-01/candidate-tokenrows.jsonl',E/'Semantic4551-hold-recovery-v1/final-01/candidate-provenance.jsonl',PINS['row_recovered'],PINS['prov_recovered'],existing,by_pair,by_prompt,by_source_target,by_path_target,tok);assert len(existing)==20191
 binding=json.loads(a.dedup_binding.read_text());assert binding['schema']=='sepalith.dat10.semantic9534.dedup_binding.v1' and binding['status']=='root_bound_after_semantic10948_finalized' and binding['current_rows']==CURRENT_ROWS and binding['semantic10948_candidate_rows']==SEMANTIC10948_ROWS and binding['combined_rows']==COMBINED_ROWS and binding.get('future_noop_rows',0)==0
 current_manifest_path=Path(binding['accepted_current_manifest']['path']);assert binding['accepted_current_manifest']['sha256']==CURRENT_MANIFEST_SHA256 and binding['accepted_current_manifest']['rows']==CURRENT_ROWS and sha(current_manifest_path)==CURRENT_MANIFEST_SHA256
 pending=binding['pending_semantic10948'];assert all('noop' not in str(pending.get(k,{}).get('path','')).lower() for k in ('manifest','tokenrows','provenance'));assert pending['manifest']['sha256']==SEMANTIC10948_MANIFEST_SHA256 and pending['tokenrows']['sha256']==SEMANTIC10948_TOKENROWS_SHA256 and pending['provenance']['sha256']==SEMANTIC10948_PROVENANCE_SHA256;manifest_path=Path(pending['manifest']['path']);assert sha(manifest_path)==pending['manifest']['sha256'];pending_manifest=json.loads(manifest_path.read_text());assert pending_manifest['status'].startswith('complete_review_only') and pending_manifest['training_admission'] is False and pending_manifest['candidate_rows']==SEMANTIC10948_ROWS and pending_manifest['existing_review_union']==CURRENT_ROWS
 rows_path=Path(pending['tokenrows']['path']);prov_path=Path(pending['provenance']['path']);assert pending['tokenrows']['rows']==SEMANTIC10948_ROWS and pending['provenance']['rows']==SEMANTIC10948_ROWS;augment(rows_path,prov_path,pending['tokenrows']['sha256'],pending['provenance']['sha256'],existing,by_pair,by_prompt,by_source_target,by_path_target,tok);assert len(existing)==binding['combined_rows']
 if a.output.exists():
  raise ValueError('fresh output')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 kept=[];prov=[];ledger=[];larger=[];counts=collections.Counter()
 try:
  for rid in sorted(selected):
   r=selected[rid];b=inputs[rid];s=side[rid]
   if r['status']!='supported':counts['provider_or_policy_hold']+=1;ledger.append({'row_id':rid,'status':'hold','reason':r.get('reason','provider_or_policy_hold'),'details':r.get('reasons') or r.get('detail'),'silent_drop':False});continue
   ctx=B.PROTOCOL.PromptContext.from_mapping(r['selected_context']);prompt=B.PROTOCOL.render_prompt(ctx);assert I.digest_text(prompt)==r['prompt_sha256'] and len(tok.encode(prompt))==r['prompt_tokens']
   row=B.PROTOCOL.build_training_row(ctx,operation='replace',region_new=s['target_lines'],tokenizer=tok,row_id=rid,family='roxygen_drafting',package_id=s['identity']['package_id'],split='train');errors=B.STRICT.validate_token_row(row,full_text=True);assert not errors,(rid,errors);applied=B._apply_global(b['preedit_text'],ctx,s['target_lines']);assert applied is not None and I.digest_text(applied)==s['postedit_source_sha256'];assert row['target_body_text'] not in row['prompt_text'] and row['target_body_text'] not in b['preedit_text']
   if row['target_token_count']>2048:counts['target_exceeds_fixed_reserve']+=1;larger.append({'row_id':rid,'target_token_count':row['target_token_count'],'target_truncated':False,'status':'larger_fixed_reserve_profile_pending','next_action':'rerender entire eligible pool with a larger fixed reserve selected without target access'});ledger.append({'row_id':rid,'status':'hold','reason':'target_exceeds_fixed_reserve','target_token_count':row['target_token_count'],'target_truncated':False,'silent_drop':False});continue
   if len(row['input_ids'])>r['context_size']:counts['actual_sequence_exceeds_selected_context']+=1;ledger.append({'row_id':rid,'status':'hold','reason':'actual_sequence_exceeds_selected_context','sequence_tokens':len(row['input_ids']),'context_size':r['context_size'],'target_truncated':False,'silent_drop':False});continue
   reason=None;matches=[]
   if rid in existing:reason='duplicate_existing_id';matches=[rid]
   elif I.pair_key(row) in by_pair:reason='duplicate_existing_prompt_target';matches=sorted(by_pair[I.pair_key(row)])
   else:
    conflicts=sorted(old for target,old in by_prompt.get(I.digest_text(row['prompt_text']),set()) if target!=I.target_key(row))
    if conflicts:reason='contradiction_existing_prompt_different_target';matches=conflicts
   identity=s['identity'];hs=I.source_hashes(identity)|{identity['source_sha256']};paths=I.normalized_source_paths(identity)
   if reason is None:
    found=set()
    for h in hs:found|=by_source_target.get((h,I.target_key(row)),set())
    if found:reason='duplicate_existing_source_target';matches=sorted(found)
   if reason is None:
    found=set()
    for q in paths:found|=by_path_target.get((row['package_id'],q,I.target_key(row)),set())
    if found:reason='duplicate_existing_source_path_target';matches=sorted(found)
   if reason:counts[reason]+=1;ledger.append({'row_id':rid,'status':'excluded','reason':reason,'matches':matches,'silent_drop':False});continue
   counts['new_candidate']+=1;kept.append(row);prov.append({'row_id':rid,'source_identity':identity,'mode':r['mode'],'context_size':r['context_size'],'generation_reserve':2048,'policy_resolution':r['policy_resolution'],'prompt_sha256':r['prompt_sha256'],'preedit_sha256':b['preedit_sha256'],'postedit_source_sha256':s['postedit_source_sha256'],'target_truncated':False,'selection_target_or_gold_used':False});ledger.append({'row_id':rid,'status':'candidate','reason':'new_candidate','silent_drop':False})
  assert len(ledger)==denominator and len({x['row_id'] for x in ledger})==denominator
  def write(name,values):
   p=tmp/name
   with p.open('x') as f:
    for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
    f.flush();os.fsync(f.fileno())
   return {'path':name,'rows':len(values),'sha256':sha(p),'bytes':p.stat().st_size}
  outputs={name:write(name,vals) for name,vals in [('candidate-tokenrows.jsonl',kept),('candidate-provenance.jsonl',prov),('decision-ledger.jsonl',ledger),('larger-fixed-reserve-pending.jsonl',larger)]};manifest={'schema':'sepalith.dat10.semantic9534.provider_materialization.v1','status':'complete_review_only_root_admission_required','provider_rows':PROVIDER_ROWS,'geometry_preparation_holds':GEOMETRY_HOLDS,'accounted_source_review_rows':SUPPORTED_DENOMINATOR,'denominator':denominator,'existing_review_union':binding['combined_rows'],'candidate_rows':len(kept),'status_counts':dict(counts),'outputs':outputs,'larger_fixed_reserve_policy':{'status':'queued_not_excluded','selection_must_remain_target_free':True,'targets_are_never_truncated':True},'checks':{'exact_id_accounting':True,'target_free_selection':True,'targets_joined_after_selection':True,'strict_protocol':True,'full_source_reapplication':True,'dedup_against_current20191_and_finalized10948':True,'no_target_truncation':True,'future_noop_rows_excluded_from_binding':True},'training_admission':False,'elapsed_seconds':time.monotonic()-started};B.atomic_json(tmp/'manifest.json',manifest);os.rename(tmp,a.output);fd=os.open(a.output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
