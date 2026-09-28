#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path

PINS={'rendered':'dcc4c06e48df17c1f5369915d907e67ed3e9405009616fc582960e35dd31e73c','batch':'ca5f58db94fd586b7eb56079a06d4a4a14441d2f8b018024bddb56ebea9ae7c3','sidecar':'94e77b61f9c708626d0700cf6992ffdf364885fd1797171d5fde0767f89fa9a7','rows15006':'3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e','context15006':'36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a','rows616':'9e114d8fd064c47d74b435882fdfe2d2345411ab9e063348ee48b5dba46dbcec','prov616':'53d9c5c53768f229a4cb0f28db776d31eaf9d96663038375eeb8d528042364ff','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'}
ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
BASE=ROOT/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py'
INTEGRATE=ROOT/'docs/campaign/work/lead/r2-semantic763-dedup-integration-v1/integrate.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=load('provider_integration_base',BASE);I=load('provider_integration_dedup',INTEGRATE)
def rows(path,key='row_id'):
 out={}
 for line in Path(path).read_text().splitlines():
  x=json.loads(line);rid=x[key];require(rid not in out,'duplicate:'+rid);out[rid]=x
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--rendered',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--rows15006',type=Path,required=True);p.add_argument('--context15006',type=Path,required=True);p.add_argument('--rows616',type=Path,required=True);p.add_argument('--prov616',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
 for name,path in [('rendered',a.rendered),('batch',a.batch),('sidecar',a.sidecar),('rows15006',a.rows15006),('context15006',a.context15006),('rows616',a.rows616),('prov616',a.prov616),('tokenizer',a.tokenizer)]:require(sha(path)==PINS[name],'pin differs:'+name)
 rendered=rows(a.rendered);batch=rows(a.batch);side=rows(a.sidecar);require(len(rendered)==len(batch)==133 and set(rendered)==set(batch)<=set(side),'133 closure differs');require(all(x['status']=='supported' for x in rendered.values()),'rendered hold remains')
 tokenizer=B.TokenizerAdapter(a.tokenizer);existing,by_pair,by_prompt,by_source_target,by_path_target=I.load_existing(a.rows15006,a.context15006)
 old616=rows(a.rows616,key='id');prov616=rows(a.prov616);require(len(old616)==len(prov616)==616 and set(old616)==set(prov616),'616 closure differs')
 for rid,row in old616.items():
  I.validate_tokenrow(row,tokenizer.backend);ev=prov616[rid];identity=ev['source_identity'];record={'id':rid,'pair':I.pair_key(row),'prompt':I.digest_text(row['prompt_text']),'target':I.target_key(row),'package_id':row['package_id'],'source_hashes':I.source_hashes(identity)|{identity['source_sha256']},'source_paths':I.normalized_source_paths(identity)};require(rid not in existing,'616 ID overlaps 15006');existing[rid]=record;by_pair[record['pair']].add(rid);by_prompt[record['prompt']].add((record['target'],rid));
  for h in record['source_hashes']:by_source_target[(h,record['target'])].add(rid)
  for q in record['source_paths']:by_path_target[(record['package_id'],q,record['target'])].add(rid)
 require(len(existing)==15622,'existing union denominator differs')
 parent=a.output.parent;parent.mkdir(parents=True,exist_ok=True);require(not a.output.exists(),'fresh output required');tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=parent));selected=[];provenance=[];ledger=[];counts=collections.Counter()
 try:
  for rid in sorted(batch):
   r=rendered[rid];s=side[rid];b=batch[rid];ctx=B.PROTOCOL.PromptContext.from_mapping(r['selected_context']);require(ctx.replacement_range.content_sha256==b['preedit_sha256'],'context document hash differs')
   prompt=B.PROTOCOL.render_prompt(ctx);require(I.digest_text(prompt)==r['prompt_sha256'],'rendered prompt differs');require(len(tokenizer.encode(prompt))==r['prompt_tokens'],'prompt retokenization differs')
   row=B.PROTOCOL.build_training_row(ctx,operation='replace',region_new=s['target_lines'],tokenizer=tokenizer,row_id=rid,family='roxygen_drafting',package_id=s['identity']['package_id'],split='train')
   errors=B.STRICT.validate_token_row(row,full_text=True);require(not errors,'strict token row:'+rid+':'+','.join(errors));require(row['target_token_count']<=1024,'target exceeds reserve')
   applied=B._apply_global(b['preedit_text'],ctx,s['target_lines']);require(applied is not None and I.digest_text(applied)==s['postedit_source_sha256'],'postedit reapplication differs')
   require(row['target_body_text'] not in row['prompt_text'] and row['target_body_text'] not in b['preedit_text'],'target leak')
   reason=None;matches=[]
   if rid in existing:reason='duplicate_existing_id';matches=[rid]
   elif I.pair_key(row) in by_pair:reason='duplicate_existing_prompt_target';matches=sorted(by_pair[I.pair_key(row)])
   else:
    conflicts=sorted(old for target,old in by_prompt.get(I.digest_text(row['prompt_text']),set()) if target!=I.target_key(row))
    if conflicts:reason='contradiction_existing_prompt_different_target';matches=conflicts
   identity=s['identity'];hashes=I.source_hashes(identity)|{identity['source_sha256']};paths=I.normalized_source_paths(identity)
   if reason is None:
    found=set()
    for h in hashes:found|=by_source_target.get((h,I.target_key(row)),set())
    if found:reason='duplicate_existing_source_target';matches=sorted(found)
   if reason is None:
    found=set()
    for q in paths:found|=by_path_target.get((row['package_id'],q,I.target_key(row)),set())
    if found:reason='duplicate_existing_source_path_target';matches=sorted(found)
   if reason:
    counts[reason]+=1;ledger.append({'row_id':rid,'status':'excluded','reason':reason,'matches':matches,'silent_drop':False});continue
   counts['new_candidate']+=1;selected.append(row);provenance.append({'row_id':rid,'source_identity':identity,'mode':r['mode'],'context_size':32768,'prompt_sha256':r['prompt_sha256'],'preedit_sha256':b['preedit_sha256'],'postedit_source_sha256':s['postedit_source_sha256'],'namespace_path':r['namespace_path'],'namespace_sha256':r['namespace_sha256'],'external_import_dependencies':r['observed_dependencies'],'selected_reference_count':len(r['selected_context']['selected_references']),'target_truncated':False,'selection_target_or_gold_used':False});ledger.append({'row_id':rid,'status':'candidate','reason':'new_candidate_32k','silent_drop':False})
  require(len(ledger)==133 and len({x['row_id'] for x in ledger})==133,'ledger closure')
  pairs=collections.Counter(I.pair_key(x) for x in selected);prompts=collections.defaultdict(set)
  for x in selected:prompts[I.digest_text(x['prompt_text'])].add(I.target_key(x))
  require(max(pairs.values(),default=0)<=1 and all(len(v)==1 for v in prompts.values()),'internal duplicate or contradiction')
  def write(name,values):
   path=tmp/name
   with path.open('x') as f:
    for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
    f.flush();os.fsync(f.fileno())
   return {'path':name,'rows':len(values),'bytes':path.stat().st_size,'sha256':sha(path)}
  outputs={'candidate-tokenrows.jsonl':write('candidate-tokenrows.jsonl',selected),'candidate-provenance.jsonl':write('candidate-provenance.jsonl',provenance),'exclusion-ledger.jsonl':write('exclusion-ledger.jsonl',ledger)}
  manifest={'schema':'sepalith.dat10.semantic_provider_integration.v1','status':'complete_review_only_root_admission_required','input_candidates':133,'existing_union_rows':15622,'candidate_rows':len(selected),'excluded_rows':133-len(selected),'status_counts':dict(counts),'candidate_ids_sha256':I.digest_text('\n'.join(x['id'] for x in selected)+('\n' if selected else '')),'outputs':outputs,'inputs':PINS,'checks':{'target_free_selection':True,'exact_133_accounting':True,'document_and_namespace_hashes':True,'prompt_retokenized':True,'complete_target_retokenized':True,'full_source_reapplication':True,'strict_protocol':True,'cross_corpus_dedup_15006_plus_616':True,'internal_dedup':True},'training_admission':False,'elapsed_seconds':time.monotonic()-started}
  B.atomic_json(tmp/'manifest.json',manifest);fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,a.output);fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(manifest,sort_keys=True))
 except Exception:
  shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
