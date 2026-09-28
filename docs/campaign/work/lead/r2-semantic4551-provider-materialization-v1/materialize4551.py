#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');BASE=ROOT/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py';INTEGRATE=ROOT/'docs/campaign/work/lead/r2-semantic763-dedup-integration-v1/integrate.py'
PINS={'sidecar':'e31cc3c910de6fb426a6bda8c35382cc4dd50514c4ea5fd196513943cd0d1da6','rows15006':'3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e','context15006':'36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a','rows616':'9e114d8fd064c47d74b435882fdfe2d2345411ab9e063348ee48b5dba46dbcec','prov616':'53d9c5c53768f229a4cb0f28db776d31eaf9d96663038375eeb8d528042364ff','rows133':'2fbfbcb96849289dd65105cee708cb6f07a7f1c5b02c2d9fd57fb8402082f822','prov133':'9a974db879b8d42c99e6b09b3caebb51f9bad650096858927537527b74b22b5c','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def req(v,m):
 if not v:raise ValueError(m)
def loadmod(n,p):spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
B=loadmod('s4551_base',BASE);I=loadmod('s4551_integrate',INTEGRATE)
def rows(paths,key='row_id'):
 out={}
 for path in paths:
  for l in Path(path).read_text().splitlines():
   x=json.loads(l);rid=x[key];req(rid not in out,'duplicate:'+rid);out[rid]=x
 return out
def augment(path,prov_path,expected,prov_expected,existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer):
 req(sha(path)==expected and sha(prov_path)==prov_expected,'prior candidate pin');rs=rows([path],key='id');ps=rows([prov_path]);req(set(rs)==set(ps),'prior candidate closure')
 for rid,row in rs.items():
  I.validate_tokenrow(row,tokenizer.backend);identity=ps[rid]['source_identity'];rec={'id':rid,'pair':I.pair_key(row),'prompt':I.digest_text(row['prompt_text']),'target':I.target_key(row),'package_id':row['package_id'],'source_hashes':I.source_hashes(identity)|{identity['source_sha256']},'source_paths':I.normalized_source_paths(identity)};req(rid not in existing,'prior overlap');existing[rid]=rec;by_pair[rec['pair']].add(rid);by_prompt[rec['prompt']].add((rec['target'],rid))
  for h in rec['source_hashes']:by_source_target[(h,rec['target'])].add(rid)
  for q in rec['source_paths']:by_path_target[(rec['package_id'],q,rec['target'])].add(rid)
def main():
 ap=argparse.ArgumentParser();
 for n in ['selected','sidecar','input0','input1','rows15006','context15006','rows616','prov616','rows133','prov133','tokenizer','output']:ap.add_argument('--'+n,type=Path,required=True)
 a=ap.parse_args();started=time.monotonic()
 for n in PINS:req(sha(getattr(a,n))==PINS[n],'pin:'+n)
 selected=rows([a.selected]);side=rows([a.sidecar]);inputs=rows([a.input0,a.input1]);req(len(selected)==len(side)==len(inputs)==4551 and set(selected)==set(side)==set(inputs),'4551 closure')
 tokenizer=B.TokenizerAdapter(a.tokenizer);existing,by_pair,by_prompt,by_source_target,by_path_target=I.load_existing(a.rows15006,a.context15006)
 augment(a.rows616,a.prov616,PINS['rows616'],PINS['prov616'],existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer);augment(a.rows133,a.prov133,PINS['rows133'],PINS['prov133'],existing,by_pair,by_prompt,by_source_target,by_path_target,tokenizer);req(len(existing)==15755,'prior union denominator')
 parent=a.output.parent;parent.mkdir(parents=True,exist_ok=True);req(not a.output.exists(),'fresh output');tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=parent));kept=[];prov=[];ledger=[];counts=collections.Counter()
 try:
  for rid in sorted(selected):
   r=selected[rid];b=inputs[rid];s=side[rid]
   if r['status']!='supported':counts['provider_or_policy_hold']+=1;ledger.append({'row_id':rid,'status':'hold','reason':r.get('reason','provider_or_policy_hold'),'details':r.get('reasons') or r.get('detail'),'silent_drop':False});continue
   ctx=B.PROTOCOL.PromptContext.from_mapping(r['selected_context']);req(ctx.replacement_range.content_sha256==b['preedit_sha256'],'document hash:'+rid);prompt=B.PROTOCOL.render_prompt(ctx);req(I.digest_text(prompt)==r['prompt_sha256'] and len(tokenizer.encode(prompt))==r['prompt_tokens'],'prompt parity:'+rid)
   row=B.PROTOCOL.build_training_row(ctx,operation='replace',region_new=s['target_lines'],tokenizer=tokenizer,row_id=rid,family='roxygen_drafting',package_id=s['identity']['package_id'],split='train');errs=B.STRICT.validate_token_row(row,full_text=True);req(not errs,'strict:'+rid+':'+','.join(errs));applied=B._apply_global(b['preedit_text'],ctx,s['target_lines']);req(applied is not None and I.digest_text(applied)==s['postedit_source_sha256'],'reapply:'+rid);req(row['target_body_text'] not in row['prompt_text'] and row['target_body_text'] not in b['preedit_text'],'leak:'+rid)
   if row['target_token_count']>1024:counts['target_exceeds_generation_reserve']+=1;ledger.append({'row_id':rid,'status':'hold','reason':'target_exceeds_generation_reserve','target_token_count':row['target_token_count'],'target_truncated':False,'silent_drop':False});continue
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
   counts['new_candidate']+=1;kept.append(row);prov.append({'row_id':rid,'source_identity':identity,'mode':r['mode'],'context_size':r['context_size'],'policy_resolution':r['policy_resolution'],'prompt_sha256':r['prompt_sha256'],'preedit_sha256':b['preedit_sha256'],'postedit_source_sha256':s['postedit_source_sha256'],'namespace_path':r.get('namespace_path'),'namespace_sha256':r.get('namespace_sha256'),'external_import_dependencies':r.get('observed_dependencies',[]),'selected_reference_count':len(r['selected_context']['selected_references']),'target_truncated':False,'selection_target_or_gold_used':False});ledger.append({'row_id':rid,'status':'candidate','reason':'new_candidate','context_size':r['context_size'],'silent_drop':False})
  req(len(ledger)==4551 and len({x['row_id'] for x in ledger})==4551,'ledger closure');pairs=collections.Counter(I.pair_key(x) for x in kept);prompts=collections.defaultdict(set)
  for x in kept:prompts[I.digest_text(x['prompt_text'])].add(I.target_key(x))
  req(max(pairs.values(),default=0)<=1 and all(len(v)==1 for v in prompts.values()),'internal dedup')
  def write(name,vals):
   p=tmp/name
   with p.open('x') as f:
    for x in vals:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
    f.flush();os.fsync(f.fileno())
   return {'path':name,'rows':len(vals),'bytes':p.stat().st_size,'sha256':sha(p)}
  outputs={n:write(n,v) for n,v in [('candidate-tokenrows.jsonl',kept),('candidate-provenance.jsonl',prov),('decision-ledger.jsonl',ledger)]};manifest={'schema':'sepalith.dat10.semantic4551.provider_materialization.v1','status':'complete_review_only_root_admission_required','denominator':4551,'existing_review_union':15755,'candidate_rows':len(kept),'hold_or_excluded_rows':4551-len(kept),'status_counts':dict(counts),'candidate_ids_sha256':I.digest_text('\n'.join(x['id'] for x in kept)+('\n' if kept else '')),'context_counts':dict(collections.Counter(str(x['context_size']) for x in prov)),'mode_counts':dict(collections.Counter(x['mode'] for x in prov)),'outputs':outputs,'checks':{'exact_4551_accounting':True,'target_free_selection':True,'prompt_and_target_retokenized':True,'full_source_reapplication':True,'no_target_truncation':True,'strict_protocol':True,'dedup_against_15006_616_133':True,'internal_dedup':True},'inputs':PINS,'training_admission':False,'elapsed_seconds':time.monotonic()-started};B.atomic_json(tmp/'manifest.json',manifest);fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,a.output);fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
