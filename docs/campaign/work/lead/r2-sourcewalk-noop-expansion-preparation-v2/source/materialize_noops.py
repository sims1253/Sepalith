#!/usr/bin/env python3
"""Build exact no-op rows from reviewed provider contexts and deduplicate the 20,191-row review union."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,sys,tempfile,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import campaign_protocol as protocol;import audit_authoritative_stream as strict
TOKENIZER_SHA='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81';TARGET='[NO_EDIT]\n>>>>>>> UPDATED'
def req(v,m):
 if not v:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def read_rows(paths,key):
 out={}
 for path in paths:
  for n,line in enumerate(Path(path).open(),1):
   x=json.loads(line);rid=x.get(key);req(isinstance(rid,str)and rid and rid not in out,f'row identity:{path}:{n}');out[rid]=x
 return out
class Tok:
 def __init__(self,path):
  req(sha(path)==TOKENIZER_SHA,'tokenizer pin');from tokenizers import Tokenizer;self.backend=Tokenizer.from_file(str(path));self.backend.encode_special_tokens=True
 def encode(self,text,*,add_special_tokens=False,split_special_tokens=True):req(add_special_tokens is False and split_special_tokens is True,'token policy');return self.backend.encode(text,add_special_tokens=False).ids
def prompt_source_projection(prompt):
 lines=prompt.splitlines();req(lines and lines[0]==protocol.TASK_CONTRACT and lines[1].startswith('<filename>'),'prompt frame')
 def at(value,start=0):
  try:return lines.index(value,start)
  except ValueError:raise ValueError('prompt frame')
 refs=at('<filename>selected_references',2);suffix=at('<[fim-suffix]>',refs);current=at('<<<<<<< CURRENT',suffix);equal=at('=======',current);middle=at('<[fim-middle]>',equal);req(middle==len(lines)-1,'prompt trailing frame');prefix=lines[2:refs];suffix_lines=lines[suffix+1:current];region=[x.replace(protocol.CURSOR_MARKER,'')for x in lines[current+1:equal]];return digest(json.dumps([prefix,region,suffix_lines],ensure_ascii=False,separators=(',',':')))
def validate_row(row,tok):
 req(row['family']=='no_op'and row['split']=='train'and row['target_operation']=='no_op'and row['target_body_text']==protocol.NO_EDIT and row['target_text']==TARGET,'no-op target contract');req(not strict.validate_token_row(row,full_text=True),'strict token row');req(tok.encode(row['prompt_text'])==row['input_ids'][1:row['target_start']]and tok.encode(row['target_text'])==row['input_ids'][row['target_start']:-1],'retokenization')
def main():
 p=argparse.ArgumentParser();p.add_argument('--selected',type=Path,required=True);p.add_argument('--preparation-root',type=Path,required=True);p.add_argument('--existing-manifest',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prep=json.loads((a.preparation_root/'manifest.json').read_text());req(prep['status']in('partial_review_only','complete_review_only')and prep['training_admission']is False,'preparation status');side_paths=[a.preparation_root/f"shard-{x['shard']:04d}.sidecar.jsonl"for x in prep['shards']];pred_paths=[a.preparation_root/f"shard-{x['shard']:04d}.jsonl"for x in prep['shards']];side=read_rows(side_paths,'row_id');pred=read_rows(pred_paths,'row_id');selected=read_rows([a.selected],'row_id');req(set(side)==set(pred)==set(selected),'provider join');tok=Tok(a.tokenizer);existing_manifest=json.loads(a.existing_manifest.read_text());req(existing_manifest['rows']==20191 and len(existing_manifest['cohorts'])==5,'existing union manifest');existing_ids=set();pairs=collections.defaultdict(set);prompts=collections.defaultdict(set);source_targets=collections.defaultdict(set)
 for cohort in existing_manifest['cohorts']:
  path=Path(cohort['path']);req(sha(path)==cohort['sha256'],'existing pin');n=0
  for line in path.open():
   row=json.loads(line);rid=row['id'];req(rid not in existing_ids,'existing duplicate id');existing_ids.add(rid);n+=1;t=digest(row['target_text']);pairs[digest(row['prompt_text']+'\0'+row['target_text'])].add(rid);prompts[digest(row['prompt_text'])].add((t,rid));source_targets[(prompt_source_projection(row['prompt_text']),t)].add(rid)
  req(n==cohort['rows'],'existing row count')
 req(len(existing_ids)==20191,'existing denominator');kept=[];prov=[];ledger=[];seen_pairs={};seen_sources={}
 for rid in sorted(selected):
  s=selected[rid];b=pred[rid];meta=side[rid]
  if s.get('status')!='supported':ledger.append({'row_id':rid,'status':'hold','reason':s.get('reason','provider_hold'),'silent_drop':False});continue
  ctx=protocol.PromptContext.from_mapping(s['selected_context']);req(ctx.replacement_range.content_sha256==b['preedit_sha256'],'preedit binding');prompt=protocol.render_prompt(ctx);req(digest(prompt)==s['prompt_sha256']and len(tok.encode(prompt))==s['prompt_tokens'],'provider prompt parity');row=protocol.build_training_row(ctx,operation='no_op',region_new=[],tokenizer=tok,row_id=rid,family='no_op',package_id=meta['identity']['package_id'],split='train');validate_row(row,tok);req(meta['full_source_reapplication_sha256']==b['preedit_sha256']and b['selection_target_or_gold_used']is False,'full source/target-free binding');pair=digest(row['prompt_text']+'\0'+row['target_text']);prompt_sha=digest(row['prompt_text']);target_sha=digest(row['target_text']);source_key=(prompt_source_projection(row['prompt_text']),target_sha);reason=None;matches=[]
  if rid in existing_ids:reason='duplicate_existing_id';matches=[rid]
  elif pair in pairs:reason='duplicate_existing_prompt_target';matches=sorted(pairs[pair])
  else:
   conflicts=sorted(old for target,old in prompts.get(prompt_sha,set())if target!=target_sha)
   if conflicts:reason='contradiction_existing_prompt';matches=conflicts
  if reason is None and source_key in source_targets:reason='duplicate_existing_source_projection_target';matches=sorted(source_targets[source_key])
  if reason is None and pair in seen_pairs:reason='duplicate_candidate_prompt_target';matches=[seen_pairs[pair]]
  if reason is None and source_key in seen_sources:reason='duplicate_candidate_source_projection_target';matches=[seen_sources[source_key]]
  if reason:ledger.append({'row_id':rid,'status':'excluded','reason':reason,'matches':matches,'silent_drop':False});continue
  seen_pairs[pair]=rid;seen_sources[source_key]=rid;kept.append(row);prov.append({'row_id':rid,'source_identity':meta['identity'],'prompt_sha256':prompt_sha,'target_sha256':target_sha,'source_projection_sha256':source_key[0],'preedit_sha256':b['preedit_sha256'],'target_free_selection':True,'full_source_reapplication':True});ledger.append({'row_id':rid,'status':'candidate','reason':'novel_after_20191_union','silent_drop':False})
 req(len(ledger)==len(selected),'candidate accounting');parent=a.output.parent;parent.mkdir(parents=True,exist_ok=True);req(not a.output.exists(),'fresh output');tmp=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=parent))
 try:
  outputs={}
  for name,values in [('candidate-tokenrows.jsonl',kept),('candidate-provenance.jsonl',prov),('decision-ledger.jsonl',ledger)]:
   q=tmp/name
   with q.open('x')as f:
    for x in values:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
    f.flush();os.fsync(f.fileno())
   outputs[name]={'rows':len(values),'bytes':q.stat().st_size,'sha256':sha(q)}
  manifest={'schema':'sepalith.dat10.sourcewalk-noop-materialization.v1','status':'partial_review_only'if not prep['full_41_shard_closure']else'complete_review_only_root_admission_required','input_rows':len(selected),'candidate_rows':len(kept),'decision_rows':len(ledger),'existing_review_union':20191,'dedup_dimensions':['row_id','prompt+target','prompt contradiction','source_projection+target'],'target_only_digest_is_recorded_not_excluded':'all legitimate no-op rows share the same target','outputs':outputs,'training_admission':False,'full_41_shard_closure':prep['full_41_shard_closure']};(tmp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(tmp,a.output);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
