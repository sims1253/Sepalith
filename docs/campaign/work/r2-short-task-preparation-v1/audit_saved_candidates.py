#!/usr/bin/env python3
"""Read-only replay of saved TRAIN candidates; no model or training launch."""
import hashlib,json,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_short_candidates as b
from sepalith.campaign_protocol import validate_training_row,utf16_to_codepoint_column

def main():
 started=time.monotonic();packets=[json.loads(x) for x in (b.OUT/'candidate-packets.jsonl').read_text().splitlines()]
 docs={x['document_id']:x for x in map(json.loads,b.DOCUMENTS.read_text().splitlines())}
 partition=json.loads(b.PARTITION.read_text())['groups'];seen=set();prompts={};cache={};pairs=defaultdict(list);counts=Counter();target_mass=Counter();ids=[]
 for p in packets:
  r=validate_training_row(p['row']);pr=p['source_provenance'];meta=docs[pr['source_document_id']]
  assert p['admitted_for_training'] is False and r['split']=='train'
  assert meta['cpt_partition']==partition[meta['group_id']]=='cpt_train' and meta['split']=='train_group'
  assert all(pr[a]==meta[z] for a,z in [('source_path','path'),('source_sha256','sha256'),('group_id','group_id'),('package_id','package')])
  if meta['path'] not in cache:
   path=Path(meta['path']);s0=path.stat();raw=path.read_bytes();s1=path.stat()
   assert (s0.st_ino,s0.st_mtime_ns,s0.st_size)==(s1.st_ino,s1.st_mtime_ns,s1.st_size)
   assert b.sha(raw)==meta['sha256'] and len(raw)==meta['source_utf8_bytes'];cache[meta['path']]=raw
  raw=cache[meta['path']];ctx=b.PromptContext.from_dict(p['context']);before=pr['selection_source']['document_text'];after=pr['gold_applied_document_text']
  assert b.sha(before.encode())==pr['selection_source']['content_sha256']==ctx.replacement_range.content_sha256
  assert b.sha(after.encode())==pr['gold_applied_document_sha256']
  assert b.parse_r(before)==pr['before_R_parse'] and b.parse_r(after) and pr['after_R_parse']
  assert b.render_prompt(ctx)==r['prompt_text'];pred=b.parse_output(r['target_text'],ctx);assert pred.status=='accepted'
  lines=before.split('\n');rr=ctx.replacement_range
  start=sum(len(x)+1 for x in lines[:rr.start.line])+utf16_to_codepoint_column(lines[rr.start.line],rr.start.character)
  end=sum(len(x)+1 for x in lines[:rr.end.line])+utf16_to_codepoint_column(lines[rr.end.line],rr.end.character)
  assert before[start:end]=='\n'.join(ctx.region_old)
  applied=before if pred.operation=='no_op' else before[:start]+pred.body_text+before[end:]
  assert applied==after
  new=ctx.region_old if pred.operation=='no_op' else pred.body
  replay=b.build_training_row(ctx,operation=pred.operation,region_new=new,tokenizer=b.ENCODER,row_id=r['id'],family=r['family'],package_id=r['package_id'],split='train')
  assert replay==r and r['target_token_count']+1<=192
  assert r['id'] not in seen;seen.add(r['id']);ids.append(r['id'])
  key=b.sha(r['prompt_text'].encode());assert key not in prompts or prompts[key]==r['target_text'];prompts[key]=r['target_text']
  if r['family']=='finish_block':
   assert raw[pr['function_source_byte_start']:pr['target_source_byte_start']].decode()==before
   assert raw[pr['target_source_byte_start']:pr['target_source_byte_end']].decode()==r['target_body_text']
   assert raw[pr['function_source_byte_start']:pr['function_source_byte_end']].decode()==after
   assert not ctx.history and not ctx.suffix_lines and not pr['unseen_tail_identifiers']
   counts['literal_source_splices']+=1
  else:pairs[pr['pair_id']].append(p)
  counts[r['family']]+=1;target_mass[r['family']]+=r['target_token_count']+1
 for pairid,pair in pairs.items():
  assert len(pair)==2;edit=next(p for p in pair if p['row']['family']=='pipe_rewrite');noop=next(p for p in pair if p['row']['family']=='no_op')
  pr=edit['source_provenance'];meta=docs[pr['source_document_id']];raw=cache[meta['path']]
  assert b.pipe_pair(meta,raw)==[edit,noop]
  assert noop['source_provenance']['parent_candidate_id']==edit['row']['id']
  assert not b.eligible_pipe_sites(b.scenario.Bundle(meta['package'],noop['context']['path'],noop['source_provenance']['selection_source']['document_text'].encode()))
  counts['source_derived_pair_replays']+=1
 source_paths=[b.OUT/'build_short_candidates.py',b.OUT/'audit_saved_candidates.py',b.OUT/'test_short_candidates.py',Path(sys.executable).resolve(),b.TOKENIZER,b.DOCUMENTS,b.PARTITION]
 for m in list(sys.modules.values()):
  origin=getattr(m,'__file__',None)
  if origin and (str(origin).startswith(str(b.EXEC)) or str(origin).startswith(str(b.PLAN/'work/r2-development-diagnosis-v1/reward-candidate')) or '/site-packages/tokenizers/' in str(origin) or '/site-packages/tree_sitter' in str(origin)):
   source_paths.append(Path(origin))
 pins=[]
 for path in sorted(set(source_paths)):
  h=hashlib.sha256()
  with path.open('rb') as f:
   while chunk:=f.read(4*1024*1024):h.update(chunk)
  pins.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
 result={'status':'PASS','rows':len(packets),'counts':dict(counts),'unique_row_ids':len(seen),'unique_prompts':len(prompts),'contradictory_prompt_targets':0,'source_files_independently_rehashed':len(cache),'CPT_validation_group_overlap':0,'target_tokens_including_EOS_by_family':dict(target_mass),'no_op_row_fraction':counts['no_op']/len(packets),'no_op_target_token_fraction':target_mass['no_op']/sum(target_mass.values()),'model_framework_imported':any(x in sys.modules for x in ['torch','transformers']),'seconds':time.monotonic()-started,'limits':['Candidate source/parse/protocol audit, not semantic equivalence or scientific admission.','Pinned imported first-party/parser/tokenizer files and interpreter; standard-library/OS trust is not a complete software-closure claim.']}
 (b.OUT/'saved-packet-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 (b.OUT/'source-pins.json').write_text(json.dumps({'schema':1,'files':pins,'trust_boundary':result['limits'][1]},indent=2)+'\n')
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
