#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, importlib.util, json, os, shutil, sys, tempfile, time
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb');E=Path('/mnt/e/sepalith/campaign-20260915/data-work')
BASE=PLAN/'docs/campaign/work/lead/r2-semantic763-context-admission-v1/source/base_materializer.py';PROTOCOL=PLAN/'docs/campaign/work/lead/r2-full-weight-edit-sft-preparation-v1/source/packages/sepalith/src/sepalith/campaign_protocol.py';STRICT=PLAN/'docs/campaign/work/lead/r2-roxy8597-root-fixes-v1/audit_authoritative_stream.py'
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json')
MAIN=E/'Semantic4551-provider-materialization-v1/final-01';REC=E/'Semantic4551-hold-recovery-v1/final-01';OUT=E/'Semantic4435-token-review-v1/review-01'
PINS={'base':'2edfe6c4761271889e2b2f008fcccd00b8bf2adaf712c18cc2d588eb9840136f','protocol':'5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156','strict':'7f042596a67ec9983f6f409a8913f4bc8df3361548a1e91fb77c9f914dafcf37','tokenizer':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','main_manifest':'819d61049ceec8f5513d6ae8fd4a9cfa5e8b6d201fb76d7de81a91e55b14f61e','main_rows':'4c3d7898c803d565a001c14f56bab24a1ad24019d1bfb4f0e79e644a516133cc','main_provenance':'138f7de823e23442ca79a3e9016718e2d9c791d7b875e094f16eafdbbedf9acd','recovered_manifest':'59729bae81cc45c67f57bf8b953a0c7832b05d54e652636e29152d99c15215b8','recovered_rows':'6f2c2d76e0acfb70b1dc374da4e48c965303cc0f5e1cbdd3aed707806b754a06','recovered_provenance':'a6929e93055659bd0f5d37b14e19c3e5ba5e6fd972689ca5a1850c856584a6cf'}
def digest_path(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def loadmod(name,path):spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load_keyed(path:Path,key='row_id'):
 h=hashlib.sha256();rows={}
 with path.open('rb') as f:
  for raw in f:
   h.update(raw)
   if not raw.strip():continue
   x=json.loads(raw);rid=x[key]
   if rid in rows:raise ValueError('duplicate:'+rid)
   rows[rid]=x
 return rows,h.hexdigest()
def quantiles(values):
 values=sorted(values)
 def q(p):return values[min(len(values)-1,int((len(values)-1)*p))]
 return {'min':values[0],'p50':q(.5),'p90':q(.9),'p95':q(.95),'p99':q(.99),'max':values[-1]}
def main():
 started=time.monotonic()
 for name,path in [('base',BASE),('protocol',PROTOCOL),('strict',STRICT),('tokenizer',TOKENIZER),('main_manifest',MAIN/'manifest.json'),('recovered_manifest',REC/'manifest.json')]:
  if digest_path(path)!=PINS[name]:raise ValueError('pin:'+name)
 B=loadmod('token_review_base',BASE);P=loadmod('token_review_protocol',PROTOCOL);S=loadmod('token_review_strict',STRICT);tokenizer=B.TokenizerAdapter(TOKENIZER)
 if tokenizer.backend.encode_special_tokens is not True:raise ValueError('split_special_tokens_true_not_active')
 main_prov,prov_sha=load_keyed(MAIN/'candidate-provenance.jsonl');rec_prov,rec_prov_sha=load_keyed(REC/'candidate-provenance.jsonl')
 if prov_sha!=PINS['main_provenance'] or rec_prov_sha!=PINS['recovered_provenance']:raise ValueError('provenance pin')
 if OUT.exists():raise ValueError('fresh output required')
 OUT.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.review-01.',dir=OUT.parent));ledger_path=tmp/'row-review.jsonl';counts=collections.Counter();failures=[];prompt_counts=[];target_counts=[];sequence_counts=[];margins=[];rows_sha=hashlib.sha256();seen=set()
 terminal_ids=tokenizer.encode(P.TERMINAL,add_special_tokens=False,split_special_tokens=True)
 def validate(row,prov,reserve,denominator):
  rid=row['id'];errors=[]
  def check(condition,reason):
   if not condition:errors.append(reason)
  prompt_ids=tokenizer.encode(row['prompt_text'],add_special_tokens=False,split_special_tokens=True);target_ids=tokenizer.encode(row['target_text'],add_special_tokens=False,split_special_tokens=True);joint=tokenizer.encode(row['prompt_text']+row['target_text'],add_special_tokens=False,split_special_tokens=True)
  check(joint==prompt_ids+target_ids,'joint_boundary_retokenized');check(row['input_ids']==[0,*joint,1],'input_ids_retokenization_mismatch');check(row['target_start']==1+len(prompt_ids),'target_start_mismatch');check(row['prompt_token_count']==len(prompt_ids),'prompt_count_mismatch');check(row['target_token_count']==len(target_ids),'target_count_mismatch');check(len(target_ids)>=len(terminal_ids) and target_ids[-len(terminal_ids):]==terminal_ids,'terminal_suffix_mismatch');check(row['target_terminal_tokens']==terminal_ids and row['target_terminal_token_count']==len(terminal_ids),'stored_terminal_mismatch');check(row['target_body_tokens']==target_ids[:-len(terminal_ids)] and row['target_body_token_count']==len(target_ids)-len(terminal_ids),'body_segment_mismatch');check(row['input_ids'][row['target_start']:row['target_start']+len(target_ids)]==target_ids,'target_slice_mismatch');check(row['input_ids'][0]==0 and row['input_ids'].count(0)==1,'bos_exactly_once');check(row['input_ids'][-1]==1 and row['input_ids'].count(1)==1,'eos_exactly_once');check(all(type(x) is int and 0<=x<P.VOCAB_SIZE for x in row['input_ids']),'token_range');check(not any(P.is_native_control_token(x) for x in row['input_ids'][1:-1]),'internal_native_control');check(not S.validate_token_row(row,full_text=True),'strict_validator');
  try:P.validate_training_row(row)
  except Exception as error:errors.append('protocol_validator:'+type(error).__name__)
  context=int(prov['context_size']);actual=len(row['input_ids']);selection_ceiling=1+len(prompt_ids)+reserve;reserved_with_eos=selection_ceiling+1
  check(context in (16384,32768),'context_profile');check(len(target_ids)<=reserve,'target_exceeds_reserve');check(actual<=context,'actual_sequence_exceeds_context');check(selection_ceiling<=context,'selection_reserve_exceeds_context')
  result={'row_id':rid,'denominator':denominator,'status':'PASS' if not errors else 'FAIL','context_size':context,'generation_reserve':reserve,'prompt_tokens':len(prompt_ids),'target_body_tokens':len(target_ids)-len(terminal_ids),'terminal_tokens':len(terminal_ids),'target_tokens':len(target_ids),'protocol_eos_tokens':1,'sequence_tokens':actual,'selection_reserved_tokens_without_protocol_eos':selection_ceiling,'selection_reserved_tokens_with_protocol_eos':reserved_with_eos,'reserved_plus_eos_fits':reserved_with_eos<=context,'actual_sequence_fits':actual<=context,'target_includes_terminal_excludes_eos':True,'errors':errors}
  return result
 try:
  with (MAIN/'candidate-tokenrows.jsonl').open('rb') as stream,ledger_path.open('x') as ledger:
   for raw in stream:
    rows_sha.update(raw)
    if not raw.strip():continue
    row=json.loads(raw);rid=row['id'];
    if rid in seen or rid not in main_prov:raise ValueError('main ID closure:'+rid)
    seen.add(rid);result=validate(row,main_prov[rid],1024,'frozen4435');ledger.write(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');counts[result['status']]+=1
    if result['errors']:failures.append({'row_id':rid,'errors':result['errors']})
    prompt_counts.append(result['prompt_tokens']);target_counts.append(result['target_tokens']);sequence_counts.append(result['sequence_tokens']);margins.append(result['context_size']-result['sequence_tokens']);counts['context_'+str(result['context_size'])]+=1;counts['reserved_plus_eos_fits_'+str(result['reserved_plus_eos_fits']).lower()]+=1
   ledger.flush();os.fsync(ledger.fileno())
  if rows_sha.hexdigest()!=PINS['main_rows'] or len(seen)!=4435 or seen!=set(main_prov):raise ValueError('main row closure')
  rec_row=json.loads((REC/'candidate-tokenrows.jsonl').read_text());rec_result=validate(rec_row,rec_prov[rec_row['id']],2048,'recovered1675');(tmp/'recovered-review.json').write_text(json.dumps(rec_result,indent=2,sort_keys=True)+'\n')
  if digest_path(REC/'candidate-tokenrows.jsonl')!=PINS['recovered_rows']:raise ValueError('recovered row pin')
  if rec_result['errors']:failures.append({'row_id':rec_result['row_id'],'errors':rec_result['errors']})
  output_sha=digest_path(ledger_path);manifest={'schema':'sepalith.dat10.semantic4435.token_review.v1','status':'PASS' if not failures else 'FAIL','denominators':{'frozen_candidates':4435,'recovered_candidate':1},'input_pins':PINS,'tokenizer_contract':{'adapter':'base_materializer.TokenizerAdapter','encode_special_tokens':True,'calls':{'add_special_tokens':False,'split_special_tokens':True},'bos_id':0,'eos_id':1,'terminal_ids':terminal_ids,'terminal_token_count':len(terminal_ids),'vocab_size':P.VOCAB_SIZE},'main':{'counts':dict(counts),'prompt_tokens':quantiles(prompt_counts),'target_tokens_including_terminal_excluding_eos':quantiles(target_counts),'sequence_tokens_including_bos_and_eos':quantiles(sequence_counts),'actual_context_margin_tokens':quantiles(margins),'failures':failures,'ledger':{'path':'row-review.jsonl','sha256':output_sha,'bytes':ledger_path.stat().st_size,'rows':4435}},'recovered':rec_result,'reserve_interpretation':{'target_token_count_includes_terminal':True,'target_token_count_excludes_protocol_eos':True,'selector_formula':'BOS(1) + prompt + generation_reserve <= context_size','training_full_sequence_formula':'BOS(1) + prompt + target(including terminal) + EOS(1)','strict_reserved_training_ceiling_formula':'BOS(1) + prompt + generation_reserve + EOS(1)','eos_consumes_additional_training_context_token':True},'context_selection_used_target_or_gold':False,'elapsed_seconds':time.monotonic()-started}
  with (tmp/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,OUT);fd=os.open(OUT.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(manifest,sort_keys=True))
 except Exception:shutil.rmtree(tmp,ignore_errors=True);raise
if __name__=='__main__':main()
