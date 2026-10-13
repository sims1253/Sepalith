#!/usr/bin/env python3
"""Materialize a fresh 15,006-row TRAIN view with 3,503 source braces restored."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source/packages/sepalith/src'))
from sepalith.campaign_protocol import PromptContext, build_training_row, validate_training_row

ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15006-v1/combined-token-rows.jsonl')
CONTEXTS=Path('/mnt/e/sepalith/campaign-20260915/data-work/RL11-15006-context-v1/context-sidecar.jsonl')
NEW_ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-token-rows.jsonl')
NEW_PROV=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/candidate-context-provenance.jsonl')
TOKENIZER=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')
AUDIT=PLAN/'docs/campaign/receipts/DAT-10-train-finish-boundary-audit.json'
EXTRACTOR=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/synthetic-data/finish_block.py')
ADAPTER=Path('/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/training/campaign_admission_completion.py')
PINS={
 ROWS:'65b2feb2e53970f02e7cfbe8947d628d584c204f741dad3a5a8b3254af5c1cd7',CONTEXTS:'36ee88c60e4d6d2ac95c425669f8d0c717ebfb79efda893eafb3262b131f6d1a',
 NEW_ROWS:'7887686e022e520e746c04c6197a9a3c2484fba2668d09a650d0627ff267187b',NEW_PROV:'e499c07d6da2ef325f7d9f3156b6c2e70361d3b1bc140198b6c3563b4471e7a0',
 AUDIT:'0b3921f7386f6610a471559d0ecc9a2841a00f0b7b610258e5c15b04903a0346',
 EXTRACTOR:'47759356b85eb9b48e0461e9265a9fa5a272df14de87747a63e62304be48a3fb',
 ADAPTER:'fafb852f31e5e59f003c3ccc0fb34919752020fb6c231a09d77b3ea158f173cc',
 TOKENIZER/'tokenizer.json':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81',
 TOKENIZER/'tokenizer_config.json':'e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b',
 HERE/'source/packages/sepalith/src/sepalith/campaign_protocol.py':'5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156',
 HERE/'parse_only.R':'524c23bd098420a8b90ad97a6d6e23f28dc9dde61c077cd01d51affc6b13ee51',
}
SOURCE_CASE_PINS={
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_compound.jsonl'):'a832e83153c92e41916159183dec3383d794bf993e5833c68faa5dd2a714c5cd',
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_compound_random.jsonl'):'d403796fe5320f8930ee827e5446455cfd290386e39c85c748a2f043ef621f5b',
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_authored_agy.jsonl'):'3f298f0362ab31f69e570105c224cabebe49f523be45e3f13b1baeda43cb0d44',
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_authored_zai.jsonl'):'afa095a752a7a40c5c937fd29239ae8fda13b60e4f53eef3efa3bcfbbe8462e6',
 Path('/mnt/h/sepalith/datasets/cases_v1/finish_block_authored_gpt56sol.jsonl'):'dd2ed2c724cae0d7363bb68a46196c75cbb0f69581b53f5e44635ecad1dcd636',
}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def load_map(path,key):
 out={}
 with path.open() as f:
  for line in f:
   x=json.loads(line);k=x[key]
   if k in out:raise ValueError(f'duplicate:{k}')
   out[k]=x
 return out
def line_sha(line):return hashlib.sha256(line.encode()).hexdigest()
def utf16_index(line,units):
 used=0
 for i,ch in enumerate(line):
  if used==units:return i
  used+=len(ch.encode('utf-16-le'))//2
  if used>units:raise ValueError('utf16 split')
 if used!=units:raise ValueError('utf16 overflow')
 return len(line)
def apply(document,rr,replacement):
 lines=document.split('\n')
 def offset(pos):return sum(len(x)+1 for x in lines[:pos['line']])+utf16_index(lines[pos['line']],pos['character'])
 left,right=offset(rr['start']),offset(rr['end'])
 if hashlib.sha256(document.encode()).hexdigest()!=rr['content_sha256']:raise ValueError('document range hash')
 if right!=len(document):raise ValueError('range not EOF')
 return document[:left]+replacement+document[right:]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise SystemExit('output must be fresh')
 for p,s in {**PINS,**SOURCE_CASE_PINS}.items():
  actual=sha(p)
  if actual!=s:raise SystemExit(f'pin mismatch:{p}:{actual}')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,trust_remote_code=False)
 new_rows=load_map(NEW_ROWS,'id'); prov=load_map(NEW_PROV,'row_id')
 if len(new_rows)!=3503 or len(prov)!=3503 or set(new_rows)!=set(prov):raise ValueError('new join mismatch')
 source_lines={}; wanted={p:set() for p in SOURCE_CASE_PINS}
 for x in prov.values():
  ident=x['source_provenance']['source_identity'];path=Path(ident['file'])
  if path not in wanted or SOURCE_CASE_PINS[path]!=ident['source_sha256']:raise ValueError('unbound source file')
  wanted[path].add(ident['line'])
 for path,line_numbers in wanted.items():
  with path.open() as f:
   for n,line in enumerate(f,1):
    if n in line_numbers:source_lines[(path,n)]=(line,json.loads(line))
 if set(source_lines)!={(p,n) for p,ns in wanted.items() for n in ns}:raise ValueError('source replay lines missing')
 attempt=a.output.with_name(a.output.name+f'.attempt-{os.getpid()}');attempt.mkdir(parents=True)
 parse_dir=attempt/'parse-staging';parse_dir.mkdir(); parse_paths=[]; ledger=[]; changed=0; unchanged=0; maxseq=0; maxtarget=0
 out_rows=attempt/'train-token-rows.jsonl'; out_ctx=attempt/'context-sidecar.jsonl'
 with ROWS.open() as rf, CONTEXTS.open() as cf, out_rows.open('w') as of, out_ctx.open('w') as oc:
  for position,(rl,cl) in enumerate(zip(rf,cf,strict=True)):
   row=json.loads(rl);ctxrec=json.loads(cl)
   if row['id']!=ctxrec['row_id']:raise ValueError(f'context order:{position}')
   validate_training_row(row); context=PromptContext.from_mapping(ctxrec['context'])
   oc.write(cl); old=row
   if row['id'] in new_rows:
    p=prov[row['id']];sp=p['source_provenance'];ident=sp['source_identity']; rawline,raw=source_lines[(Path(ident['file']),ident['line'])]
    document=sp['selection_source']['document_text']; rr=ctxrec['context']['replacement_range']
    old_applied=apply(document,rr,row['target_body_text'])
    source_target=raw.get('corpus_target',raw.get('target'))
    normalized_source_target=source_target.replace('\r\n','\n') if isinstance(source_target,str) else source_target
    normalized_source_prefix=raw.get('prefix','').replace('\r\n','\n')
    checks={
     'raw_line_sha':line_sha(rawline)==ident['raw_line_sha256'],
     'candidate_row_target':row['target_body_text']==new_rows[row['id']]['target_body_text'],
     'source_target_authority':isinstance(source_target,str) and sp['target_candidates'][0]['text_sha256']==hashlib.sha256(normalized_source_target.encode()).hexdigest(),
     'literal_source_fragment_replay':old_applied==normalized_source_prefix+normalized_source_target,
     'extractor_gate':raw.get('gates',{}).get('splice_exact','').startswith('internal@rules_finish_block'),
     'family':raw.get('family')=='finish_block'==row['family'], 'split':ident.get('split')=='train_group',
     'constructor':sp['pre_edit_document']['source_constructor']=='finish_block_v5_prefix',
     'boundary':sp['r_fragment']['body_fragment_boundary']=='raw_prefix_plus_corpus_target_before_outer_brace',
     'outer_absent':sp['finish_splice']=={'literal_source_splice_verified':True,'outer_closing_brace_in_label':False},
     'target_hash':hashlib.sha256(row['target_body_text'].encode()).hexdigest()==sp['target_body_sha256'],
    }
    if not all(checks.values()):raise ValueError(f'source evidence:{row["id"]}:{checks}')
    repaired_body=row['target_body_text']+'}'
    rebuilt=build_training_row(context,operation='replace',region_new=repaired_body.split('\n'),tokenizer=tok,row_id=row['id'],family=row['family'],package_id=row['package_id'],split='train')
    stable=['id','family','package_id','renderer_id','prompt_text','prompt_token_count','target_start','bos_token_id','eos_token_id','tokenizer_revision','tokenizer_json_sha256','tokenization_policy','split']
    if any(rebuilt[k]!=old[k] for k in stable) or rebuilt['input_ids'][:rebuilt['target_start']]!=old['input_ids'][:old['target_start']]:raise ValueError('prompt identity changed')
    applied=apply(document,rr,repaired_body)
    pp=parse_dir/f'{position:05d}.R';pp.write_text(applied);parse_paths.append(pp)
    ledger.append({'position':position,'row_id':row['id'],'source_file':ident['file'],'source_file_sha256':ident['source_sha256'],'source_line':ident['line'],'source_raw_line_sha256':ident['raw_line_sha256'],'extractor_path':str(EXTRACTOR),'extractor_sha256':PINS[EXTRACTOR],'extractor_rule':'node_text(braced_expression)[1:-1] excludes exactly opening and closing AST brace','original_line_sha256':hashlib.sha256(rl.encode()).hexdigest(),'original_target_body_sha256':sp['target_body_sha256'],'repaired_target_body_sha256':hashlib.sha256(repaired_body.encode()).hexdigest(),'appended_bytes_hex':'7d','appended_outer_brace':True,'prompt_text_sha256':hashlib.sha256(row['prompt_text'].encode()).hexdigest(),'prompt_ids_sha256':hashlib.sha256(json.dumps(row['input_ids'][:row['target_start']],separators=(',',':')).encode()).hexdigest(),'replacement_range':ctxrec['context']['replacement_range'],'context_suffix_lines':ctxrec['context']['suffix_lines'],'applied_document_sha256':hashlib.sha256(applied.encode()).hexdigest(),'source_checks':checks,'raw_applied_parse_ok':None,'new_line_sha256':hashlib.sha256((canonical(rebuilt)+'\n').encode()).hexdigest()})
    row=rebuilt;changed+=1
   else:
    unchanged+=1
   maxseq=max(maxseq,len(row['input_ids']));maxtarget=max(maxtarget,len(row['input_ids'])-row['target_start'])
   if len(row['input_ids'])>4096 or len(row['input_ids'])-row['target_start']>1024:raise ValueError(f'cap:{row["id"]}')
   of.write(rl if row is old else canonical(row)+'\n')
 if changed!=3503 or unchanged!=11503:raise ValueError('coverage')
 paths=parse_dir/'paths.txt';results=parse_dir/'results.txt';paths.write_text(''.join(str(x)+'\n' for x in parse_paths))
 run=subprocess.run(['Rscript','--vanilla',str(HERE/'parse_only.R'),str(paths),str(results)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=900,check=False)
 statuses=results.read_text().splitlines() if results.exists() else []
 if run.returncode or len(statuses)!=3503 or any(x!='1' for x in statuses):raise ValueError(f'R parse failed:exit{run.returncode}:count{len(statuses)}:bad{sum(x!="1" for x in statuses)}')
 for x in ledger:x['raw_applied_parse_ok']=True
 with (attempt/'repair-ledger.jsonl').open('w') as f:
  for x in ledger:f.write(canonical(x)+'\n')
 shutil.rmtree(parse_dir)
 artifacts={}
 for path in [out_rows,out_ctx,attempt/'repair-ledger.jsonl']:
  artifacts[path.name]={'path':str(a.output/path.name),'bytes':path.stat().st_size,'sha256':sha(path)}
 manifest={'schema':'sepalith.dat10.finish-source-repair.v3','status':'prepared_root_review_required','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':{str(p):s for p,s in {**PINS,**SOURCE_CASE_PINS}.items()},'coverage':{'rows':15006,'changed_finish_rows':3503,'unchanged_rows_byte_exact':11503,'raw_applied_parse_pass':3503,'raw_applied_parse_fail':0,'targets_truncated':0,'max_sequence_tokens':maxseq,'max_target_tokens_including_eos':maxtarget},'repair':{'operation':'append exactly one source-proven ASCII outer brace byte','appended_bytes_hex':'7d','prompt_and_range_unchanged':True,'source_extractor_rule':'braced expression bytes [1:-1] excluded the exact outer brace','original_rows_preserved':True},'artifacts':artifacts,'launch_authorized':False}
 (attempt/'materialization.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 os.replace(attempt,a.output);print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
