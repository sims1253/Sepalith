#!/usr/bin/env python3
"""Freeze 384 exact 8K all-token TRAIN chunks from distinct admitted packages."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1/groups')
VALIDATION=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-corpus-preparation-v1/profile-shard-v2-2k/cpt_validation.jsonl')
VALIDATION_SHA='efb434950dcaab38432b61891e87de64ef1cffb80a6054c2dc739ae3801d28d8'
RECHUNK=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-lossless-rechunk-v2/source/lossless_rechunk.py')
RECHUNK_SHA='6a9b72704b124ec0ecfd715de883f5faf4c8269a16c07d4810bc0e77338c7eea'
RAW=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-lossless-rechunk-v2/source/raw_cpt_broader.py')
RAW_SHA='84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab'
COUNT=384;CONTEXT=8192

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def prepare(output):
 output=Path(output)
 if output.exists():raise FileExistsError('fresh output required')
 if sha(VALIDATION)!=VALIDATION_SHA or sha(RECHUNK)!=RECHUNK_SHA or sha(RAW)!=RAW_SHA:raise ValueError('pinned validator/rechunk input differs')
 heldout=[json.loads(x) for x in VALIDATION.read_text().splitlines() if x]
 heldout_packages={x['package'] for x in heldout};heldout_documents={x['document_id'] for x in heldout}
 candidates=[];seen_packages=set();metadata_files=0
 # Use only earlier, complete group directories; ignore active .staging and any group without a terminal receipt.
 for root in sorted(BASE.iterdir()):
  receipt_path=root/'receipt.json';docs_path=root/'documents.jsonl';rows_path=root/'cpt_train.jsonl'
  if not (receipt_path.is_file() and docs_path.is_file() and rows_path.is_file()):continue
  receipt=json.loads(receipt_path.read_text())
  if receipt.get('status')!='complete':continue
  if receipt.get('artifacts',{}).get('documents.jsonl',{}).get('sha256')!=sha(docs_path) or receipt.get('artifacts',{}).get('cpt_train.jsonl',{}).get('sha256')!=sha(rows_path):raise ValueError(f'committed group binding differs:{root.name}')
  metadata_files+=1
  eligible=[x for x in map(json.loads,docs_path.read_text().splitlines()) if x.get('cpt_partition')=='cpt_train' and x.get('source_code_tokens',0)>8190 and x.get('package') not in heldout_packages and x.get('document_id') not in heldout_documents]
  if not eligible:continue
  chosen=min(eligible,key=lambda x:x['document_id']);package=chosen['package']
  if package in seen_packages:continue
  seen_packages.add(package);candidates.append((root,receipt_path,docs_path,rows_path,receipt,chosen))
  if len(candidates)==COUNT:break
 if len(candidates)!=COUNT:raise ValueError(f'only {len(candidates)} distinct eligible packages with full 8K chunks')
 lossless=module('pilot_lossless',RECHUNK);raw=module('pilot_raw',RAW)
 output.mkdir(parents=True);rows_out=(output/'train384-ctx8192.jsonl').open('x');ledger=(output/'provenance.jsonl').open('x')
 row_ids=[];input_tokens=loss_tokens=source_tokens=original_rows=0
 try:
  for root,receipt_path,docs_path,rows_path,receipt,meta in candidates:
   doc_id=meta['document_id']; original=[x for x in map(json.loads,rows_path.read_text().splitlines()) if x.get('document_id')==doc_id]
   common,payload,original_ids=lossless.validate_and_reassemble(original)
   if len(payload)!=meta['source_code_tokens'] or common['package']!=meta['package']:raise ValueError('document metadata/token reconstruction differs')
   chunk=next(iter(raw.chunks(payload,CONTEXT)))
   if len(chunk['input_ids'])!=CONTEXT or chunk['source_token_start']!=0 or chunk['source_token_end']!=8190 or chunk['is_document_end'] is not False:raise ValueError('selected document does not yield a full nonterminal 8K chunk')
   row={'schema':1,'row_id':f'{doc_id}:ctx8192:0','document_id':doc_id,'package':common['package'],'group_id':common['group_id'],'cpt_partition':'cpt_train','source_path':common['source_path'],'source_sha256':common['source_sha256'],'chunk_index':0,**chunk}
   if row['labels'][0]!=-100 or row['labels'][1:-1]!=row['input_ids'][1:-1] or row['labels'][-1]!=-100 or row['attention_mask']!=[1]*CONTEXT:raise ValueError('all-token label contract differs')
   rows_out.write(canonical(row)+'\n');row_ids.append(row['row_id']);input_tokens+=CONTEXT;loss_tokens+=row['supervised_tokens'];source_tokens+=len(payload);original_rows+=len(original)
   ledger.write(canonical({'schema':'sepalith.sft11.cpt-lr-pilot-row-provenance.v1','row_id':row['row_id'],'document_id':doc_id,'package':common['package'],'group_id':common['group_id'],'selected_source_range':[0,8190],'complete_document_tokens':len(payload),'complete_original_row_ids':original_ids,'committed_group_rows':{'path':str(rows_path),'sha256':receipt['artifacts']['cpt_train.jsonl']['sha256']},'committed_documents':{'path':str(docs_path),'sha256':receipt['artifacts']['documents.jsonl']['sha256']},'committed_receipt':{'path':str(receipt_path),'sha256':sha(receipt_path)}})+'\n')
 finally:
  for f in (rows_out,ledger):f.flush();os.fsync(f.fileno());f.close()
 rows_path=output/'train384-ctx8192.jsonl';ledger_path=output/'provenance.jsonl'
 schedule={'schema':'sepalith.sft11.cpt-lr-pilot-draws.v1','seed':3407,'method':'sorted_committed_group_then_smallest_eligible_document_id_one_per_package','token_rows_sha256':sha(rows_path),'row_ids':row_ids}
 (output/'draws384.json').write_text(json.dumps(schedule,indent=2,sort_keys=True)+'\n')
 manifest={'schema':'sepalith.sft11.cpt-lr-pilot-panel.v1','status':'short_lr_experiment_panel_not_production_dataset_cap','rows':COUNT,'draws':COUNT,'updates':24,'effective_batch':16,'packages':COUNT,'documents':COUNT,'max_sequence_tokens':CONTEXT,'exact_sequence_tokens_each':CONTEXT,'input_tokens':input_tokens,'loss_tokens':loss_tokens,'selected_source_tokens':8190*COUNT,'complete_source_document_tokens_verified':source_tokens,'complete_original_rows_verified':original_rows,'named_replays':0,'validation':{'path':str(VALIDATION),'sha256':VALIDATION_SHA,'rows':len(heldout),'package_overlap':0,'document_overlap':0},'selection':{'metadata_files_scanned':metadata_files,'distinct_package_required':True,'minimum_document_tokens':8191,'scope':'short LR signal experiment only; unselected eligible data remains queued for production'},'artifacts':{'rows':{'path':str(rows_path.resolve()),'sha256':sha(rows_path),'bytes':rows_path.stat().st_size},'draw_schedule':{'path':str((output/'draws384.json').resolve()),'sha256':sha(output/'draws384.json'),'bytes':(output/'draws384.json').stat().st_size},'provenance':{'path':str(ledger_path.resolve()),'sha256':sha(ledger_path),'bytes':ledger_path.stat().st_size}},'pins':{'lossless_rechunk_sha256':RECHUNK_SHA,'raw_chunks_sha256':RAW_SHA,'validation_sha256':VALIDATION_SHA},'guarantees':{'no_padding':True,'no_training_row_truncation':True,'all_new_raw_code_labels_supervised':True,'nonterminal_eos_masked':True,'all_complete_source_documents_reassembled_before_chunk_selection':True}}
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(canonical(prepare(a.output)))
if __name__=='__main__':main()
