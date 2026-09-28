#!/usr/bin/env python3
"""Build deterministic all-token CPT resource fixtures from two complete admitted documents."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
MASK=-100
RAW_SHA='84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab'
RECHUNK_SHA='6a9b72704b124ec0ecfd715de883f5faf4c8269a16c07d4810bc0e77338c7eea'
VALIDATOR_SHA='8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa'
SMOKE=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-full-weight-hybrid-smoke-root-v4/full_weight_smoke.py')
SMOKE_SHA='08281c5303ebe48b656fc38f19b7494f62876859bda695995558a027154e19ba'
SOURCES=(
 {'document_id':'ac3fbdfa0440227a9e7ab1a6ba6cd35aafc9d7ffdab7438331af843e9bd27fea','tokens':37154,'group':'000781-g-ddd90c45c26f72079f15','package':'geospatialsuite','group_id':'g-ddd90c45c26f72079f15','rows':19,'receipt_sha256':'07cc4c1f7896fbff8f6a05a8e7d4fda112812dbd3052cba8d2a7243373e427cd','rows_sha256':'f17a6db5816a6c79c470c6a3a66e02091332cbf1538895fecf9168276d45f6c5','documents_sha256':'f8c61ef5fccee25316e98c88c204aa29e81aa114fc754e1b53cfeb5a7e884ad0'},
 {'document_id':'5bfa39b16126efa998db0d51fb380f00fd71129e50718ecd4e2dbd554f1fcbc6','tokens':33063,'group':'000805-g-d75bc793b3a68ab86f69','package':'binGroup','group_id':'g-d75bc793b3a68ab86f69','rows':17,'receipt_sha256':'db34963f2787b8a1e084e3bba023255a1281d1e4a61706c4a6acab9b57fab8bd','rows_sha256':'54d6b1c124180ff38257561721d1b4dec2adfe2a46d71aabf8a2f2af218af2a1','documents_sha256':'b3d8166ccf95a4dd96d8dc69510ecacb496d6259e79ec044963bb2292d51c25f'},
)
BASE=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1/groups')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module
 assert spec.loader is not None; spec.loader.exec_module(module); return module

def validate_source(item):
 root=BASE/item['group']; receipt=root/'receipt.json'; rows=root/'cpt_train.jsonl'; documents=root/'documents.jsonl'
 if sha(receipt)!=item['receipt_sha256'] or sha(rows)!=item['rows_sha256'] or sha(documents)!=item['documents_sha256']: raise ValueError('committed group artifact differs')
 r=json.loads(receipt.read_text())
 if r.get('status')!='complete' or r.get('group_id')!=item['group_id'] or r.get('package')!=item['package'] or r['artifacts']['cpt_train.jsonl']['sha256']!=item['rows_sha256']: raise ValueError('committed group receipt differs')
 metadata=[row for row in map(json.loads,documents.read_text().splitlines()) if row.get('document_id')==item['document_id']]
 if len(metadata)!=1 or metadata[0].get('source_code_tokens')!=item['tokens'] or metadata[0].get('cpt_partition')!='cpt_train' or metadata[0].get('package')!=item['package']: raise ValueError('selected document metadata differs')
 selected=[row for row in map(json.loads,rows.read_text().splitlines()) if row.get('document_id')==item['document_id']]
 if len(selected)!=item['rows'] or any(x.get('document_token_count')!=item['tokens'] for x in selected): raise ValueError('selected original chunks differ')
 return root,receipt,rows,documents,metadata[0],selected

def prove_fixture(row,size):
 ids,labels,attention=row['input_ids'],row['labels'],row['attention_mask']
 if row.get('schema')!=1 or row.get('cpt_partition')!='cpt_train': raise ValueError('fixture is not schema-1 TRAIN')
 if row.get('source_sha256')!=row.get('document_id') or row.get('row_id')!=f"{row['document_id']}:ctx{size}:1": raise ValueError('fixture source/document/row identity differs')
 if len(ids)!=size or len(labels)!=size or attention!=[1]*size: raise ValueError('fixture must be exactly full with no padding')
 if row.get('chunk_index')!=1 or row.get('overlap_context_tokens')!=1 or row.get('is_document_end') is not False: raise ValueError('fixture must exercise continuation carry and nonterminal EOS')
 if ids[0]!=0 or ids[-1]!=1 or labels[:2]!=[MASK,MASK] or labels[-1]!=MASK or labels[2:-1]!=ids[2:-1]: raise ValueError('all-token labels/carry/nonterminal EOS differ')
 if row['source_token_end']-row['source_token_start']!=size-3 or row['supervised_tokens']!=size-3: raise ValueError('fixture supervised range differs')
 return {'row_id':row['row_id'],'document_id':row['document_id'],'tokens':size,'source_token_start':row['source_token_start'],'source_token_end':row['source_token_end'],'supervised_tokens':row['supervised_tokens'],'overlap_context_tokens':1,'terminal_eos_supervised':False,'padding_tokens':0}

def prepare(output):
 output=Path(output)
 if output.exists(): raise FileExistsError('fresh output required')
 for name,expected in [('raw_cpt_broader.py',RAW_SHA),('lossless_rechunk.py',RECHUNK_SHA),('campaign_cpt_data.py',VALIDATOR_SHA)]:
  if sha(HERE/name)!=expected: raise ValueError(f'accepted rechunker closure differs:{name}')
 if sha(SMOKE)!=SMOKE_SHA: raise ValueError('root smoke source differs')
 # The accepted rechunker atomically publishes its ``rechunked`` subdirectory.
 # Build the small enclosing preparation directory at its final name so every
 # persisted provenance path remains resolvable after this process exits.
 stage=output; stage.mkdir(parents=True)
 try:
  inputs=[]; source_provenance=[]
  original=stage/'original'; original.mkdir()
  for item in SOURCES:
   root,receipt,rows,documents,metadata,selected=validate_source(item)
   target=original/(item['document_id']+'.jsonl'); target.write_text(''.join(canonical(x)+'\n' for x in selected))
   inputs.append({'path':str(target.resolve()),'sha256':sha(target),'bytes':target.stat().st_size,'rows':len(selected),'documents':1,'payload_tokens':item['tokens'],'package':item['package'],'group_id':item['group_id'],'cpt_partition':'cpt_train','source_path':str(rows),'committed_receipt':str(receipt)})
   source_provenance.append({'document_id':item['document_id'],'document_tokens':item['tokens'],'package':item['package'],'group_id':item['group_id'],'source_path':metadata['path'],'source_sha256':metadata['sha256'],'committed_group_rows':{'path':str(rows),'sha256':item['rows_sha256']},'committed_documents':{'path':str(documents),'sha256':item['documents_sha256']},'committed_receipt':{'path':str(receipt),'sha256':item['receipt_sha256']},'selected_original':{'path':str(target),'sha256':sha(target),'rows':len(selected)}})
  manifest={'schema':'sepalith.cpt.lossless-rechunk-input.v1','context_sizes':[8192,16384,32768],'raw_chunks':{'path':str((HERE/'raw_cpt_broader.py').resolve()),'sha256':RAW_SHA,'bos':0,'eos':1,'source_chunk_size':2048},'inputs':inputs,'expected_totals':{'rows':sum(x['rows'] for x in inputs),'documents':2,'payload_tokens':sum(x['payload_tokens'] for x in inputs)}}
  manifest_path=stage/'input-manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  rechunk=load_module('fixture_lossless_rechunk',HERE/'lossless_rechunk.py'); result=rechunk.rechunk(manifest_path,stage/'rechunked')
  validator=load_module('fixture_campaign_cpt_data',HERE/'campaign_cpt_data.py')
  smoke=load_module('fixture_full_weight_smoke',SMOKE)
  fixture_dir=stage/'fixtures'; fixture_dir.mkdir(); fixture_stats={}
  for size in (8192,16384):
   source=stage/'rechunked'/f'cpt_train_ctx{size}.jsonl'; candidates=list(map(json.loads,source.read_text().splitlines()))
   by_id={x['document_id']:x for x in candidates if x['chunk_index']==1}
   rows=[by_id[item['document_id']] for item in SOURCES]
   proofs=[prove_fixture(row,size) for row in rows]
   [validator.validate_materialized_row(row,max_sequence_tokens=size) for row in rows]
   target=fixture_dir/f'cpt-smoke-ctx{size}-2rows.jsonl'; target.write_text(''.join(canonical(x)+'\n' for x in rows))
   loaded=smoke.read_rows(target,size)
   if len(loaded)!=2: raise ValueError('smoke harness row count differs')
   fixture_stats[str(size)]={'path':str(target.resolve()),'sha256':sha(target),'bytes':target.stat().st_size,'rows':2,'exact_sequence_tokens_each':size,'supervised_tokens':sum(x['supervised_tokens'] for x in rows),'proofs':proofs}
  proof={'schema':'sepalith.sft11.cpt-long-context-fixtures.v1','status':'prepared_no_cuda_no_launch','accepted_rechunker':{'artifact_manifest_sha256':'8bcc7a776c0861de579db8326b758e4148b4bf214b0b8d672b9b829975f4d90a','source_sha256':RECHUNK_SHA,'raw_chunks_sha256':RAW_SHA,'validator_sha256':VALIDATOR_SHA},'root_smoke':{'path':str(SMOKE),'sha256':SMOKE_SHA},'source_documents':source_provenance,'token_conservation':{'source_documents':2,'source_payload_tokens':70217,'rechunk_result_sha256':sha(stage/'rechunked'/'result.json'),'all_source_tokens_once_per_context':result['guarantees']['all_source_tokens_once_per_context'],'per_context_payload_tokens':{k:v['payload_tokens'] for k,v in result['outputs'].items()},'per_context_terminal_eos':{k:v['terminal_eos'] for k,v in result['outputs'].items()}},'fixtures':fixture_stats,'selection':'chunk_index=1 from each complete document; exact full-length continuation rows exercise masked carry and masked nonterminal EOS','padding':'none','truncation':False,'fabricated_tokens':False}
  (stage/'fixture-manifest.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
  for p in stage.rglob('*'):
   if p.is_file():
    with p.open('rb') as f: os.fsync(f.fileno())
  dfd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)
  pfd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(pfd);os.close(pfd)
  return proof
 except BaseException:
  shutil.rmtree(stage,ignore_errors=True); raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(canonical(prepare(a.output)))
if __name__=='__main__':main()
