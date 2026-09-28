#!/usr/bin/env python3
"""Freeze a receipt-bound contiguous CPT prefix for lossless 16K rechunking."""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter
from pathlib import Path

START = 775
TOTAL_GLOBAL_GROUPS = 8092
FINAL_EXCLUSIVE = START + TOTAL_GLOBAL_GROUPS
EMPTY_SHA = hashlib.sha256(b'').hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def write(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('xb') as f:f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--groups',type=Path,required=True);ap.add_argument('--progress',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    progress=json.loads(a.progress.read_text());end=progress['first_uncommitted_seeded_index']
    if progress.get('status')!='in_progress' or end-START!=progress.get('groups_committed'): raise ValueError('frozen progress is not a contiguous prefix')
    if not START < end <= FINAL_EXCLUSIVE: raise ValueError('frontier outside global schedule')
    for name,pin in {'lossless_rechunk.py':'0530b93885e4db9f4c8f87acc6e5efd9bd735f82b5b2c24b8006d145acb5fa13','campaign_cpt_data.py':'8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa','raw_cpt_broader.py':'84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab'}.items():
        if sha(a.source/name)!=pin: raise ValueError(f'owned source pin differs: {name}')
    inputs=[];bindings=[];repairs=[];totals=Counter();statuses=Counter();zero=[];seen_groups=set()
    for index in range(START,end):
        matches=list(a.groups.glob(f'{index:06d}-*'))
        if len(matches)!=1 or not matches[0].is_dir(): raise ValueError(f'group directory count differs at {index}')
        folder=matches[0];receipt_path=folder/'receipt.json';raw=receipt_path.read_bytes();receipt=json.loads(raw)
        status=receipt.get('status');
        if status not in ('complete','complete_with_repairs_pending'): raise ValueError(f'unaccepted receipt status at {index}: {status}')
        if receipt.get('schema')!='sepalith.cpt.all-eligible-group.v1' or receipt.get('seeded_index')!=index: raise ValueError(f'receipt identity differs at {index}')
        gid=receipt.get('group_id')
        if not isinstance(gid,str) or folder.name!=f'{index:06d}-{gid}' or gid in seen_groups: raise ValueError(f'group ID/path differs at {index}')
        seen_groups.add(gid);counts=receipt.get('counts',{});artifacts=receipt.get('artifacts',{})
        cpt=folder/'cpt_train.jsonl';cpt_pin=artifacts.get('cpt_train.jsonl',{})
        if not cpt.is_file() or cpt.stat().st_size!=cpt_pin.get('bytes'): raise ValueError(f'cpt_train byte size differs at {index}')
        for key in ('rows','documents','payload_tokens'):
            if type(counts.get(key,0)) is not int or counts.get(key,0)<0: raise ValueError(f'invalid {key} count at {index}')
            totals[key]+=counts.get(key,0)
        repair_count=counts.get('repair_items',0)
        if type(repair_count) is not int or repair_count<0: raise ValueError(f'invalid repair count at {index}')
        queue=folder/'repair-queue.jsonl';queue_pin=artifacts.get('repair-queue.jsonl',{})
        if not queue.is_file() or queue.stat().st_size!=queue_pin.get('bytes') or sha(queue)!=queue_pin.get('sha256'): raise ValueError(f'repair queue binding differs at {index}')
        queue_rows=[json.loads(line) for line in queue.read_text().splitlines() if line]
        if len(queue_rows)!=repair_count or (status=='complete_with_repairs_pending')!=(repair_count>0): raise ValueError(f'repair status/count differs at {index}')
        for row in queue_rows: repairs.append({'seeded_index':index,'group_id':gid,'package':receipt['package'],'receipt':str(receipt_path),'repair':row})
        if counts.get('rows',0)==0:
            if cpt.stat().st_size!=0 or cpt_pin.get('sha256')!=EMPTY_SHA or counts.get('documents',0)!=0 or counts.get('payload_tokens',0)!=0: raise ValueError(f'invalid zero-row group at {index}')
            zero.append({'seeded_index':index,'group_id':gid,'package':receipt['package'],'repair_items':repair_count})
        inputs.append({'seeded_index':index,'group_id':gid,'package':receipt['package'],'committed_receipt':str(receipt_path),'receipt_sha256':hashlib.sha256(raw).hexdigest(),'receipt_status':status,'path':str(cpt),'source_path':str(cpt),'bytes':cpt_pin['bytes'],'sha256':cpt_pin['sha256'],'rows':counts.get('rows',0),'documents':counts.get('documents',0),'payload_tokens':counts.get('payload_tokens',0),'cpt_partition':'cpt_train'})
        bindings.append({'seeded_index':index,'group_id':gid,'package':receipt['package'],'receipt_path':str(receipt_path),'receipt_sha256':hashlib.sha256(raw).hexdigest(),'status':status,'counts':counts,'cpt_train':cpt_pin,'repair_queue':queue_pin})
        statuses[status]+=1
    if len(inputs)!=progress['groups_committed'] or sum(statuses.values())!=len(inputs): raise ValueError('frozen group count differs')
    if sum(x.get('repair_items',0) for x in [b['counts'] for b in bindings])!=len(repairs): raise ValueError('aggregate repairs differ')
    manifest={'schema':'sepalith.cpt.lossless-rechunk-input.v1','context_sizes':[16384],
      'frontier':{'start_inclusive':START,'end_exclusive':end,'groups':len(inputs),'frozen_progress_path':str(a.progress),'frozen_progress_sha256':sha(a.progress)},
      'expected_totals':{'rows':totals['rows'],'documents':totals['documents'],'payload_tokens':totals['payload_tokens']},'inputs':inputs,
      'raw_chunks':{'path':str(a.source/'raw_cpt_broader.py'),'sha256':'84d6a865a5d86bc7b81a274942ce8798f37f471dd6a336e182da1a44b00ea7ab','bos':0,'eos':1,'source_chunk_size':2048},
      'source':{'lossless_rechunk_sha256':'0530b93885e4db9f4c8f87acc6e5efd9bd735f82b5b2c24b8006d145acb5fa13','campaign_cpt_data_sha256':'8ae0271b7c133171a4db85c9c4a693c0b1930b756d0499d7a46961ff1b2276fa'},
      'original_tokenizer':{'path':'/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json','sha256':'3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','retokenized':False},
      'queued':{'later_main_groups':{'start_inclusive':end,'end_exclusive':FINAL_EXCLUSIVE,'groups':FINAL_EXCLUSIVE-end},'alias_supplemental':'separate pending corpus','final_global_dedup':'pending after main and supplemental completion'}}
    write(a.output/'input-manifest.json',manifest)
    with (a.output/'receipt-bindings.jsonl').open('xb') as f:
        for row in bindings:f.write((canonical(row)+'\n').encode())
        f.flush();os.fsync(f.fileno())
    with (a.output/'repair-ledger.jsonl').open('xb') as f:
        for row in repairs:f.write((canonical(row)+'\n').encode())
        f.flush();os.fsync(f.fileno())
    audit={'schema':'sepalith.cpt.16k-frontier-audit.v1','status':'complete','frontier':manifest['frontier'],'receipt_statuses':dict(statuses),'zero_row_groups':zero,'repairs':repairs,'totals':dict(totals),'artifacts':{name:{'bytes':(a.output/name).stat().st_size,'sha256':sha(a.output/name)} for name in ('input-manifest.json','receipt-bindings.jsonl','repair-ledger.jsonl')},'queued':manifest['queued']}
    write(a.output/'frontier-audit.json',audit);print(canonical(audit))

if __name__=='__main__':main()
