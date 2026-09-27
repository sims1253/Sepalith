#!/usr/bin/env python3
"""Prepare a semantic-preserving microbatch-2 schedule and analyze document packing."""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_tensor_row(x):
 ids=x['input_ids']; labels=x['labels']; mask=x['attention_mask']; rid=x['row_id']
 if not rid or not (len(ids)==len(labels)==len(mask)) or mask!=[1]*len(ids):raise ValueError(f'invalid tensor row:{rid}')
 if ids[0]!=0 or ids[-1]!=1 or labels[0]!=-100:raise ValueError(f'boundary contract:{rid}')
 if not any(v!=-100 for v in labels[1:]):raise ValueError(f'no shifted loss:{rid}')
 return {'row_id':rid,'length':len(ids),'input_tokens':len(ids),'loss_tokens':sum(v!=-100 for v in labels[1:]),'package':x['package'],'document_id':x['document_id']}
def paired_schedule(rows,seed=20260914,effective_batch=16):
 """Adjacent-length microbatch pairs, randomized whole optimizer blocks, replay only at end."""
 if effective_batch!=16:raise ValueError('reviewed full-weight contract requires effective batch 16')
 byid={x['row_id']:x for x in rows}
 if len(byid)!=len(rows):raise ValueError('duplicate row ID')
 ordered=sorted(rows,key=lambda x:(x['length'],x['row_id']))
 n=(-len(ordered))%effective_batch
 # Named deterministic replay comes only after all unique rows.
 replay=ordered[:n]
 unique_ids=[x['row_id'] for x in ordered]
 blocks=[unique_ids[i:i+effective_batch] for i in range(0,len(unique_ids)-len(unique_ids)%effective_batch,effective_batch)]
 tail=unique_ids[len(blocks)*effective_batch:]+[x['row_id'] for x in replay]
 rng=random.Random(seed); rng.shuffle(blocks)
 # Avoid a length curriculum while retaining adjacent-length pairs inside every block.
 draws=[rid for block in blocks for rid in block]+tail
 # Replays are all at the terminal tail; uniqueness is assessed by first occurrences.
 first=[];seen=set()
 for rid in draws:
  if rid not in seen:first.append(rid);seen.add(rid)
 if set(first)!=set(unique_ids) or len(first)!=len(rows):raise AssertionError('coverage failure')
 pair_padding=0
 for i in range(0,len(draws),2):
  a,b=(byid[draws[i]]['length'],byid[draws[i+1]]['length']);pair_padding+=2*max(a,b)-a-b
 return {'row_ids':draws,'unique_rows':len(rows),'named_replays':n,'updates':len(draws)//effective_batch,'effective_batch':16,'micro_batch':2,'gradient_accumulation':8,'pair_padding_tokens':pair_padding,'pair_padding_fraction':pair_padding/sum(byid[x]['length'] for x in draws)}
def best_fit_decreasing_count(lengths,cap):
 """Deterministic bin count for analysis. Does not change or emit training rows."""
 import bisect
 remaining=[]
 for length in sorted(lengths,reverse=True):
  if length>cap:raise ValueError('row exceeds packing cap')
  i=bisect.bisect_left(remaining,length)
  if i==len(remaining):bisect.insort(remaining,cap-length)
  else:
   value=remaining.pop(i)-length;bisect.insort(remaining,value)
 return len(remaining),sum(remaining),sum((cap-x)**2 for x in remaining)
def pack_rows(rows,cap):
 """Reference packed-row constructor used only for conservation/quality experiments."""
 out=[]; ids=[];labels=[];members=[]
 for x in rows:
  validate_tensor_row(x)
  if len(ids)+len(x['input_ids'])>cap:raise ValueError('members exceed cap')
  start=len(ids); ids.extend(x['input_ids']);labels.extend(x['labels'])
  members.append({'row_id':x['row_id'],'package':x['package'],'document_id':x['document_id'],'start':start,'end':len(ids),'tensor_sha256':canonical({'input_ids':x['input_ids'],'labels':x['labels'],'attention_mask':x['attention_mask']})})
 return {'schema':'sepalith.cpt.packed-row.experimental.v1','row_id':'packed-'+canonical(members)[:24],'input_ids':ids,'labels':labels,'attention_mask':[1]*len(ids),'members':members,'cross_document_attention':'allowed_causal','boundary_loss_masking':'original labels retained byte-for-byte'}
def quantiles(v):
 v=sorted(v); n=len(v)
 def q(x):return v[min(n-1,int((n-1)*x))]
 return {'min':v[0],'p25':q(.25),'median':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'p99':q(.99),'max':v[-1]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True);p.add_argument('--rows-sha256',required=True);p.add_argument('--heldout',type=Path,required=True);p.add_argument('--heldout-sha256',required=True);p.add_argument('--schedule-output',type=Path,required=True);p.add_argument('--report-output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260914);a=p.parse_args()
 if a.schedule_output.exists() or a.report_output.exists():raise ValueError('outputs must be fresh')
 if sha(a.rows)!=a.rows_sha256 or sha(a.heldout)!=a.heldout_sha256:raise ValueError('input hash mismatch')
 rows=[];packages=set();documents=set();total_in=total_loss=0
 with a.rows.open() as f:
  for line in f:
   raw=json.loads(line); x=validate_tensor_row(raw);rows.append(x);packages.add(x['package']);documents.add(x['document_id']);total_in+=x['input_tokens'];total_loss+=x['loss_tokens']
 held_packages=set();held_documents=set()
 with a.heldout.open() as f:
  for line in f:
   x=json.loads(line);held_packages.add(x['package']);held_documents.add(x['document_id'])
 if packages&held_packages or documents&held_documents:raise ValueError('heldout overlap')
 schedule=paired_schedule(rows,a.seed); schedule.update({'schema':'sepalith.cpt.micro2-length-paired-schedule.v1','seed':a.seed,'token_rows_sha256':a.rows_sha256,'row_ids_sha256':canonical(schedule['row_ids']),'policy':'all unique row IDs first by first occurrence; named terminal replay only'})
 a.schedule_output.write_text(json.dumps(schedule,indent=2,sort_keys=True)+'\n')
 lengths=[x['length'] for x in rows]; standalone_position_pairs=sum(x*x for x in lengths); pack={}
 for cap in (8192,16384):
  count,waste,packed_pairs=best_fit_decreasing_count(lengths,cap);pack[str(cap)]={'packed_sequences':count,'unused_capacity_tokens':waste,'utilization':sum(lengths)/(count*cap),'causal_attention_position_pair_ratio_vs_standalone_rows':packed_pairs/standalone_position_pairs,'optimizer_updates_at_effective_batch16':math.ceil(count/16),'attention_semantics':'tokens can causally attend across member boundaries; original BOS/EOS and labels remain but standalone-document conditioning does not'}
 report={'schema':'sepalith.cpt.throughput-analysis.v1','scope':'frozen_remaining_30421_cohort_evidence_not_all_corpus','rows':len(rows),'packages':len(packages),'documents':len(documents),'input_tokens':total_in,'loss_tokens':total_loss,'lengths':quantiles(lengths),'counts':{'lt512':sum(x<512 for x in lengths),'lt1024':sum(x<1024 for x in lengths),'eq2048':sum(x==2048 for x in lengths)},'micro2_candidate':{k:schedule[k] for k in ('unique_rows','named_replays','updates','micro_batch','gradient_accumulation','pair_padding_tokens','pair_padding_fraction')},'packing_analysis':pack,'observed_reference':{'8k_effective_batch16_seconds_approx':50,'8k_input_tokens_per_update':131072,'16k_effective_batch16_seconds_approx':147,'16k_input_tokens_per_update':262144,'optimizer_step_seconds_approx':6,'claims':'observations supplied by root; not reproduced by CPU preparation'},'recommendation':'First benchmark length-paired microbatch2/accumulation8 at 8K against microbatch1/accumulation16 on identical row IDs. It preserves standalone attention and all labels. Do not admit document packing without a controlled heldout-NLL comparison because cross-document attention changes conditioning.'}
 a.report_output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'schedule_sha256':sha(a.schedule_output),'report_sha256':sha(a.report_output),'rows':len(rows),'micro2_padding_fraction':schedule['pair_padding_fraction']}))
if __name__=='__main__':main()
