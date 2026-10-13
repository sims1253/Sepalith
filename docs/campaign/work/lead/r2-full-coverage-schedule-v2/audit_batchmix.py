#!/usr/bin/env python3
"""Independent row-join, coverage, order, and padding audit for batch-mix v2."""
import collections,hashlib,json
from pathlib import Path
from full_coverage_schedule import load_rows,sha
H=Path(__file__).resolve().parent
ROWS=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-expanded-15008-v1/combined-token-rows.jsonl')
ROWS_SHA='f4c93a760b328f154087bc036e5ef4f36f4dc37e204499abfc8fa270c1fc67cc'
OLD=H.parent/'r2-full-coverage-schedule-v1/full-coverage-15008-draw-manifest.json';OLD_SHA='cd082ce03fad07e1031c5fdf0e6eb650fa7121a5ddb61b52e9e80324f5b0200c'
NEW=H/'full-coverage-15008-batchmix-draw-manifest.json';NEW_SHA='7d52208e25d06be50cf1b3f6596e2ca925a74f0c487e25c87b2abd6c4e464907'
OUT=H/'batchmix-coverage-padding-report.json'
def analyze(path,expected,byid):
 if sha(path)!=expected:raise ValueError('schedule SHA differs')
 value=json.loads(path.read_text());draws=value['draws'];exposure=collections.Counter();first={};first_replay={'edit':None,'noop':None};actual_padding=bucket_padding=0;edit_bucket_order=[]
 scheduled={'input_tokens':0,'prompt_tokens_excluding_bos':0,'supervised_target_tokens_including_eos':0,'target_body_tokens':0,'target_protocol_tokens_excluding_eos':0}
 for start in range(0,len(draws),16):
  batch=draws[start:start+16]
  if len(batch)!=16 or sum(d['semantic_noop'] for d in batch)!=4:raise ValueError('12 edit / 4 noop batch differs')
  batch_max=max(d['total_tokens'] for d in batch);declared={d['batch_sequence_bucket'] for d in batch}
  if len(declared)!=1 or next(iter(declared))<batch_max:raise ValueError('batch bucket differs')
  actual_padding+=batch_max*16-sum(d['total_tokens'] for d in batch);bucket_padding+=next(iter(declared))*16-sum(d['total_tokens'] for d in batch)
  edit_bucket_order.append(max(d['sequence_bucket'] for d in batch if not d['semantic_noop']))
  for d in batch:
   r=byid.get(d['row_id'])
   if r is None:raise ValueError('unknown row')
   expected_row={'family':r['family'],'package_id':r['package_id'],'semantic_noop':r['semantic_noop'],'prompt_tokens':r['prompt_tokens'],'supervised_target_tokens':r['supervised_target_tokens'],'target_body_tokens':r['target_body_tokens'],'target_protocol_tokens':r['target_protocol_tokens'],'total_tokens':r['total_tokens'],'sequence_bucket':r['sequence_bucket']}
   if any(d.get(k)!=v for k,v in expected_row.items()):raise ValueError('draw/row join differs')
   if d['supervised_target_tokens']!=d['target_body_tokens']+d['target_protocol_tokens']+1:raise ValueError('target truncated')
   key='noop' if d['semantic_noop'] else 'edit';exposure[d['row_id']]+=1
   if exposure[d['row_id']]==1:first[d['row_id']]=d['draw_index']
   elif first_replay[key] is None:first_replay[key]=d['draw_index']
   scheduled['input_tokens']+=d['total_tokens'];scheduled['prompt_tokens_excluding_bos']+=d['prompt_tokens'];scheduled['supervised_target_tokens_including_eos']+=d['supervised_target_tokens'];scheduled['target_body_tokens']+=d['target_body_tokens'];scheduled['target_protocol_tokens_excluding_eos']+=d['target_protocol_tokens']
 if set(exposure)!=set(byid):raise ValueError('eligible ID omitted')
 last_unique={'edit':max(first[x] for x,r in byid.items() if not r['semantic_noop']),'noop':max(first[x] for x,r in byid.items() if r['semantic_noop'])}
 if any(first_replay[k] is not None and first_replay[k]<=last_unique[k] for k in first_replay):raise ValueError('pool replay before full unique coverage')
 if any(value['token_denominators']['scheduled'][k]!=v for k,v in scheduled.items()):raise ValueError('scheduled token denominator differs')
 return {'sha256':expected,'updates':len(draws)//16,'draws':len(draws),'distinct_rows':len(exposure),'edit_draws':sum(not d['semantic_noop'] for d in draws),'noop_draws':sum(d['semantic_noop'] for d in draws),'last_unique_draw_by_pool':last_unique,'first_replay_draw_by_pool':first_replay,'actual_max_padding_tokens':actual_padding,'declared_bucket_padding_tokens':bucket_padding,'edit_bucket_transitions':sum(a!=b for a,b in zip(edit_bucket_order,edit_bucket_order[1:])),'distinct_edit_buckets_first_50_batches':len(set(edit_bucket_order[:50])),'first_20_edit_batch_buckets':edit_bucket_order[:20],'scheduled_token_denominators':scheduled}
def main():
 rows,excluded=load_rows(ROWS,ROWS_SHA,[512,1024,2048,4096])
 if excluded:raise ValueError('admitted row excluded')
 byid={r['row_id']:r for r in rows};old=analyze(OLD,OLD_SHA,byid);new=analyze(NEW,NEW_SHA,byid)
 result={'schema':'sepalith.sft.full-coverage-batchmix-audit.v1','status':'PASS','input':{'path':str(ROWS),'sha256':ROWS_SHA,'rows':len(rows),'row_ids_sha256':hashlib.sha256(('\n'.join(sorted(byid))+'\n').encode()).hexdigest()},'v1_long_first':old,'v2_seeded_batchmix':new,'change':{'actual_max_padding_tokens':new['actual_max_padding_tokens']-old['actual_max_padding_tokens'],'declared_bucket_padding_tokens':new['declared_bucket_padding_tokens']-old['declared_bucket_padding_tokens'],'scheduled_input_tokens':new['scheduled_token_denominators']['input_tokens']-old['scheduled_token_denominators']['input_tokens'],'edit_bucket_transitions':new['edit_bucket_transitions']-old['edit_bucket_transitions']},'checks':{'all_15008_ids_preserved':new['distinct_rows']==15008,'exact_12_edit_4_noop':new['edit_draws']==13920 and new['noop_draws']==4640,'full_targets_preserved':True,'unique_edit_before_replay':new['first_replay_draw_by_pool']['edit']>new['last_unique_draw_by_pool']['edit'],'unique_noop_before_replay':new['first_replay_draw_by_pool']['noop']>new['last_unique_draw_by_pool']['noop'],'not_long_first':new['distinct_edit_buckets_first_50_batches']>1 and new['edit_bucket_transitions']>old['edit_bucket_transitions'],'actual_padding_not_increased':new['actual_max_padding_tokens']<=old['actual_max_padding_tokens']}}
 if not all(result['checks'].values()):raise ValueError('batchmix acceptance check failed')
 if OUT.exists():raise ValueError('fresh report required')
 OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':result['checks'],'change':result['change']}))
if __name__=='__main__':main()
