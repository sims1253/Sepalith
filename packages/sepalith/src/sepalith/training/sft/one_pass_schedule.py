#!/usr/bin/env python3
"""Build one full eligible-row pass plus only terminal batch padding."""
import argparse,collections,hashlib,json,os
from pathlib import Path
from sepalith.training.sft.full_coverage_schedule import load_rows,stable,denominator

def build(rows,excluded,seed=3407,effective_batch=16,buckets=(512,1024,2048,4096,8192,16384)):
 if excluded:raise ValueError('one-pass alternative cannot exclude an eligible row')
 pools=collections.defaultdict(list)
 for row in rows:pools[row['sequence_bucket']].append(row)
 chunks=[];leftovers=[]
 for bucket,pool in sorted(pools.items()):
  pool=sorted(pool,key=lambda r:stable(seed,'one-pass-bucket',bucket,r['row_id']))
  full=len(pool)//effective_batch
  chunks.extend((pool[i*effective_batch:(i+1)*effective_batch],'homogeneous_bucket_'+str(bucket)) for i in range(full))
  leftovers.extend(pool[full*effective_batch:])
 chunks.sort(key=lambda item:stable(seed,'one-pass-whole-batch',*(r['row_id'] for r in item[0])))
 leftovers=sorted(leftovers,key=lambda r:(r['sequence_bucket'],stable(seed,'one-pass-leftover',r['row_id'])))
 while len(leftovers)>=effective_batch:chunks.append((leftovers[:effective_batch],'cross_bucket_unique'));leftovers=leftovers[effective_batch:]
 exposure=collections.Counter();draws=[]
 for batch_index,(batch,kind) in enumerate(chunks,1):
  cap=max(r['sequence_bucket'] for r in batch)
  for row in batch:
   exposure[row['row_id']]+=1;draws.append(_draw(row,len(draws),batch_index,cap,exposure[row['row_id']],None,kind))
 # The terminal partial batch contains every remaining unique row, then exactly
 # enough deterministic already-seen rows to reach the effective batch.
 if leftovers:
  missing=effective_batch-len(leftovers);cap=max(r['sequence_bucket'] for r in leftovers)
  leftover_ids={r['row_id'] for r in leftovers}
  candidates=[r for r in rows if r['sequence_bucket']<=cap and r['row_id'] not in leftover_ids]
  replay=sorted(candidates,key=lambda r:stable(seed,'terminal-batch-padding',r['row_id']))[:missing]
  batch=leftovers+replay;batch_index=len(chunks)+1
  for row in batch:
   exposure[row['row_id']]+=1;reason=None if row['row_id'] in leftover_ids else 'terminal_batch_padding_replay'
   draws.append(_draw(row,len(draws),batch_index,cap,exposure[row['row_id']],reason,'terminal_partial_unique_chunk'))
 ids={r['row_id'] for r in rows};first={};first_replay=None
 for d in draws:
  first.setdefault(d['row_id'],d['draw_index'])
  if d['presentation']>1 and first_replay is None:first_replay=d['draw_index']
 if set(exposure)!=ids or first_replay is None or first_replay<=max(first.values()):raise AssertionError('unique-before-terminal-replay failed')
 unique=denominator(rows);scheduled=denominator(draws)
 return {'schema':'sepalith.sft.full-coverage-batchmix-draws.v2','status':'alternative_preparation_only_not_root_selected','policy':{'seed':seed,'effective_batch':effective_batch,'no_op_fraction':sum(r['semantic_noop'] for r in draws)/len(draws),'sequence_buckets':list(buckets),'target_cap':None,'target_policy':'retain complete target body, protocol terminal, and terminal EOS whenever full sequence fits','batch_order':'deterministic seeded mixing of whole actual-length-bucket batches; unique cross-bucket leftovers follow; terminal partial batch last','replay_policy':'exactly two terminal batch-padding replays after every eligible ID appears','selected_for_training':False},'max_steps':len(draws)//effective_batch,'draw_count':len(draws),'row_ids':sorted(ids),'excluded':[],'coverage':{'eligible_rows':len(rows),'eligible_edit_rows':sum(not r['semantic_noop'] for r in rows),'eligible_noop_rows':sum(r['semantic_noop'] for r in rows),'distinct_rows_drawn':len(exposure),'first_all_rows_step':len(draws)//effective_batch,'min_row_exposures':min(exposure.values()),'max_row_exposures':max(exposure.values()),'first_replay_draw':first_replay,'last_unique_draw':max(first.values()),'replay_draws_by_reason':{'terminal_batch_padding_replay':len(draws)-len(rows)},'exposure_histogram':dict(sorted(collections.Counter(exposure.values()).items()))},'token_denominators':{'unique':unique,'scheduled':scheduled},'checks':{'all_eligible_ids_drawn':set(exposure)==ids,'no_target_truncation':all(d['supervised_target_tokens']==d['target_body_tokens']+d['target_protocol_tokens']+1 for d in draws),'targets_over_1024_retained':sum(d['supervised_target_tokens']>1024 for d in draws),'every_unique_before_any_replay':first_replay>max(first.values())},'draws':draws}

def _draw(row,index,update,cap,presentation,reason,kind):
 return {'draw_index':index,'update':update,'row_id':row['row_id'],'family':row['family'],'package_id':row['package_id'],'semantic_noop':row['semantic_noop'],'presentation':presentation,'replay_reason':reason,'edit_chunk_kind':kind,'sequence_bucket':row['sequence_bucket'],'batch_sequence_bucket':cap,'prompt_tokens':row['prompt_tokens'],'supervised_target_tokens':row['supervised_target_tokens'],'target_body_tokens':row['target_body_tokens'],'target_protocol_tokens':row['target_protocol_tokens'],'total_tokens':row['total_tokens']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True);p.add_argument('--rows-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 buckets=(512,1024,2048,4096,8192,16384);rows,excluded=load_rows(a.rows,a.rows_sha256,buckets);result=build(rows,excluded);result['input']={'path':str(a.rows.resolve()),'sha256':a.rows_sha256,'rows_seen':len(rows),'eligible_row_ids_sha256':hashlib.sha256(('\n'.join(result['row_ids'])+'\n').encode()).hexdigest()};tmp=a.output.with_suffix('.new');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,a.output);print(json.dumps({'rows':len(rows),'draws':result['draw_count'],'updates':result['max_steps'],'replays':result['coverage']['replay_draws_by_reason'],'checks':result['checks']}))
if __name__=='__main__':main()
