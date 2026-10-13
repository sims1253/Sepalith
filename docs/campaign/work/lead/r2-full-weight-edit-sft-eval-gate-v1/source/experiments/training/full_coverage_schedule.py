#!/usr/bin/env python3
"""Build deterministic, length-bucketed full-coverage SFT draw schedules."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def stable(seed,*values):return hashlib.sha256((str(seed)+'\0'+'\0'.join(map(str,values))).encode()).hexdigest()
def bucket_for(total,buckets):
 for value in buckets:
  if total<=value:return value
 raise ValueError('eligible sequence has no bucket')
def load_rows(path,expected,buckets):
 if sha(path)!=expected:raise ValueError('token row SHA256 differs')
 rows=[];excluded=[];ids=set()
 with Path(path).open() as f:
  for line_number,line in enumerate(f,1):
   r=json.loads(line);rid=r.get('id')
   if not isinstance(rid,str) or not rid or rid in ids:raise ValueError(f'row ID absent/duplicate at line {line_number}')
   ids.add(rid);tokens=r.get('input_ids');start=r.get('target_start');target=r.get('target_token_count')
   if r.get('split')!='train' or not isinstance(tokens,list) or not tokens or tokens[0]!=r.get('bos_token_id') or tokens[-1]!=r.get('eos_token_id'):raise ValueError(f'{rid}: malformed TRAIN/BOS/EOS row')
   if start!=r.get('prompt_token_count')+1 or len(tokens)!=start+target+1:raise ValueError(f'{rid}: prompt/target geometry differs')
   if target!=r.get('target_body_token_count')+r.get('target_terminal_token_count') or tokens[start:start+r['target_body_token_count']]!=r.get('target_body_tokens') or tokens[start+ r['target_body_token_count']:-1]!=r.get('target_terminal_tokens'):raise ValueError(f'{rid}: complete target body/terminal differs')
   supervised=len(tokens)-start
   if supervised!=target+1 or supervised<1:raise ValueError(f'{rid}: terminal EOS supervision differs')
   row={'row_id':rid,'family':r['family'],'package_id':r['package_id'],'semantic_noop':r.get('target_operation')=='no_op','prompt_tokens':r['prompt_token_count'],'target_body_tokens':r['target_body_token_count'],'target_protocol_tokens':r['target_terminal_token_count'],'supervised_target_tokens':supervised,'total_tokens':len(tokens)}
   if len(tokens)>buckets[-1]:excluded.append(dict(row,reason='sequence_exceeds_max'))
   else:row['sequence_bucket']=bucket_for(len(tokens),buckets);rows.append(row)
 return rows,excluded
def ordered(rows,seed,label):return sorted(rows,key=lambda r:stable(seed,label,r['row_id']))
def edit_batch_queue(rows,seed,width):
 chunks=[];leftovers=[]
 for bucket,pool in sorted(collections.defaultdict(list,((bucket,[r for r in rows if r['sequence_bucket']==bucket]) for bucket in {r['sequence_bucket'] for r in rows})).items()):
  pool=ordered(pool,seed,'edit-bucket-'+str(bucket));full=len(pool)//width
  for index in range(full):
   group=pool[index*width:(index+1)*width];chunks.append((group,'homogeneous_bucket_'+str(bucket)))
  leftovers.extend(pool[full*width:])
 leftovers=ordered(leftovers,seed,'edit-bucket-leftovers')
 for index in range(0,len(leftovers)//width*width,width):chunks.append((leftovers[index:index+width],'cross_bucket_leftover_chunk'))
 chunks.sort(key=lambda x:stable(seed,'edit-whole-chunk-order',*(r['row_id'] for r in x[0])))
 partial=leftovers[len(leftovers)//width*width:]
 if partial:chunks.append((partial,'terminal_partial_unique_chunk'))
 return chunks
def denominator(rows):
 return {'rows':len(rows),'input_tokens':sum(r['total_tokens'] for r in rows),'prompt_tokens_excluding_bos':sum(r['prompt_tokens'] for r in rows),'supervised_target_tokens_including_eos':sum(r['supervised_target_tokens'] for r in rows),'target_body_tokens':sum(r['target_body_tokens'] for r in rows),'target_protocol_tokens_excluding_eos':sum(r['target_protocol_tokens'] for r in rows),'targets_over_1024_tokens':sum(r['supervised_target_tokens']>1024 for r in rows)}
def build(rows,excluded,seed,effective_batch,noop_per_batch,buckets):
 noops=ordered([r for r in rows if r['semantic_noop']],seed,'noop-unique');edits=ordered([r for r in rows if not r['semantic_noop']],seed,'edit-unique');edit_per=effective_batch-noop_per_batch
 if (noop_per_batch and not noops) or (edit_per and not edits):raise ValueError('requested batch composition has an empty pool')
 updates=max(math.ceil(len(noops)/noop_per_batch) if noop_per_batch else 0,math.ceil(len(edits)/edit_per) if edit_per else 0)
 edit_chunks=edit_batch_queue(edits,seed,edit_per);noop_remaining=list(noops)
 exposure=collections.Counter();draws=[];batch_buckets=collections.Counter();reasons=collections.Counter();chunk_kinds=collections.Counter()
 def replay(pool,capacity,label,slot):
  candidates=[r for r in pool if r['total_tokens']<=capacity] or list(pool)
  return min(candidates,key=lambda r:(exposure[r['row_id']],abs(capacity-r['total_tokens']),stable(seed,label,slot,r['row_id'])))
 def unique_noop(capacity,slot):
  fitting=[r for r in noop_remaining if r['total_tokens']<=capacity]
  candidates=fitting or noop_remaining
  row=min(candidates,key=lambda r:(abs(capacity-r['total_tokens']),stable(seed,'noop-unique-fit',slot,r['row_id'])))
  noop_remaining.remove(row);return row
 for update in range(1,updates+1):
  selected=[];chunk,chunk_kind=edit_chunks[update-1] if update<=len(edit_chunks) else ([], 'edit_batch_completion_alignment_replay')
  chunk_kinds[chunk_kind]+=1
  for row in chunk:
   exposure[row['row_id']]+=1;selected.append((row,None,exposure[row['row_id']],chunk_kind))
  while len(selected)<edit_per:
   row=replay(edits,max((r['total_tokens'] for r,_,_,_ in selected),default=buckets[-1]),'edit_batch_completion_alignment_replay',len(draws)+len(selected));reason='edit_batch_completion_alignment_replay'
   exposure[row['row_id']]+=1;selected.append((row,reason,exposure[row['row_id']],chunk_kind))
  capacity=max((r['total_tokens'] for r,_,_,_ in selected),default=buckets[-1])
  for _ in range(noop_per_batch):
   if noop_remaining:row=unique_noop(capacity,len(draws)+len(selected));reason=None
   else:row=replay(noops,capacity,'noop_ratio_length_alignment_replay',len(draws)+len(selected));reason='noop_ratio_length_alignment_replay'
   exposure[row['row_id']]+=1;selected.append((row,reason,exposure[row['row_id']],None))
  selected.sort(key=lambda x:(x[2]>1,stable(seed,'within-batch',update,x[0]['row_id'],x[2])))
  cap=bucket_for(max(r['total_tokens'] for r,_,_,_ in selected),buckets);batch_buckets[str(cap)]+=1
  for row,reason,presentation,row_chunk_kind in selected:
   if reason:reasons[reason]+=1
   draws.append({'draw_index':len(draws),'update':update,'row_id':row['row_id'],'family':row['family'],'package_id':row['package_id'],'semantic_noop':row['semantic_noop'],'presentation':presentation,'replay_reason':reason,'edit_chunk_kind':row_chunk_kind,'sequence_bucket':row['sequence_bucket'],'batch_sequence_bucket':cap,'prompt_tokens':row['prompt_tokens'],'supervised_target_tokens':row['supervised_target_tokens'],'target_body_tokens':row['target_body_tokens'],'target_protocol_tokens':row['target_protocol_tokens'],'total_tokens':row['total_tokens']})
 if noop_remaining or set(exposure)!={r['row_id'] for r in rows}:raise AssertionError('full coverage not achieved')
 first={};first_replay={'noop':None,'edit':None}
 for d in draws:
  first.setdefault(d['row_id'],d['draw_index'])
  key='noop' if d['semantic_noop'] else 'edit'
  if d['presentation']>1 and first_replay[key] is None:first_replay[key]=d['draw_index']
 last_first={'noop':max(first[r['row_id']] for r in noops),'edit':max(first[r['row_id']] for r in edits)}
 if any(first_replay[k] is not None and first_replay[k]<=last_first[k] for k in first_replay):raise AssertionError('pool replay preceded unique coverage')
 first_draws=set(first.values());draw_den=denominator(draws)
 return {'schema':'sepalith.sft.full-coverage-batchmix-draws.v2','status':'preparation_only_parent_and_recipe_unbound','policy':{'seed':seed,'effective_batch':effective_batch,'noop_per_batch':noop_per_batch,'edit_per_batch':edit_per,'no_op_fraction':noop_per_batch/effective_batch,'sequence_buckets':buckets,'target_cap':None,'target_policy':'retain complete target body, protocol terminal, and terminal EOS whenever full sequence fits','edit_batch_order':'deterministic seeded mixing of whole actual-length-bucket edit chunks; terminal partial unique chunk last','noop_selection':'closest fit from remaining unique no-op pool before any replay','unique_before_replay':'separate no-op and edit pools are exhausted before their respective named alignment replays','replay_reasons':['noop_ratio_length_alignment_replay','edit_batch_completion_alignment_replay']},'max_steps':updates,'draw_count':len(draws),'row_ids':sorted(exposure),'excluded':excluded,'coverage':{'eligible_rows':len(rows),'eligible_edit_rows':len(edits),'eligible_noop_rows':len(noops),'distinct_rows_drawn':len(exposure),'first_all_rows_step':max(d['update'] for d in draws if d['draw_index'] in first_draws),'min_row_exposures':min(exposure.values()),'max_row_exposures':max(exposure.values()),'last_unique_draw_by_pool':last_first,'first_replay_draw_by_pool':first_replay,'replay_draws_by_reason':dict(reasons),'edit_chunk_batches':dict(sorted(chunk_kinds.items())),'exposure_histogram':dict(sorted(collections.Counter(exposure.values()).items()))},'sequence_buckets':{'unique_rows':dict(sorted(collections.Counter(str(r['sequence_bucket']) for r in rows).items())),'draws':dict(sorted(collections.Counter(str(d['sequence_bucket']) for d in draws).items())),'batches':dict(sorted(batch_buckets.items()))},'token_denominators':{'unique':denominator(rows),'scheduled':draw_den},'checks':{'all_eligible_ids_drawn':len(exposure)==len(rows),'no_target_truncation':all(d['supervised_target_tokens']==d['target_body_tokens']+d['target_protocol_tokens']+1 for d in draws),'targets_over_1024_retained':sum(d['supervised_target_tokens']>1024 for d in draws),'every_unique_noop_before_noop_replay':first_replay['noop'] is None or first_replay['noop']>last_first['noop'],'every_unique_edit_before_edit_replay':first_replay['edit'] is None or first_replay['edit']>last_first['edit']},'draws':draws}
def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True);p.add_argument('--rows-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--effective-batch',type=int,default=16);p.add_argument('--noop-per-batch',type=int,default=4);p.add_argument('--max-sequence',type=int,default=4096);p.add_argument('--seed',type=int,default=3407);a=p.parse_args()
 if a.output.exists() or not 0<=a.noop_per_batch<=a.effective_batch:raise ValueError('fresh output and valid batch composition required')
 buckets=[value for value in (512,1024,2048,4096,8192,16384) if value<a.max_sequence]+[a.max_sequence];buckets=sorted(set(buckets))
 rows,excluded=load_rows(a.rows,a.rows_sha256,buckets);result=build(rows,excluded,a.seed,a.effective_batch,a.noop_per_batch,buckets);result['input']={'path':str(a.rows.resolve()),'sha256':a.rows_sha256,'rows_seen':len(rows)+len(excluded),'eligible_row_ids_sha256':hashlib.sha256(('\n'.join(result['row_ids'])+'\n').encode()).hexdigest()}
 temporary=a.output.with_suffix(a.output.suffix+'.new');temporary.write_text(json.dumps(result,indent=2)+'\n');os.replace(temporary,a.output);print(json.dumps({'output':str(a.output),'rows':len(rows),'excluded':len(excluded),'updates':result['max_steps'],'draws':result['draw_count'],'coverage':result['coverage'],'checks':result['checks']}))
if __name__=='__main__':main()
