"""Exact target-only editing rows and deterministic full-coverage schedule."""
import hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate_row(row,max_sequence_tokens):
 rid=row.get('id'); ids=row.get('input_ids'); start=row.get('target_start'); body=row.get('target_body_tokens'); terminal=row.get('target_terminal_tokens')
 if not isinstance(rid,str) or not rid:raise ValueError('row ID missing')
 if row.get('split')!='train' or row.get('renderer_id')!='zeta2-prm03-v1':raise ValueError(f'{rid}: split/renderer differs')
 if row.get('tokenizer_json_sha256')!='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81':raise ValueError(f'{rid}: tokenizer differs')
 if not isinstance(ids,list) or not 4<=len(ids)<=max_sequence_tokens or any(type(x) is not int or not 0<=x<130560 for x in ids):raise ValueError(f'{rid}: sequence invalid')
 if ids[0]!=0 or ids[-1]!=1 or ids.count(0)!=1 or ids.count(1)!=1:raise ValueError(f'{rid}: BOS/EOS differs')
 if start!=row.get('prompt_token_count')+1 or not 1<start<len(ids)-1:raise ValueError(f'{rid}: prompt boundary differs')
 if not isinstance(body,list) or not isinstance(terminal,list) or not terminal:raise ValueError(f'{rid}: target parts missing')
 if row.get('target_body_token_count')!=len(body) or row.get('target_terminal_token_count')!=len(terminal) or row.get('target_token_count')!=len(body)+len(terminal):raise ValueError(f'{rid}: target counts differ')
 if ids[start:]!=body+terminal+[1]:raise ValueError(f'{rid}: complete target body/terminal/EOS differs')
 if len(ids)-start>max_sequence_tokens:raise ValueError(f'{rid}: target exceeds sequence contract')
 return {'id':rid,'input_ids':ids,'target_start':start,'target_body_tokens':body,'target_terminal_tokens':terminal,'family':row['family'],'package_id':row['package_id']}
def inspect(rows_record,schedule_record,max_sequence_tokens,effective_batch=16):
 rows_path=Path(rows_record['path']);schedule_path=Path(schedule_record['path'])
 if sha(rows_path)!=rows_record['sha256'] or sha(schedule_path)!=schedule_record['sha256']:raise ValueError('data/schedule hash differs')
 rows=[];index={};tot={'input_tokens':0,'loss_tokens':0,'body_tokens':0}
 with rows_path.open() as f:
  for line in f:
   x=validate_row(json.loads(line),max_sequence_tokens);rid=x['id']
   if rid in index:raise ValueError('duplicate row ID')
   index[rid]=len(rows);rows.append(x);tot['input_tokens']+=len(x['input_ids']);tot['loss_tokens']+=len(x['input_ids'])-x['target_start'];tot['body_tokens']+=len(x['target_body_tokens'])
 schedule=json.loads(schedule_path.read_text());records=schedule.get('draws')
 if schedule.get('schema')!='sepalith.sft.full-coverage-batchmix-draws.v2' or schedule.get('input',{}).get('sha256')!=rows_record['sha256'] or schedule.get('excluded')!=[]:raise ValueError('schedule source/exclusions differ')
 if schedule.get('row_ids')!=sorted(index) or schedule.get('coverage',{}).get('distinct_rows_drawn')!=len(rows):raise ValueError('schedule eligible identity differs')
 if not isinstance(records,list) or len(records)!=schedule.get('draw_count') or len(records)!=schedule.get('max_steps')*effective_batch:raise ValueError('schedule draw count differs')
 draws=[]
 for pos,r in enumerate(records):
  rid=r.get('row_id');row=rows[index[rid]] if rid in index else None
  if row is None or r.get('draw_index')!=pos or r.get('update')!=pos//effective_batch+1 or r.get('total_tokens')!=len(row['input_ids']) or r.get('supervised_target_tokens')!=len(row['input_ids'])-row['target_start'] or r.get('semantic_noop')!=(row['family']=='no_op'):raise ValueError(f'draw differs:{pos}')
  draws.append(index[rid])
 if set(draws)!=set(range(len(rows))) or schedule.get('checks',{}).get('no_target_truncation') is not True:raise ValueError('coverage/truncation differs')
 return rows,draws,schedule,tot
def target_only_collator(batch,max_sequence_tokens):
 import torch
 if not batch:raise ValueError('empty batch')
 checked=[]
 for x in batch:
  if len(x['input_ids'])>max_sequence_tokens or x['input_ids'][x['target_start']:]!=x['target_body_tokens']+x['target_terminal_tokens']+[1]:raise ValueError('target row changed')
  checked.append(x)
 width=max(len(x['input_ids']) for x in checked);ids=torch.full((len(checked),width),1,dtype=torch.long);mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
 for i,x in enumerate(checked):
  n=len(x['input_ids']);v=torch.tensor(x['input_ids'],dtype=torch.long);ids[i,:n]=v;mask[i,:n]=1;labels[i,x['target_start']:n]=v[x['target_start']:]
 return {'input_ids':ids,'attention_mask':mask,'labels':labels}
def verify_batch(batch,rows):
 import torch
 if set(batch)!= {'input_ids','attention_mask','labels'}:raise ValueError('model inputs differ')
 total=0
 for i,row in enumerate(rows):
  n=len(row['input_ids']);start=row['target_start'];
  if batch['input_ids'][i,:n].tolist()!=row['input_ids'] or not bool(batch['attention_mask'][i,:n].eq(1).all()) or not bool(batch['attention_mask'][i,n:].eq(0).all()):raise ValueError('tokens/mask differ')
  if not bool(batch['labels'][i,:start].eq(-100).all()) or batch['labels'][i,start:n].tolist()!=row['input_ids'][start:] or not bool(batch['labels'][i,n:].eq(-100).all()):raise ValueError('target labels differ')
  total+=n-start
 if int(batch['labels'][:,1:].ne(-100).sum())!=total:raise ValueError('shifted target denominator differs')
 return total
