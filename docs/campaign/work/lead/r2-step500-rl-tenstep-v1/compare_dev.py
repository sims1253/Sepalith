#!/usr/bin/env python3
"""Compare complete paired DEV75 artifacts; refuse partial or changed panels."""
import argparse,datetime,hashlib,json
from pathlib import Path

def load(path):
 raw=path.read_bytes();value=json.loads(raw)
 if value.get('status')!='complete':raise ValueError(f'{path.name}: development evaluation is not complete')
 rows=value['results'];byid={r['id']:r for r in rows}
 if len(rows)!=75 or len(byid)!=75:raise ValueError('requires 75 unique cases')
 if sum(bool(r['expected_noop']) for r in rows)!=32:raise ValueError('requires 32 no-op and 43 edit cases')
 return byid,hashlib.sha256(raw).hexdigest()

def metrics(rows):
 values=list(rows.values());prompt=sum(r['loss']['prompt_tokens'] for r in values);target=sum(r['loss']['target_tokens'] for r in values)
 return {'cases':len(values),'edit_cases':sum(not r['expected_noop'] for r in values),'noop_cases':sum(r['expected_noop'] for r in values),'exact_edits':sum(not r['expected_noop'] and r['exact_region'] for r in values),'correct_noops':sum(r['expected_noop'] and r['exact_region'] for r in values),'false_positive_suggestions':sum(r['expected_noop'] and r['suggestion'] for r in values),'protocol_valid':sum(r['protocol_valid'] for r in values),'caps':sum(r['cap_hit'] for r in values),'prompt_tokens':prompt,'target_tokens':target,'prompt_nll':sum(r['loss']['prompt_nll_sum'] for r in values)/prompt,'target_nll':sum(r['loss']['target_nll_sum'] for r in values)/target}

def compare(a,b):
 old,oldsha=load(a);new,newsha=load(b)
 if old.keys()!=new.keys():raise ValueError('case identity mismatch')
 for key in old:
  for field in ['expected_noop','family','package_id','prompt_tokens','reference_tokens','strata']:
   if old[key][field]!=new[key][field]:raise ValueError(f'case contract mismatch: {key} {field}')
 changes=[]
 for key in old:
  delta={field:{'before':old[key][field],'after':new[key][field]} for field in ['exact_region','suggestion','protocol_valid','cap_hit','predicted_noop'] if old[key][field]!=new[key][field]}
  if delta:changes.append({'id':key,'family':new[key]['family'],'expected_noop':new[key]['expected_noop'],'changes':delta})
 return {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete_paired_development_comparison','before':{'path':str(a),'sha256':oldsha,'metrics':metrics(old)},'after':{'path':str(b),'sha256':newsha,'metrics':metrics(new)},'changes':changes,'promotion':False,'limitations':['Development evidence only; final set remains sealed.','Different fresh training lengths replay initial updates; not independent samples.','HF adapter quality does not prove quantized runtime or speculative acceptance.']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=compare(a.before,a.after);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','before','after','changes']}))
