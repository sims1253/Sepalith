#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True);p.add_argument('--rows-sha256',required=True);p.add_argument('--schedule',type=Path,required=True);p.add_argument('--schedule-sha256',required=True);p.add_argument('--count',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 if sha(a.rows)!=a.rows_sha256 or sha(a.schedule)!=a.schedule_sha256:raise ValueError('input hash mismatch')
 wanted=json.loads(a.schedule.read_text())['row_ids'][:a.count]; need=set(wanted);found={}
 with a.rows.open() as f:
  for line in f:
   x=json.loads(line)
   if x['row_id'] in need:found[x['row_id']]=x
 if set(found)!=need:raise ValueError('schedule rows missing')
 with a.output.open('w') as f:
  for rid in wanted:f.write(json.dumps(found[rid],sort_keys=True,separators=(',',':'))+'\n')
 print(json.dumps({'rows':len(wanted),'sha256':sha(a.output),'order_exact':True}))
if __name__=='__main__':main()
