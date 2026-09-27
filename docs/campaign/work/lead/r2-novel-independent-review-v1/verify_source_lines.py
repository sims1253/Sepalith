#!/usr/bin/env python3
"""Replay frozen TRAIN source-line hashes without parsing or executing R."""
import collections,datetime,hashlib,json,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
PACKET=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/expansion-increment-na-rm-v1/root-review-packet-v4-na-rm.jsonl')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 started=time.monotonic();needed=collections.defaultdict(lambda:collections.defaultdict(list))
 with PACKET.open() as f:
  for text in f:
   x=json.loads(text);sr=x['source_ref'];needed[Path(sr['file'])][int(sr.get('line') or sr.get('source_line'))].append((x['row']['id'],sr['raw_line_sha256']))
 files=[];mismatch=[];matched=0;requested=0
 for path,lines in sorted(needed.items(),key=lambda z:str(z[0])):
  requested+=sum(len(v) for v in lines.values());found=set();read_bytes=0;read_lines=0
  with path.open('rb') as f:
   for line_no,line in enumerate(f,1):
    read_bytes+=len(line);read_lines=line_no
    if line_no not in lines:continue
    found.add(line_no); actual=hashlib.sha256(line).hexdigest();trimmed=hashlib.sha256(line.rstrip(b'\r\n')).hexdigest()
    for rid,expected in lines[line_no]:
     if expected in (actual,trimmed):matched+=1
     else:mismatch.append({'row_id':rid,'path':str(path),'line':line_no,'expected':expected,'actual_including_separator':actual,'actual_without_separator':trimmed})
    if found==set(lines):break
  missing=sorted(set(lines)-found)
  for line_no in missing:
   for rid,expected in lines[line_no]:mismatch.append({'row_id':rid,'path':str(path),'line':line_no,'expected':expected,'failure':'line_missing'})
  files.append({'path':str(path),'file_bytes':path.stat().st_size,'lines_requested':sum(len(v) for v in lines.values()),'distinct_lines_requested':len(lines),'lines_read':read_lines,'bytes_read':read_bytes,'matched':sum(1 for vals in lines.values() for rid,e in vals if not any(m['row_id']==rid for m in mismatch)),'mismatched_or_missing':sum(1 for vals in lines.values() for rid,e in vals if any(m['row_id']==rid for m in mismatch))})
 out={'schema':'sepalith.dat10.novel-source-line-replay.v1','status':'pass' if not mismatch and matched==requested else 'fail','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packet':{'path':str(PACKET),'sha256':sha(PACKET),'rows':requested},'requested_rows':requested,'matched_rows':matched,'mismatched_or_missing_rows':len(mismatch),'files':files,'mismatches':mismatch,'elapsed_seconds':time.monotonic()-started,'constraints':{'cpu_threads':2,'nice':10,'ionice':'idle','r_executed':False,'heldout_payload_opened':False}}
 (WORK/'source-line-replay.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'requested':requested,'matched':matched,'mismatched':len(mismatch),'files':len(files),'bytes_read':sum(x['bytes_read'] for x in files),'elapsed':out['elapsed_seconds']}))
if __name__=='__main__':main()
