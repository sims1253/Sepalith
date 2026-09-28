#!/usr/bin/env python3
"""Independently stream-compare a CPT cache with its immutable inputs."""
import argparse,json,mmap,resource,sqlite3,struct
from pathlib import Path
from cpt_streaming_cache import StreamingCptDataset,require,sha256

def verify(cache):
 cache=Path(cache);manifest=json.loads((cache/'manifest.json').read_text());d=StreamingCptDataset(cache,sha256(cache/'manifest.json'));db=sqlite3.connect(f'file:{cache/"index.sqlite3"}?mode=ro',uri=True)
 files=[]
 for name in ('input_ids.i32le','labels.i32le'):
  f=(cache/name).open('rb');files.append((f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)))
 rows=0
 with Path(manifest['source']['rows']['path']).open('rb') as source:
  for ordinal,line in enumerate(source):
   raw=json.loads(line);rec=db.execute('select row_id,token_offset,length,row_sha256 from rows where ordinal=?',(ordinal,)).fetchone();require(rec is not None and rec[0]==raw['row_id'],'row ordinal/identity differs');require(rec[3]==__import__('hashlib').sha256(line).hexdigest(),'row source hash differs');off,n=rec[1:3]
   require(list(struct.unpack_from(f'<{n}i',files[0][1],off*4))==raw['input_ids'],'cached input IDs differ');require(list(struct.unpack_from(f'<{n}i',files[1][1],off*4))==raw['labels'],'cached labels differ');rows+=1
 schedule=json.loads(Path(manifest['source']['draw_schedule']['path']).read_text());draws=schedule.get('schedule',schedule)['row_ids'];drawmap=d._open() or d._maps[2][1]
 for position,rid in enumerate(draws):
  ordinal=struct.unpack_from('<Q',d._maps[2][1],position*8)[0];actual=db.execute('select row_id from rows where ordinal=?',(ordinal,)).fetchone()[0];require(actual==rid,'cached draw identity differs')
 for f,m in files:m.close();f.close()
 db.close();d.close();return {'status':'PASS','rows':rows,'draws':len(draws),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'manifest_sha256':sha256(cache/'manifest.json')}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);a=p.parse_args();print(json.dumps(verify(a.cache),sort_keys=True))
