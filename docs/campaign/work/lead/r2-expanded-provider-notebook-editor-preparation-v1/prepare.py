#!/usr/bin/env python3
import hashlib,json,pathlib,tarfile,sys
ROOT=pathlib.Path(__file__).resolve().parent
REMOTE=ROOT/'remote'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def closure():
 rows=[]
 for p in sorted(x for x in REMOTE.rglob('*') if x.is_file() and x.name!='packet-manifest.json'):
  rows.append({'path':p.relative_to(REMOTE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'mode':oct(p.stat().st_mode&0o777)})
 canon=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
 packet=hashlib.sha256(canon).hexdigest()
 bundle=sha(REMOTE/'candidate-extension/dist/extension.js')
 value={'schema':'sepalith.run06.notebook-editor-packet.v1','packet_sha256':packet,'candidate_source_manifest_sha256':'e2bad4d79dff659067ff78105cd1bb0d696db7fc976fe5e4d71e369de798faef','candidate_bundle_sha256':bundle,'files':rows}
 (REMOTE/'packet-manifest.json').write_text(json.dumps(value,indent=2)+'\n')
 return value
if __name__=='__main__':
 v=closure(); out=ROOT/'notebook-editor-packet.tar'
 with tarfile.open(out,'w') as t:
  t.add(REMOTE,arcname='packet',recursive=True)
 print(json.dumps({'status':'prepared_not_launched','packet_sha256':v['packet_sha256'],'bundle_sha256':v['candidate_bundle_sha256'],'tar':str(out),'tar_bytes':out.stat().st_size,'tar_sha256':sha(out)},indent=2))
