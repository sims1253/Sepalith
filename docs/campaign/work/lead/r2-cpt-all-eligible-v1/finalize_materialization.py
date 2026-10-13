#!/usr/bin/env python3
"""Verify and concatenate a complete per-group DAT10 materialization."""
import collections,hashlib,json,os,shutil
from pathlib import Path
E=Path('/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1')
ORDER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/r2-cpt-long-stage-review-v1/next-global-package-order.json')
START=775;EXPECTED=8092
RESOLUTIONS=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-all-eligible-v1/repair-resolutions.json')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 temp=path.with_name(path.name+'.tmp')
 with temp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 temp.replace(path)

def main():
 order=json.loads(ORDER.read_text())['packages'][START:];assert len(order)==EXPECTED
 receipts=[];counts=collections.Counter();repair_records={}
 for offset,entry in enumerate(order):
  index=START+offset;folder=E/'groups'/f'{index:06d}-{entry["group_id"]}';p=folder/'receipt.json'
  if not p.is_file():raise ValueError(f'missing group receipt: {index}')
  r=json.loads(p.read_text())
  if (r['seeded_index'],r['group_id'],r['package'])!=(index,entry['group_id'],entry['name']):raise ValueError(f'group identity mismatch: {index}')
  for name,record in r['artifacts'].items():
   path=folder/name
   if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError(f'group artifact differs: {path}')
  for line_number,line in enumerate((folder/'repair-queue.jsonl').read_text().splitlines(),1):
   key=(index,line_number,hashlib.sha256((line+'\n').encode()).hexdigest(),sha(p));repair_records[key]=json.loads(line)
  receipts.append((folder,r));counts.update(r['counts'])
 resolutions=json.loads(RESOLUTIONS.read_text());resolved=set()
 for item in resolutions['items']:
  key=(item['seeded_index'],item['line'],item['repair_record_sha256'],item['group_receipt_sha256'])
  if key not in repair_records:raise ValueError(f'repair resolution does not bind an exact queued record: {item}')
  resolved.add(key)
 unresolved=set(repair_records)-resolved
 if unresolved:
  blocker={'schema':'sepalith.cpt.all-eligible-finalization.v1','status':'blocked_on_explicit_repairs','groups_verified':len(receipts),'repair_items':len(repair_records),'repairs_resolved':len(resolved),'repairs_unresolved':len(unresolved),'counts':dict(counts)}
  write(E/'finalization-blocker.json',blocker);print(json.dumps(blocker));raise SystemExit(2)
 names=('cpt_train.jsonl','documents.jsonl','source-inventory.jsonl','exclusions.jsonl','repair-queue.jsonl')
 artifacts={}
 for name in names:
  partial=E/(name+'.partial')
  with partial.open('wb') as out:
   for folder,_ in receipts:
    with (folder/name).open('rb') as source:shutil.copyfileobj(source,out,4<<20)
   out.flush();os.fsync(out.fileno())
  target=E/name;partial.replace(target);artifacts[name]={'sha256':sha(target),'bytes':target.stat().st_size}
 manifest={'schema':'sepalith.cpt.all-eligible-materialization.v1','status':'complete_verified','start_seeded_index':START,'groups':EXPECTED,'counts':dict(counts),'repairs_recorded':len(repair_records),'repairs_resolved':len(resolved),'repairs_pending':0,
           'order_sha256':sha(ORDER),'source_manifest_sha256':sha(E/'run-manifest.json'),'artifacts':artifacts,
           'coverage':'Every supported eligible regular R payload from the 8,092-group frozen order slice is present exactly once after frozen provenance/heldout/exact-dedup gates; no package/group/file/token/time cap.'}
 write(E/'manifest.json',manifest);print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
