"""Enumerate only registry TRAIN package R files; do not read R payloads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib,json,os,time
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/h/sepalith/normalized')
SPLIT=Path('/mnt/e/sepalith/campaign-20260915/data-work/DAT-02-global-split-v2.json')
SPLIT_SHA='c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fields(text):
 result={};key=None
 for line in text.splitlines():
  if line[:1].isspace() and key:result[key]+=' '+line.strip()
  elif ':' in line:key,value=line.split(':',1);result[key]=value.strip()
 return result
def package(entry):
 name,group=entry;root=ROOT/name;rows=[];rejects=[]
 try:
  versions=[x for x in os.scandir(root) if x.is_dir(follow_symlinks=False)]
  if len(versions)!=1:return [],[{'package':name,'reason':'version_count_not_one','count':len(versions)}]
  version=versions[0].name;source=root/version/name;desc=source/'DESCRIPTION'
  if source.is_symlink() or desc.is_symlink():return [],[{'package':name,'reason':'symlink'}]
  if not desc.is_file() or desc.stat().st_size>128*1024:return [],[{'package':name,'reason':'description_missing_or_large'}]
  raw=desc.read_bytes();dcf=fields(raw.decode('utf8'));license=dcf.get('License','')
  if dcf.get('Package')!=name:return [],[{'package':name,'reason':'DESCRIPTION_package_mismatch'}]
  if not license or dcf.get('License_restricts_use','').lower()=='yes' or dcf.get('License_is_FOSS','').lower()=='no':return [],[{'package':name,'reason':'license_requires_review','license':license}]
  rdir=source/'R'
  if not rdir.is_dir() or rdir.is_symlink():return [],[{'package':name,'reason':'no_regular_R_directory'}]
  for current,dirs,files in os.walk(rdir,followlinks=False):
   dirs[:]=sorted(d for d in dirs if not (Path(current)/d).is_symlink())
   for filename in sorted(files):
    p=Path(current)/filename
    if p.suffix.lower()!='.r' or p.is_symlink() or not p.is_file():continue
    st=p.stat()
    if not 0<st.st_size<=4*1024*1024:rejects.append({'package':name,'path':str(p),'reason':'R_file_empty_or_over_4MiB','bytes':st.st_size});continue
    rows.append({'package':name,'version':version,'group_id':group['group_id'],'split':'train_group','path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev,'description_path':str(desc),'description_sha256':hashlib.sha256(raw).hexdigest(),'license':license,'raw_code_hashed':False})
  return rows,rejects
 except (OSError,UnicodeError) as e:return [],[{'package':name,'reason':'metadata_read_failure','detail':type(e).__name__}]
def main():
 if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 assert digest(SPLIT)==SPLIT_SHA
 split=json.loads(SPLIT.read_text());mapping={};nontrain_parent_hashes=set()
 for g in split['groups']:
  if g['split']!='train_group':nontrain_parent_hashes.update(x.removeprefix('parent:') for x in g['parent_tokens'])
  for token in g['identity_forms']:
   if token.startswith('pkg:'):
    key=token[4:].lower()
    if key in mapping:assert mapping[key]['group_id']==g['group_id']
    mapping[key]=g
 entries=[x for x in os.scandir(ROOT) if x.is_dir(follow_symlinks=False)];counts=Counter(mapping.get(x.name.lower(),{}).get('split','unmapped') for x in entries)
 train=sorted((x.name,mapping[x.name.lower()]) for x in entries if mapping.get(x.name.lower(),{}).get('split')=='train_group')
 (HERE/'known-nontrain-parent-hashes.json').write_text(json.dumps(sorted(nontrain_parent_hashes))+'\n')
 total_files=total_bytes=processed=0;accepted_packages=0;reasons=Counter();started=time.monotonic()
 with (HERE/'raw-train-file-candidates.jsonl').open('w') as out,(HERE/'raw-train-metadata-exclusions.jsonl').open('w') as excluded,ThreadPoolExecutor(max_workers=2) as pool:
  for rows,rejects in pool.map(package,train):
   processed+=1;total_files+=len(rows);total_bytes+=sum(x['bytes'] for x in rows);accepted_packages+=bool(rows)
   for row in rows:out.write(json.dumps(row,separators=(',',':'))+'\n')
   for r in rejects:excluded.write(json.dumps(r,separators=(',',':'))+'\n');reasons[r['reason']]+=1
   if processed%500==0:
    print(json.dumps({'processed_train_packages':processed,'R_files':total_files,'raw_bytes':total_bytes,'seconds':time.monotonic()-started}),flush=True)
 result={'status':'TRAIN_metadata_inventory_not_CPT_admission','normalized_root':str(ROOT),'split_id':split['split_id'],'split_sha256':SPLIT_SHA,'package_directory_counts':dict(counts),'mapped_train_packages':len(train),'packages_with_eligible_files':accepted_packages,'R_files':total_files,'raw_bytes':total_bytes,'known_nontrain_parent_hashes':len(nontrain_parent_hashes),'metadata_exclusion_reasons':dict(reasons),'seconds':time.monotonic()-started,'code_content_read':False,'usable_unique_bytes_and_tokens':'pending source hash/UTF8/tokenization pass','heldout_content_read':False}
 (HERE/'raw-train-inventory-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
