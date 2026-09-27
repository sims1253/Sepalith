"""Pin this candidate source tree; does not create runner state or a Git checkout."""
import difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
base=json.loads((H/'base-source-manifest.json').read_text())
root=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-cpt-v1/snapshots')/base['id']/'source'
entries=[];diff=[];changed=[];unchanged=[]
for p in sorted((H/'source').rglob('*.py')):
 rel=str(p.relative_to(H/'source'));sha=hashlib.sha256(p.read_bytes()).hexdigest();entries.append({'path':rel,'sha256':sha,'mode':292})
 old=root/rel
 if old.exists() and old.read_bytes()==p.read_bytes():unchanged.append(rel)
 else:
  changed.append(rel);diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel))
id=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
manifest={'schema_version':1,'id':id,'files':entries,'base_source_identity':base['id'],'status':'candidate_bytes_frozen; not yet captured/admitted by root runner','changed_or_added':changed,'unchanged':unchanged,'scope':'14 explicit Python source files; inherited installed environment is not newly proven by this packet'}
(H/'source-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(H/'source.patch').write_text(''.join(diff))
print(json.dumps({'candidate_source_id':id,'files':len(entries),'changed_or_added':changed,'unchanged':len(unchanged)}))
