#!/usr/bin/env python3
"""Freeze the small cloud packet source closure after review."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 files=[]
 for p in sorted((HERE/'payload').iterdir()):
  if p.is_file():files.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
 for name in ('transport-manifest.json','binding.template.json','job-template.proposed.yaml'):
  p=HERE/name;files.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
 out={'schema':'sepalith.cloud-cpt.payload-manifest.v1','candidate_only':True,'files':files,'frozen_recipe_sha256':'254de1150b0fcc8e8566178905542bdadc1255610a2dc3cb8a9168dc7017d538','transport_manifest_sha256':'a9afe5d58034b56e36abf83a0c1d169ec578b79cc215daa32603ccd210e7045b'}
 (HERE/'payload-manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'files':len(files)}))
if __name__=='__main__':main()
