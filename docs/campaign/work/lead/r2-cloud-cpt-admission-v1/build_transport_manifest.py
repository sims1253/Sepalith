#!/usr/bin/env python3
"""Build the immutable private-input map from the frozen CPT recipe."""
import hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
RECIPE=PLAN/'docs/campaign/work/lead/r2-cpt-remaining-v1/recipe.json'
PRODUCER=PLAN/'docs/campaign/receipts/SFT-11-CPT-remaining-preparation.json'
RECIPE_SHA='254de1150b0fcc8e8566178905542bdadc1255610a2dc3cb8a9168dc7017d538'
PREFIX='r2-cpt-inputs/'+RECIPE_SHA

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def rel(original,index,model):
 p=Path(original)
 if str(p).startswith(model+'/'): return 'model/'+p.name
 if '/r2-cpt-remaining-v1/source/' in str(p): return 'source/'+str(p).split('/source/',1)[1]
 if 'CPT-remaining-v1' in str(p): return 'data/'+p.name
 return f'lineage/{index:02d}-{p.name}'

def main():
 if sha(RECIPE)!=RECIPE_SHA: raise ValueError('frozen recipe differs')
 recipe=json.loads(RECIPE.read_text());rows=[]
 for i,item in enumerate(recipe['inputs']):
  p=Path(item['path']);actual=sha(p)
  if actual!=item['sha256']:raise ValueError(f'input differs: {p}')
  q=rel(str(p),i,recipe['model_path'])
  rows.append({'role':'recipe_input','original_path':str(p),'relative_path':q,
               'remote_path':PREFIX+'/files/'+q,'sha256':actual,'bytes':p.stat().st_size})
 for role,p,q,expected in [
  ('recipe',RECIPE,'recipe.json',RECIPE_SHA),
  ('producer_receipt',PRODUCER,'producer-receipt.json','ce34e52357802e4e95e632a6eb7115ae70f89e89ba8a64f75ffe9c6ad0973f9f')]:
  if sha(p)!=expected:raise ValueError(role+' differs')
  rows.append({'role':role,'original_path':str(p),'relative_path':q,
               'remote_path':PREFIX+'/files/'+q,'sha256':expected,'bytes':p.stat().st_size})
 out={'schema':'sepalith.cloud-cpt.transport.v1','recipe_sha256':RECIPE_SHA,
      'source_identity':'7e4e80059416b3783c783a532d95b40af6067076e7a0a23d64e3b571f478c9df',
      'repo_id':'scholzmx/sepalith-lora','prefix':PREFIX,'files':rows,
      'files_count':len(rows),'bytes':sum(x['bytes'] for x in rows),
      'credential_policy':'credentials are never manifest fields'}
 (HERE/'transport-manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'files':len(rows),'bytes':out['bytes'],'prefix':PREFIX}))
if __name__=='__main__':main()
