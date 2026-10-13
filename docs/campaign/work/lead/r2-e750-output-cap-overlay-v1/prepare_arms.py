#!/usr/bin/env python3
import argparse,hashlib,json,os,shutil
from pathlib import Path
CAPS=(192,384,768)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--template',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args()
 if x.out.exists():raise SystemExit('fresh output required')
 x.out.mkdir(parents=True)
 for cap in CAPS:
  arm=x.out/f'cap-{cap}'; shutil.copytree(x.template,arm)
  p=arm/'native_evaluator/profile.json'; d=json.loads(p.read_text());d['output']=cap;d['model_profile']['maxOutputTokens']=cap
  p.write_text(json.dumps(d,indent=2)+'\n')
  # Derived closure/log files from the accepted run are not valid for this new source.
  for name in ('source-closure.prepared.json','client-origins.json','independent-origin-comparison.json','observation-terminal.json','policy-check-terminal.json','client-observation.log','policy-check.log'):
   q=arm/'native_controller'/name
   if q.exists():q.unlink()
  files=[]
  for q in sorted(y for y in arm.rglob('*') if y.is_file()):files.append({'path':str(q.relative_to(arm)),'bytes':q.stat().st_size,'sha256':sha(q)})
  manifest={'schema':'sepalith.run06.e750-cap-arm-source.v1','status':'prepared_requires_root_closure_and_admission','cap':cap,'context':4096,'case_deadline_seconds':5,'profile_sha256':sha(p),'files':files}
  (arm/'arm-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
