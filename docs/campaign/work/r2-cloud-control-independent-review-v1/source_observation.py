import hashlib,json,os
from pathlib import Path
H=Path(__file__).resolve().parent;W=H.parent/'r2-cloud-control-entry-v1';V=H.parent/'r2-task-trainer-review-v2'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((W/'trainer-source-manifest.json').read_text());rows=[]
for r in m['files']:
 q=W/r['path'];v=V/r['path'];a=sha(q);b=sha(v);assert a==b==r['sha256'];rows.append({'path':r['path'],'sha256':a,'v2_match':True})
assert (W/'trainer-source-manifest.json').read_bytes()==(V/'source-manifest.json').read_bytes()
core=[{'path':str(W/n),'sha256':sha(W/n)} for n in ('cloud_entry.py','cloud_launch.py','cloud_train.py','runtime_setup.py','artifact_upload.py','job-template.yaml','root-bound-entry.sh','requirements.txt','package-pins.json')]
x={'trainer_files':rows,'trainer_manifest_sha256':sha(W/'trainer-source-manifest.json'),'core_in_progress':core}
(H/'source-observation-initial.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'trainer_files_matched':len(rows),'core_files_pinned':len(core)}))
