#!/usr/bin/env python3
"""Convert reviewed bindings to executable bindings after a root admission."""
import argparse,hashlib,json,os
from pathlib import Path
PACKET=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--admission',type=Path,required=True);a=ap.parse_args()
ad=json.loads(a.admission.read_text());review=PACKET/'payload-review.json'
if ad.get('schema')!='sepalith.sft11.varlen330.evaluation-root-admission.v1' or ad.get('status')!='admitted' or ad.get('launch_authorized') is not True or ad.get('payload_review_sha256')!=sha(review):raise ValueError('root admission differs')
for arm in ('ordinary_reference','varlen_candidate'):
 src=PACKET/arm/'binding.review.json';value=json.loads(src.read_text())
 if ad.get('binding_review_sha256',{}).get(arm)!=sha(src) or value.get('status')!='prepared_requires_root_admission' or value.get('canary_arm')!=arm:raise ValueError('arm binding admission differs:'+arm)
 value['status']='admitted';value['root_admission_sha256']=sha(a.admission);dst=PACKET/arm/'binding.json'
 with dst.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 command=['/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-B','/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-long-eval-root-v2/evaluate.py','--binding',str(dst),'--output',f'/mnt/e/sepalith/campaign-20260915/evaluations/SFT11-varlen330-{arm}-matched-v1']
 with (PACKET/arm/'command.json').open('x') as f:json.dump(command,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'status':'admitted_bindings_prepared_no_launch','root_admission_sha256':sha(a.admission)}))
