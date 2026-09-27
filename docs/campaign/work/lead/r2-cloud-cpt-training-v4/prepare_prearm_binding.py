#!/usr/bin/env python3
"""Build one fresh pre-arm binding from a reviewed full-training admission."""
import argparse,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--absolute-deadline-utc',required=True);p.add_argument('--watchdog-armed-at-utc',required=True);p.add_argument('--root-admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not re.fullmatch('[0-9a-f]{32}',a.run_id) or a.output.exists():raise ValueError('fresh UUID32/output required')
 if a.root_admission.resolve()!=H/'root-recipe-admission.json' or a.output.resolve()!=H/'binding.prearm.json':raise ValueError('exact root admission/prearm paths required')
 admission=json.loads(a.root_admission.read_text());status=str(admission.get('status','')).lower()
 if 'diagnostic' in status or not (admission.get('admitted') is True or any(x in status for x in ('accept','admit','pass'))):raise ValueError('full training root admission required')
 template=json.loads((H/'binding.template.json').read_text());template.update(admitted=True,run_id=a.run_id,artifact_prefix='r2-cpt/'+a.run_id,root_recipe_admission_sha256=sha(a.root_admission),watchdog_armed_at_utc=a.watchdog_armed_at_utc,watchdog_armed_receipt_sha256='0'*64,payload_manifest_sha256=sha(H/'payload-manifest.json'),absolute_deadline_utc=a.absolute_deadline_utc)
 if template['input_revision']!='1e4df9c9db3cee1a7df5bbcc7caa62920d2ab2db' or template['recipe_sha256'] not in json.dumps(admission,sort_keys=True):raise ValueError('frozen input/admission binding differs')
 a.output.write_text(json.dumps(template,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'prearm_ready_not_submitted','path':str(a.output),'sha256':sha(a.output)}))
if __name__=='__main__':main()
