"""Emit root-editable export/admission templates from completed metadata only."""
import argparse,hashlib,json
from pathlib import Path
from selection_contract import check_task_recipe,export_commands
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-manifest',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--export-dir',type=Path,required=True);a.add_argument('--host-guard-output',type=Path,required=True);x=a.parse_args()
 assert x.output_dir.is_absolute() and not x.output_dir.exists();m=json.loads(x.parent_manifest.read_text())
 check_task_recipe(m['sft_identity'],m['recipe_sha256'])
 assert m['sft_checkpoint_manifest']['full'] is True and m['sft_checkpoint_manifest']['step']==1000 and m['source_cursor']==16000
 assert m['sft_checkpoint_manifest']['identity']==m['sft_identity']
 x.output_dir.mkdir()
 pins=json.loads((H/'conversion-source-pins.json').read_text())
 assert x.host_guard_output.is_absolute() and not x.host_guard_output.exists()
 spec={'status':'root_admitted','parent_manifest':{'path':str(x.parent_manifest),'sha256':sha(x.parent_manifest)},'output':str(x.export_dir),'host_guard_preflight':str(x.host_guard_output/'preflight.json'),'converter_source_files':pins['converter_source_files'],'quantizer_files':pins['quantizer_files'],'commands':export_commands(m['merged_model_path'],str(x.export_dir))}
 spec['host_guard']={'path': '/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/host-memory-guard-v4/cuda_host_guard.py', 'sha256': '25f85d1da5bfee8da085a209982617e39e176b89dda11febc851c248a75240e0', 'admission_free_mib': 12288, 'operating_free_mib': 6144}
 (x.output_dir/'export-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 approval=json.loads((H/'native-approval.template.json').read_text())
 (x.output_dir/'native-approval.json').write_text(json.dumps(approval,indent=2)+'\n')
if __name__=='__main__':main()
