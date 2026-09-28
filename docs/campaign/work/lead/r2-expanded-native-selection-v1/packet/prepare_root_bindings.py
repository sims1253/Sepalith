"""Emit root-editable export/admission templates from completed metadata only."""
import argparse,hashlib,json
from pathlib import Path
from selection_contract import export_commands
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-manifest',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--export-dir',type=Path,required=True);a.add_argument('--host-guard-output',type=Path,required=True);x=a.parse_args()
 assert x.output_dir.is_absolute() and not x.output_dir.exists();m=json.loads(x.parent_manifest.read_text());x.output_dir.mkdir()
 pins=json.loads((H/'conversion-source-pins.json').read_text())
 assert x.host_guard_output.is_absolute() and not x.host_guard_output.exists()
 spec={'status':'root_admitted','parent_manifest':{'path':str(x.parent_manifest),'sha256':sha(x.parent_manifest)},'output':str(x.export_dir),'host_guard_preflight':str(x.host_guard_output/'preflight.json'),'converter_source_files':pins['converter_source_files'],'quantizer_files':pins['quantizer_files'],'commands':export_commands(m['merged_model_path'],str(x.export_dir))}
 (x.output_dir/'export-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 approval=json.loads((H/'native-approval.template.json').read_text())
 (x.output_dir/'native-approval.json').write_text(json.dumps(approval,indent=2)+'\n')
if __name__=='__main__':main()
