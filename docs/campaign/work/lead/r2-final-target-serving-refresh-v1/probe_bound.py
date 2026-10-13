#!/usr/bin/env python3
"""Hash an admitted target/draft, delegate to the reviewed TRAIN probe, bind result."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
from prepare_refresh import artifact,pinned_file,require
def digest_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--binding",required=True);p.add_argument("--artifact-key",choices=("f16","q8","q4_k_m","iq3_m"),required=True);p.add_argument("--draft-key",choices=("released_dspark","existing_trained_dspark"));p.add_argument("--arm",required=True);p.add_argument("--url",required=True);p.add_argument("--out",required=True);a=p.parse_args()
 bp=Path(a.binding); raw=bp.read_bytes(); b=json.loads(raw); require(b.get("launch_authorized") is True,"root authorization absent")
 model=artifact(b["outputs"].get(a.artifact_key),a.artifact_key)
 probe=pinned_file(b["tools"]["panel_probe"],"panel_probe"); panel=b["quality_panel"]
 bound={"binding_sha256":digest_bytes(raw),"target_identity":b["target"]["identity"],"artifact_key":a.artifact_key,"artifact_path":str(model),"artifact_sha256":b["outputs"][a.artifact_key]["sha256"],"tokenizer_json_sha256":b["target"]["tokenizer_json_sha256"],"draft":None}
 if a.draft_key:
  d=b["drafts"][a.draft_key];require(d.get("available") is True,"draft unavailable");dp=artifact(d,a.draft_key);artifact({"path":d["header_receipt_path"],"sha256":d["header_receipt_sha256"]},a.draft_key+" header")
  bound["draft"]={"key":a.draft_key,"path":str(dp),"sha256":d["sha256"],"target_relation":d["target_relation"]}
 with tempfile.TemporaryDirectory(prefix="sepalith-refresh-probe-") as td:
  rawout=Path(td)/"probe.json"
  cmd=["/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python","-B",str(probe),"--url",a.url,"--panel",panel["path"],"--manifest",panel["manifest_path"],"--context",str(panel["context"]),"--cap",str(panel["completion_cap"]),"--reps",str(panel["repetitions"]),"--out",str(rawout)]
  proc=subprocess.run(cmd,check=False)
  result=json.loads(rawout.read_text()); result["arm"]=a.arm;result["artifact_binding"]=bound
  Path(a.out).write_text(json.dumps(result,indent=2)+"\n")
 return 0 if proc.returncode in (0,1) else proc.returncode
if __name__=="__main__":raise SystemExit(main())
