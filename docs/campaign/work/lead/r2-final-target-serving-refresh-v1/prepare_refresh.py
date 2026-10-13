#!/usr/bin/env python3
"""Validate a root-filled serving refresh binding and emit exact stage commands."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

HEX=set("0123456789abcdef")
def sha(path: Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
 return h.hexdigest()
def require(ok: bool, msg: str)->None:
 if not ok: raise ValueError(msg)
def pinned_file(item: dict[str,Any], name: str)->Path:
 p=Path(item["path"]); require(p.is_file(),f"missing {name}: {p}")
 require(sha(p)==item["sha256"],f"{name} hash mismatch")
 return p
def artifact(item: Any,name:str)->Path:
 require(isinstance(item,dict),f"{name} is not bound")
 require(set(item)>={"path","sha256"},f"{name} lacks path/hash")
 require(len(item["sha256"])==64 and set(item["sha256"])<=HEX,f"{name} bad hash")
 return pinned_file(item,name)

def validate_and_commands(b:dict[str,Any], stage:str, check_payloads:bool=True)->list[list[str]]:
 require(b.get("schema_version")=="sepalith.r2.final-target-serving-refresh.v1","schema mismatch")
 require(b.get("launch_authorized") is True,"root launch authorization absent")
 target=b.get("target"); require(isinstance(target,dict),"target remains unselected")
 for k in ("identity","hf_dir","manifest_path","manifest_sha256","tokenizer_json_sha256","config_sha256","weights_manifest_path","weights_manifest_sha256","selection_receipt_path","selection_receipt_sha256"):
  require(target.get(k),f"target.{k} absent")
 require(target["tokenizer_json_sha256"]==b["tokenizer_contract"]["tokenizer_json_sha256"],"target tokenizer mismatch")
 require(target.get("checkpoint_kind")=="full_weights","selected target is not a full-weight checkpoint/merge")
 pinned_file({"path":target["manifest_path"],"sha256":target["manifest_sha256"]},"target manifest")
 hf=Path(target["hf_dir"]);require(hf.is_dir(),"target HF directory absent")
 pinned_file({"path":str(hf/"tokenizer.json"),"sha256":target["tokenizer_json_sha256"]},"target tokenizer.json")
 pinned_file({"path":str(hf/"config.json"),"sha256":target["config_sha256"]},"target config.json")
 pinned_file({"path":target["weights_manifest_path"],"sha256":target["weights_manifest_sha256"]},"target weights manifest")
 pinned_file({"path":target["selection_receipt_path"],"sha256":target["selection_receipt_sha256"]},"target selection receipt")
 cal=b["calibration"]; require(cal["split"]=="train" and cal["contains_dev_or_final"] is False,"calibration is not TRAIN-only")
 panel=b["quality_panel"]; require(panel["split"]=="train" and panel["contains_dev_or_final"] is False,"quality panel is not TRAIN-only")
 for key in ("text_path","manifest_path"):
  pinned_file({"path":cal[key],"sha256":cal[key.replace("path","sha256")]},"calibration "+key)
 for key in ("path","manifest_path"):
  pinned_file({"path":panel[key],"sha256":panel["sha256" if key=="path" else "manifest_sha256"]},"panel "+key)
 tools=b["tools"]
 for name,item in tools.items(): pinned_file(item,name)
 out=b["outputs"]; root=Path(out["root"])
 f16=Path(out["f16"]["path"] if isinstance(out.get("f16"),dict) else root/"model-F16.gguf")
 im=Path(out["imatrix"]["path"] if isinstance(out.get("imatrix"),dict) else root/"imatrix.gguf")
 qpaths={q:Path(out[q]["path"] if isinstance(out.get(q),dict) else root/f"model-{t}.gguf") for q,t in (("q8","Q8_0"),("q4_k_m","Q4_K_M-imatrix"),("iq3_m","IQ3_M-imatrix"))}
 py="/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python"
 if stage=="export":
  require(not f16.exists(),"F16 output already exists")
  return [[py,"-B",tools["convert_hf_to_gguf"]["path"],target["hf_dir"],"--outfile",str(f16),"--outtype","f16"]]
 if stage in {"imatrix","quantize","probe"}:
  if check_payloads: artifact(out.get("f16"),"F16 artifact")
 if stage=="imatrix":
  require(not im.exists(),"imatrix output already exists")
  return [["env","CUDA_VISIBLE_DEVICES=0","LD_LIBRARY_PATH="+str(Path(tools["cuda_backend"]["path"]).parent),"GGML_BACKEND_PATH="+tools["cuda_backend"]["path"],"GGML_CUDA_GRAPH_OPT=0",tools["imatrix"]["path"],"-m",str(f16),"-f",cal["text_path"],"-o",str(im),"-c","2048","-b","256","-ub","256","-t","6","-ngl","99","--chunks","64","--no-ppl","--process-output","--output-frequency","4"]]
 if stage=="quantize":
  if check_payloads: artifact(out.get("imatrix"),"imatrix artifact")
  require(all(not p.exists() for p in qpaths.values()),"one or more quant outputs already exist")
  base=[tools["quantize"]["path"],"--imatrix",str(im),str(f16)]
  return [base+[str(qpaths["q8"]),"Q8_0","4"],base+[str(qpaths["q4_k_m"]),"Q4_K_M","4"],base+[str(qpaths["iq3_m"]),"IQ3_M","4"]]
 if stage=="probe":
  arts={"f16":out.get("f16"),"q8":out.get("q8"),"q4_k_m":out.get("q4_k_m"),"iq3_m":out.get("iq3_m")}
  if check_payloads:
   for name,item in arts.items(): artifact(item,name)
  cmds=[]
  for name,item in arts.items():
   cmds.append([py,"-B",str(Path(__file__).with_name("probe_bound.py")),"--binding","$BINDING","--artifact-key",name,"--arm","quant-"+name,"--url",f"$URL_{name.upper()}","--out",f"$RUN/quant-{name}.json"])
  return cmds
 if stage=="serving":
  target_q8=artifact(out.get("q8"),"q8") if check_payloads else Path(out["q8"]["path"])
  cmds=[]; common=tools["server"]["path"],"-m",str(target_q8),*b["runtime"]["common_server_argv"]
  cmds.append(list(common))
  cmds.append([*common,*b["runtime"]["model_free_argv"]])
  for key in ("released_dspark","existing_trained_dspark"):
   d=b["drafts"][key]
   if d.get("available"):
    if check_payloads:
     artifact(d,key); artifact({"path":d["header_receipt_path"],"sha256":d["header_receipt_sha256"]},key+" header receipt")
    if key=="existing_trained_dspark": require(d["target_relation"]=="cross_target_trial_only_until_retrained","old trained draft mislabeled target-matched")
    cmds.append([*common,"-md",d["path"],*b["runtime"]["dspark_argv"]])
  return cmds
 raise ValueError("unknown stage")

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--binding",required=True); p.add_argument("--stage",choices=("export","imatrix","quantize","probe","serving"),required=True); p.add_argument("--out",required=True)
 a=p.parse_args(); b=json.loads(Path(a.binding).read_text()); commands=validate_and_commands(b,a.stage)
 result={"schema_version":"sepalith.r2.final-target-serving-refresh.commands.v1","stage":a.stage,"target_identity":b["target"]["identity"],"commands":commands,"execution_authorized_by_this_receipt":False}
 Path(a.out).write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"status":"pass","commands":len(commands)})); return 0
if __name__=="__main__": raise SystemExit(main())
