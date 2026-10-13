#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from prepare_refresh import pinned_file
P=Path(__file__).resolve().parent
def main()->None:
 b=json.loads((P/"refresh-template.json").read_text())
 assert b["target"] is None and b["launch_authorized"] is False
 assert b["calibration"]["split"]==b["quality_panel"]["split"]=="train"
 assert not b["calibration"]["contains_dev_or_final"] and not b["quality_panel"]["contains_dev_or_final"]
 assert b["calibration"]["rows"]==84 and b["quality_panel"]["rows"]==40
 for name,item in b["tools"].items():pinned_file(item,name)
 pinned_file({"path":b["calibration"]["text_path"],"sha256":b["calibration"]["text_sha256"]},"calibration")
 pinned_file({"path":b["calibration"]["manifest_path"],"sha256":b["calibration"]["manifest_sha256"]},"calibration manifest")
 pinned_file({"path":b["quality_panel"]["path"],"sha256":b["quality_panel"]["sha256"]},"panel")
 pinned_file({"path":b["quality_panel"]["manifest_path"],"sha256":b["quality_panel"]["manifest_sha256"]},"panel manifest")
 for key in ("released_dspark","existing_trained_dspark"):
  d=b["drafts"][key];pinned_file(d,key);pinned_file({"path":d["header_receipt_path"],"sha256":d["header_receipt_sha256"]},key+" receipt")
 common=b["runtime"]["common_server_argv"]
 for flag in ("--temp","--seed","-c","-b","-ub","-ngl","-ngld","-lv"):assert flag in common
 assert b["runtime"]["native_mtp"]["status"]=="not_assumed_or_prepared"
 assert b["drafts"]["existing_trained_dspark"]["target_relation"]=="cross_target_trial_only_until_retrained"
 print("immutable template pins PASS")
if __name__=="__main__":main()
