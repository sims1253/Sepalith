#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,tempfile
from pathlib import Path
import analyze_refresh as ar
import prepare_refresh as pr
def h(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->None:
 with tempfile.TemporaryDirectory() as td:
  d=Path(td); hf=d/"hf";hf.mkdir(); target=d/"target.json";target.write_text("{}\n");(hf/"tokenizer.json").write_text("tok");tokhash=h(hf/"tokenizer.json");(hf/"config.json").write_text("cfg");weights=d/"weights.json";weights.write_text("weights");selection=d/"selection.json";selection.write_text("selected")
  files={}
  for name in ("cal","calm","panel","panelm","convert","imatrix","quant","server","cuda","probe","f16","imat","q8","q4","iq3"):
   p=d/name;p.write_text(name);files[name]={"path":str(p),"sha256":h(p)}
  b={"schema_version":"sepalith.r2.final-target-serving-refresh.v1","launch_authorized":True,"target":{"identity":"T","hf_dir":str(hf),"manifest_path":str(target),"manifest_sha256":h(target),"tokenizer_json_sha256":tokhash,"config_sha256":h(hf/"config.json"),"weights_manifest_path":str(weights),"weights_manifest_sha256":h(weights),"selection_receipt_path":str(selection),"selection_receipt_sha256":h(selection),"checkpoint_kind":"full_weights"},"tokenizer_contract":{"tokenizer_json_sha256":tokhash},"calibration":{"split":"train","contains_dev_or_final":False,"text_path":files["cal"]["path"],"text_sha256":files["cal"]["sha256"],"manifest_path":files["calm"]["path"],"manifest_sha256":files["calm"]["sha256"]},"quality_panel":{"split":"train","contains_dev_or_final":False,"path":files["panel"]["path"],"sha256":files["panel"]["sha256"],"manifest_path":files["panelm"]["path"],"manifest_sha256":files["panelm"]["sha256"],"context":4096,"completion_cap":192,"repetitions":1},"tools":{"convert_hf_to_gguf":files["convert"],"imatrix":files["imatrix"],"quantize":files["quant"],"server":files["server"],"cuda_backend":files["cuda"],"panel_probe":files["probe"]},"outputs":{"root":str(d/"out"),"f16":files["f16"],"imatrix":files["imat"],"q8":files["q8"],"q4_k_m":files["q4"],"iq3_m":files["iq3"]},"drafts":{"released_dspark":{"available":False},"existing_trained_dspark":{"available":False}},"runtime":{"common_server_argv":["--host","127.0.0.1","--port","$PORT","--temp","0","--seed","0","-c","4096"],"model_free_argv":["--spec-type","ngram-mod"],"dspark_argv":["--spec-type","draft-dspark"]}}
  assert len(pr.validate_and_commands(b,"probe"))==4
  serving=pr.validate_and_commands(b,"serving");assert len(serving)==2 and "--host" in serving[0] and "--spec-type" in serving[1]
  staged=json.loads(json.dumps(b));staged["outputs"]={"root":str(d/"fresh"),"f16":{"path":str(d/"fresh-f16"),"sha256":"1"*64},"imatrix":{"path":str(d/"fresh-im"),"sha256":"2"*64},"q8":{"path":str(d/"fresh-q8"),"sha256":"3"*64},"q4_k_m":{"path":str(d/"fresh-q4"),"sha256":"4"*64},"iq3_m":{"path":str(d/"fresh-iq3"),"sha256":"5"*64}}
  export=pr.validate_and_commands(staged,"export",check_payloads=False);assert "--outtype" in export[0] and "f16" in export[0]
  im=pr.validate_and_commands(staged,"imatrix",check_payloads=False)[0];assert "--chunks" in im and "64" in im and "--process-output" in im
  qs=pr.validate_and_commands(staged,"quantize",check_payloads=False);assert [x[-2] for x in qs]==["Q8_0","Q4_K_M","IQ3_M"] and all("--imatrix" in x for x in qs)
  bad=json.loads(json.dumps(b));bad["target"]["checkpoint_kind"]="adapter" 
  try:pr.validate_and_commands(bad,"probe");raise AssertionError("adapter accepted")
  except ValueError as e:assert "full-weight" in str(e)
  bad=json.loads(json.dumps(b));bad["quality_panel"]["split"]="dev"
  try:pr.validate_and_commands(bad,"probe");raise AssertionError("DEV panel accepted")
  except ValueError:pass
  bad=json.loads(json.dumps(b));bad["outputs"]["q8"]["sha256"]="0"*64
  try:pr.validate_and_commands(bad,"serving");raise AssertionError("bad quant accepted")
  except ValueError as e:assert "hash mismatch" in str(e)
 rec=lambda wall,ids,raw,cap=192:{"row_id":"r","phase":"cold","rep":1,"protocol_status":"accepted","returned_token_ids":ids,"raw_text":raw,"combined_case_wall_ms":wall,"returned_token_count":len(ids),"cap":cap,"draft_n":10,"draft_n_accepted":4}
 bind={"target_identity":"T","tokenizer_json_sha256":tokhash}
 base={"artifact_binding":bind,"requests":[rec(10,[2,1],"x")]}; cand={"artifact_binding":bind,"requests":[rec(5,[2,1],"x")]}
 result=ar.compare(base,cand,"q8");assert result["status"]=="pass" and result["p50_speedup"]==2
 cand["requests"][0]["returned_token_ids"]=[3,1];assert ar.compare(base,cand,"q8")["status"]=="fail"
 cand={"artifact_binding":{"target_identity":"X","tokenizer_json_sha256":tokhash},"requests":[rec(5,[2,1],"x")]};assert ar.compare(base,cand,"q8")["status"]=="fail"
 caprec=rec(5,list(range(192)),"x"); assert ar.summary({"requests":[caprec]})["cap_hits"]==1
 probe_result={"artifact_binding":{"binding_sha256":"f"*64,"target_identity":"T","tokenizer_json_sha256":tokhash,"artifact_key":"q8","artifact_sha256":b["outputs"]["q8"]["sha256"],"draft":None},"requests":[]}
 ar.validate_bound_result(probe_result,b,"f"*64)
 probe_result["artifact_binding"]["artifact_sha256"]="0"*64
 try:ar.validate_bound_result(probe_result,b,"f"*64);raise AssertionError("analyzer accepted wrong quant binding")
 except AssertionError as e:assert "quant hash mismatch" in str(e)
 print("refresh contract tests PASS")
if __name__=="__main__":main()
