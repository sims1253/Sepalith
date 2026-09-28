#!/usr/bin/env python3
"""Compare quant/speculative TRAIN probes with exact greedy parity and cap reporting."""
from __future__ import annotations
import argparse,json,math
import hashlib
from pathlib import Path
from statistics import median
from typing import Any
def load(p:str)->dict[str,Any]:
 v=json.loads(Path(p).read_text()); assert isinstance(v.get("requests"),list); return v
def binding_hash(path:str)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_bound_result(v:dict[str,Any],b:dict[str,Any],bsha:str)->None:
 x=v.get("artifact_binding"); assert isinstance(x,dict),"missing artifact binding"
 assert x.get("binding_sha256")==bsha,"result binding hash mismatch"
 assert x.get("target_identity")==b["target"]["identity"],"result target mismatch"
 assert x.get("tokenizer_json_sha256")==b["target"]["tokenizer_json_sha256"],"result tokenizer mismatch"
 key=x.get("artifact_key"); assert key in ("f16","q8","q4_k_m","iq3_m")
 assert x.get("artifact_sha256")==b["outputs"][key]["sha256"],"result quant hash mismatch"
 if x.get("draft") is not None:
  d=x["draft"]; expected=b["drafts"][d["key"]];assert d["sha256"]==expected["sha256"],"result draft hash mismatch"
def idx(v:dict[str,Any])->dict[tuple,dict]:
 d={}
 for r in v["requests"]:
  k=(r.get("row_id"),r.get("phase"),r.get("rep")); assert k not in d; d[k]=r
 return d
def pct(v:list[float],p:float)->float|None:
 if not v:return None
 s=sorted(v); x=(len(s)-1)*p; lo=int(x); hi=math.ceil(x); return round(s[lo]+(s[hi]-s[lo])*(x-lo),3)
def summary(v:dict[str,Any])->dict[str,Any]:
 rs=v["requests"]; walls=[float(r["combined_case_wall_ms"]) for r in rs if isinstance(r.get("combined_case_wall_ms"),(int,float))]
 cap_hits=sum(r.get("returned_token_count")==r.get("cap") for r in rs)
 draft=[(r.get("draft_n"),r.get("draft_n_accepted")) for r in rs]
 counters=bool(rs) and all(isinstance(a,int) and isinstance(b,int) and 0<=b<=a for a,b in draft)
 drafted=sum(a for a,_ in draft) if counters else None; accepted=sum(b for _,b in draft) if counters else None
 return {"requests":len(rs),"accepted_protocol":sum(r.get("protocol_status")=="accepted" for r in rs),"cap_hits":cap_hits,"latency_ms":{"p50":pct(walls,.5),"p95":pct(walls,.95)},"draft":{"status":"present" if counters else "missing","drafted":drafted,"accepted":accepted,"rate":None if not drafted else round(accepted/drafted,6)}}
def compare(base:dict[str,Any],cand:dict[str,Any],label:str)->dict[str,Any]:
 if not isinstance(base.get("artifact_binding"),dict) or not isinstance(cand.get("artifact_binding"),dict):
  return {"label":label,"status":"fail","greedy_parity":False,"mismatches":[{"reason":"missing_artifact_binding"}]}
 if base["artifact_binding"].get("target_identity")!=cand["artifact_binding"].get("target_identity") or base["artifact_binding"].get("tokenizer_json_sha256")!=cand["artifact_binding"].get("tokenizer_json_sha256"):
  return {"label":label,"status":"fail","greedy_parity":False,"mismatches":[{"reason":"target_or_tokenizer_binding_mismatch"}]}
 a,b=idx(base),idx(cand); keys=set(a)|set(b); mismatch=[]
 for k in sorted(keys,key=repr):
  x,y=a.get(k),b.get(k)
  if not x or not y: mismatch.append({"key":list(k),"reason":"missing_pair"})
  elif x.get("protocol_status")!="accepted" or y.get("protocol_status")!="accepted": mismatch.append({"key":list(k),"reason":"protocol"})
  elif x.get("returned_token_ids")!=y.get("returned_token_ids") or x.get("raw_text")!=y.get("raw_text"): mismatch.append({"key":list(k),"reason":"greedy_output_mismatch"})
 sx,sy=summary(base),summary(cand); speed=None
 if sx["latency_ms"]["p50"] and sy["latency_ms"]["p50"]: speed=round(sx["latency_ms"]["p50"]/sy["latency_ms"]["p50"],6)
 return {"label":label,"status":"pass" if not mismatch and len(a)==len(b)>0 else "fail","greedy_parity":not mismatch,"mismatches":mismatch,"baseline":sx,"candidate":sy,"p50_speedup":speed}
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--binding",required=True);p.add_argument("--baseline",required=True);p.add_argument("--candidate",action="append",default=[]);p.add_argument("--require-draft-counters",action="store_true");p.add_argument("--out",required=True);a=p.parse_args(); b=json.loads(Path(a.binding).read_text());bsha=binding_hash(a.binding);base=load(a.baseline);validate_bound_result(base,b,bsha);results=[]
 for spec in a.candidate:
  label,path=spec.split("=",1);candidate=load(path);validate_bound_result(candidate,b,bsha);item=compare(base,candidate,label)
  if a.require_draft_counters and item.get("candidate",{}).get("draft",{}).get("status")!="present": item["status"]="fail";item.setdefault("mismatches",[]).append({"reason":"missing_draft_counters"})
  results.append(item)
 out={"schema_version":"sepalith.r2.final-target-serving-refresh.analysis.v1","status":"pass" if results and all(x["status"]=="pass" for x in results) else "fail","comparisons":results,"promotion_decision":None}
 Path(a.out).write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":out["status"]}));return 0 if out["status"]=="pass" else 1
if __name__=="__main__":raise SystemExit(main())
