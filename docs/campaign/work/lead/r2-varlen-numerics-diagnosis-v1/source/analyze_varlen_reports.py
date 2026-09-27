#!/usr/bin/env python3
"""Reproducible CPU audit of the fixed actual-model v3/v4 varlen reports."""
import argparse, hashlib, json, math
from pathlib import Path

EXPECTED={
 "v3":"a933cab92bbe18a18a9f2a3de2f1e2f1a225fc4df252fb24804b6ad0f9c5e517",
 "v4":"0128afbd3fe98fdb0d59624ace04ebf19c56412438208b450a68eb698a37463e",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
 if not v: raise ValueError(m)
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def analyze(v3p,v4p):
 paths={"v3":Path(v3p),"v4":Path(v4p)}; data={}
 for key,p in paths.items():
  req(p.is_file() and sha(p)==EXPECTED[key],f"{key} report differs")
  d=json.loads(p.read_text()); req(d.get("status")=="measurement_complete_root_parity_decision_required",f"{key} incomplete")
  req(d.get("no_optimizer_step") is True and d["result"]["backend"]=="xformers",f"{key} scope differs")
  req(d["contract"]=={"loss_denominator":9681,"physical_groups":1,"rows":16,"tokens":9700},f"{key} contract differs")
  r=d["result"]; req(r["cross_document_isolation"]["max_abs"]==0.0,f"{key} isolation differs")
  vals=[r["standalone_loss"],r["packed_loss"],r["loss_abs_delta"],r["hidden"]["max_member_relative_l2"],r["sampled_logits"]["max_member_relative_l2"]]
  req(all(finite(x) for x in vals),f"{key} nonfinite")
  grads={n:x["relative_l2"] for n,x in r["gradient_samples"].items()}; req(len(grads)==21 and all(finite(x) for x in grads.values()),f"{key} gradients differ")
  data[key]={"loss":vals[:3],"hidden_relative_l2_max":vals[3],"logit_relative_l2_max":vals[4],"gradients":grads}
 req(set(data["v3"]["gradients"])==set(data["v4"]["gradients"]),"gradient sample names differ")
 gs=sorted(data["v4"]["gradients"].values())
 # v4 gives every standalone call one-member packed metadata. Persistence of
 # the discrepancy rules out the mere presence/absence of that argument.
 stable={"hidden":abs(data["v4"]["hidden_relative_l2_max"]-data["v3"]["hidden_relative_l2_max"]),
         "logit":abs(data["v4"]["logit_relative_l2_max"]-data["v3"]["logit_relative_l2_max"]),
         "loss_delta":abs(data["v4"]["loss"][2]-data["v3"]["loss"][2])}
 return {"schema":"sepalith.sft11.varlen-numerics-audit.v1","reports":{k:{"path":str(paths[k]),"sha256":EXPECTED[k]} for k in paths},
   "observed":{"loss_relative_delta":data["v4"]["loss"][2]/abs(data["v4"]["loss"][0]),
    "hidden_relative_l2_max":data["v4"]["hidden_relative_l2_max"],"logit_relative_l2_max":data["v4"]["logit_relative_l2_max"],
    "gradient_relative_l2":{"min":gs[0],"median":gs[len(gs)//2],"mean":sum(gs)/len(gs),"max":gs[-1]},
    "v3_v4_stability":stable,"cross_document_isolation_max_abs":0.0},
   "inference":{"ruled_out":"packed_seq_lengths argument presence alone","not_ruled_out":["BF16 projection/GEMM shape-dependent rounding","xformers one-block versus block-diagonal attention kernel numerics","shape-dependent RMSNorm/MLP kernels"],
    "unlikely_primary":["cross-document leakage","loss denominator mismatch","RoPE position mismatch"]},
   "decision":"production varlen equivalence not established"}

def main():
 p=argparse.ArgumentParser();p.add_argument('--v3',type=Path,required=True);p.add_argument('--v4',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(); result=analyze(a.v3,a.v4); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
