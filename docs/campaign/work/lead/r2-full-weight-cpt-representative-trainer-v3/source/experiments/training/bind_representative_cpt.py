#!/usr/bin/env python3
"""Bind admitted representative whole-document data and scientific choices."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
def pin(x,name):
 p=Path(x["path"]);req(p.is_file() and sha(p)==x["sha256"],name+" differs")
def validate_saved_precision(parent):
 precision=parent.get("saved_precision",{});req(type(precision.get("fp32_tensors")) is int and precision["fp32_tensors"]>=0,"parent FP32 tensor expectation missing");req(type(precision.get("fp32_elements")) is int and precision["fp32_elements"]>=0,"parent FP32 element expectation missing")
 req(isinstance(precision.get("identity_source"),str) and precision["identity_source"],"parent precision identity source missing")
 if precision["fp32_tensors"]==0:req(precision["fp32_elements"]==0,"zero FP32 tensor expectation requires zero FP32 elements")
 return precision
def write_new(path,value):
 path=Path(path);req(not path.exists(),"bound recipe output must be fresh");path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
 try:
  with os.fdopen(fd,"w") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def bind(template_path,admission_path,output):
 template_path,admission_path=Path(template_path),Path(admission_path);t=json.loads(template_path.read_text());a=json.loads(admission_path.read_text())
 req(t.get("schema")=="sepalith.sft11.full-weight-cpt-representative-template.v3" and t.get("launch_authorized") is False,"template differs")
 req(a.get("schema")=="sepalith.sft11.full-weight-cpt-representative-root-admission.v3" and a.get("status")=="admitted" and a.get("launch_authorized") is True,"root admission missing")
 req(a.get("template_sha256")==sha(template_path),"admission refers to another template")
 dtype_audit=t.get("dtype_audit",{});pin(dtype_audit,"saved precision audit")
 req(dtype_audit.get("helper_source_sha256")==sha(Path(__file__).with_name("saved_precision.py")),"saved precision helper differs")
 cohort=a.get("cohort");req(isinstance(cohort,dict),"cohort remains unbound")
 expected={"id","scope","rows","draw_schedule","manifest","unique_rows","documents","packages","input_tokens","payload_tokens","loss_tokens","max_sequence_tokens","named_replays","updates"}
 req(set(cohort)==expected,"cohort fields differ")
 req(cohort==t.get("cohort"),"root admission cohort differs from data-bound template")
 req(cohort["scope"]=="representative_complete_documents_diagnostic_not_all_eligible_corpus","cohort scope overclaims corpus")
 for key in ("rows","draw_schedule","manifest"):pin(cohort[key],"cohort "+key)
 data_admission=a.get("data_admission");pin(data_admission,"data admission")
 req(data_admission==t.get("data_admission"),"root admission data evidence differs from template")
 req(all(type(cohort[k]) is int and cohort[k]>0 for k in ("unique_rows","documents","packages","input_tokens","payload_tokens","loss_tokens","max_sequence_tokens","updates")),"cohort denominator invalid")
 req(type(cohort["named_replays"]) is int and cohort["named_replays"]>=0,"replay denominator invalid")
 req(cohort["updates"]*16==cohort["unique_rows"]+cohort["named_replays"],"coverage/replay/update arithmetic differs")
 req(cohort["max_sequence_tokens"] in (8192,16384,32768),"representative context is not admitted")
 selected=a.get("selected",{});required={"parent_candidate_id","optimizer","micro_batch","gradient_accumulation","learning_rate","scheduler","warmup_ratio","checkpoint_every","mandatory_stop_step","evaluation_steps","selected_milestones","telemetry_every"}
 req(set(selected)==required,"scientific selection fields differ")
 parents=t["parent_candidates"];req(selected["parent_candidate_id"] in parents,"unknown parent candidate");parent=parents[selected["parent_candidate_id"]];req(parent.get("available") is True,"parent candidate is not admitted in this template")
 if parent.get("candidate_evidence") is not None:pin(parent["candidate_evidence"],"parent candidate evidence")
 precision=validate_saved_precision(parent)
 micro=selected["micro_batch"];req(micro in (1,2) and micro*selected["gradient_accumulation"]==16,"effective batch must be 16")
 dispatch=json.loads(Path(t["optimizer_dispatch"]["path"]).read_text());req(selected["optimizer"].get("arm")==dispatch.get("arm")=="aurora_mix","optimizer lacks reviewed dispatch")
 req(selected["optimizer"].get("hidden_lr")==selected["learning_rate"] and 0<selected["learning_rate"]<=1e-3,"learning rate differs")
 req(selected["scheduler"] in ("cosine","constant_with_warmup") and 0<=selected["warmup_ratio"]<1,"scheduler differs")
 stop=selected["mandatory_stop_step"];req(type(stop)is int and 0<stop<cohort["updates"],"mandatory stop must be intermediate")
 req(type(selected["checkpoint_every"])is int and selected["checkpoint_every"]>0 and stop%selected["checkpoint_every"]==0,"mandatory stop lacks checkpoint cadence")
 req(stop in selected["evaluation_steps"] and stop in selected["selected_milestones"] and cohort["updates"] in selected["selected_milestones"],"milestone/evaluation preservation differs")
 req(type(selected["telemetry_every"])is int and selected["telemetry_every"]>0,"telemetry cadence differs")
 bound=dict(t);bound.update({"schema":"sepalith.sft11.full-weight-cpt-representative-bound.v3","status":"root_admitted_not_launched","launch_authorized":True,"root_admission":{"path":str(admission_path.resolve()),"sha256":sha(admission_path)},"data_admission":data_admission,"cohort":cohort,"parent":parent,"runtime":{**selected,"effective_batch":16,"max_steps":cohort["updates"]}});bound.pop("parent_candidates")
 import full_weight_cpt_trainer as trainer
 trainer.validate_context_and_storage(bound);trainer._validated_cohort(bound)
 write_new(output,bound);return bound
def main():
 p=argparse.ArgumentParser();p.add_argument("--template",type=Path,required=True);p.add_argument("--admission",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();v=bind(a.template,a.admission,a.output);print(json.dumps({"status":v["status"],"sha256":sha(a.output)}))
if __name__=="__main__":main()
