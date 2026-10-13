#!/usr/bin/env python3
import argparse,hashlib,json,os,tempfile
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def write_new(path,x):
 path=Path(path);require(not path.exists(),'bound output must be fresh');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def bind(template_path,admission_path,output):
 template_path=Path(template_path);admission_path=Path(admission_path);t=json.loads(template_path.read_text());a=json.loads(admission_path.read_text())
 require(t.get('schema')=='sepalith.sft11.full-weight-edit-sft-eval-gate-template.v1' and t.get('parent') is None and t.get('launch_authorized') is False,'template differs')
 require(a.get('schema')=='sepalith.sft11.full-weight-edit-sft-eval-gate-root-admission.v1' and a.get('status')=='admitted' and a.get('launch_authorized') is True,'root admission differs')
 require(a.get('template_sha256')==sha(template_path),'admission template hash differs')
 parent=a.get('parent',{});require(parent.get('kind')=='merged_cpt_parent' and parent.get('selection_status')=='root_selected_for_edit_sft_parent','selected parent differs')
 require(set(parent.get('files',{}))=={'model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','parent-manifest.preparation.json'},'parent file inventory differs')
 require(parent['files']['tokenizer.json']=='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81','parent tokenizer differs')
 s=a.get('selected',{});require(set(s)=={'optimizer','micro_batch','gradient_accumulation','learning_rate','scheduler','warmup_ratio','checkpoint_every','evaluation_steps','selected_milestones','telemetry_every','mandatory_dev_gate_steps','gate_evidence_directory','development_max_new_tokens'},'runtime selection fields differ')
 require(s['micro_batch'] in (1,2) and s['micro_batch']*s['gradient_accumulation']==16,'effective batch differs')
 optimizer=s['optimizer'];expected_optimizer_fields={'arm','hidden_lr','side_lr','momentum','ns_steps','rms_scale','aurora_beta','aurora_K','adam_betas','adam_eps','weight_decay','state_dtype','bf16_update_policy','stochastic_round_chunk_elements'}
 require(set(optimizer)==expected_optimizer_fields,'optimizer fields differ')
 require(optimizer.get('arm')=='aurora_mix' and optimizer.get('hidden_lr')==s['learning_rate'],'optimizer/LR differs')
 require(type(s['learning_rate']) is float and 0<s['learning_rate']<=3e-5,'editing LR exceeds prepared post-CPT bound')
 require(type(optimizer['side_lr']) is float and 0<optimizer['side_lr']<=optimizer['hidden_lr'],'side LR differs')
 require(optimizer['momentum']==0.95 and optimizer['ns_steps']==5 and optimizer['rms_scale']==0.2 and optimizer['aurora_beta']==0.5 and optimizer['aurora_K']==2,'Aurora mechanics differ')
 require(optimizer['adam_betas']==[0.9,0.999] and optimizer['adam_eps']==1e-8 and optimizer['weight_decay']==0.0,'Adam mechanics differ')
 require(optimizer['state_dtype']=='float32' and optimizer['bf16_update_policy']=='stochastic_round' and optimizer['stochastic_round_chunk_elements']==1048576,'optimizer precision differs')
 require(s['scheduler'] in ('cosine','constant_with_warmup') and type(s['warmup_ratio']) is float and 0<=s['warmup_ratio']<1,'scheduler differs')
 cadence=s['checkpoint_every'];evals=s['evaluation_steps'];miles=s['selected_milestones'];max_steps=t['cohort']['updates']
 require(type(cadence) is int and cadence>0 and max_steps%cadence==0,'checkpoint cadence must divide horizon')
 require(isinstance(evals,list) and all(type(x)is int and 1<=x<=max_steps and x%cadence==0 for x in evals),'evaluation steps differ')
 require(set(evals)<=set(miles) and max_steps in miles,'milestone retention differs')
 require(type(s['telemetry_every'])is int and s['telemetry_every']>0,'telemetry cadence differs')
 require(s['mandatory_dev_gate_steps']==evals and evals==sorted(set(evals)) and evals[-1]==max_steps,'mandatory DEV stops must equal evaluation milestones through terminal')
 cap=s['development_max_new_tokens'];prepared=t['development']['generation_budget'];require(type(cap)is int and 1<=cap<=prepared['prepared_max_new_tokens_bound'],'development generation budget differs')
 gate_dir=Path(s['gate_evidence_directory']);require(gate_dir.is_absolute() and str(gate_dir).startswith('/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/'),'gate evidence directory differs')
 out=dict(t);out['schema']='sepalith.sft11.full-weight-edit-sft-eval-gate-bound.v1';out['status']='root_admitted_not_launched';out['parent']=parent;out['runtime']={**s,'effective_batch':16,'max_steps':max_steps};out['dev_gate']={'mandatory_steps':s['mandatory_dev_gate_steps'],'evidence_directory':s['gate_evidence_directory'],'panel':t['development']['panel'],'continuation_policy':'fail_closed_root_generation_decision_required','generation_max_new_tokens':cap};out['root_admission']={'path':str(admission_path.resolve()),'sha256':sha(admission_path)}
 write_new(output,out);return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=bind(a.template,a.admission,a.output);print(json.dumps({'status':x['status'],'output':str(a.output),'sha256':sha(a.output)}))
if __name__=='__main__':main()
