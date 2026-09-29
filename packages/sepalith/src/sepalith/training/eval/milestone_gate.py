"""Fail-closed DEV milestone receipts for optimizer-boundary SFT continuation."""
import hashlib,json
from pathlib import Path
PANEL_SHA='7c144bcd20570b1b1b1a232a5d90589c281ac2955d5fc2ef4b4b01fed8d57035'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def mandatory_steps(recipe):
 p=recipe['dev_gate'];steps=p['mandatory_steps'];require(p['panel']['sha256']==PANEL_SHA,'gate panel differs');require(isinstance(steps,list) and steps==sorted(set(steps)) and all(type(x)is int and x>0 for x in steps),'gate steps differ');require(steps==recipe['runtime']['evaluation_steps'],'every evaluation milestone must be a mandatory stop');require(steps[-1]==recipe['runtime']['max_steps'],'terminal gate missing');cap=p.get('generation_max_new_tokens');require(type(cap)is int and 1<=cap<=recipe['development']['generation_budget']['prepared_max_new_tokens_bound'],'gate generation budget differs');root=Path(p['evidence_directory']);require(root.is_absolute() and str(root).startswith('/home/m0hawk/.local/state/sepalith/campaign-20260915/evaluation-gates/'),'gate evidence directory differs');return steps
def evidence_path(recipe,step):return Path(recipe['dev_gate']['evidence_directory'])/f'dev-gate-step-{step}.json'
def verify_gate_evidence(recipe,bound_recipe_path,step,checkpoint):
 path=evidence_path(recipe,step);require(path.is_file(),f'missing root DEV gate evidence for step {step}')
 rec=json.loads(path.read_text());required={'schema','status','step','bound_recipe','checkpoint','panel','result','root_decision'};require(set(rec)==required,'gate evidence fields differ');require(rec['schema']=='sepalith.sft11.full-weight-edit-dev-gate.v1' and rec['status']=='accepted_for_continuation' and rec['step']==step,'gate evidence status differs')
 require(rec['bound_recipe']=={'path':str(Path(bound_recipe_path).resolve()),'sha256':sha(bound_recipe_path)},'gate recipe binding differs')
 checkpoint=Path(checkpoint).resolve();manifest=checkpoint/'campaign-manifest.json';require(manifest.is_file(),'gate checkpoint manifest missing');require(rec['checkpoint']=={'path':str(checkpoint),'campaign_manifest_sha256':sha(manifest)},'gate checkpoint binding differs')
 require(rec['panel']==recipe['dev_gate']['panel'],'gate panel binding differs')
 result=rec['result'];require(Path(result['path']).is_file() and sha(result['path'])==result['sha256'],'gate result differs');value=json.loads(Path(result['path']).read_text());require(value['status']=='complete' and value['step']==step and value['checkpoint']['campaign_manifest_sha256']==sha(manifest) and value['bound_recipe']['sha256']==sha(bound_recipe_path) and value['panel']['sha256']==PANEL_SHA,'generation result binding differs');require(value['summary']['denominators']['cases']==75 and value['summary']['counts'].get('protocol_valid') is not None and value['summary'].get('generation_max_new_tokens')==recipe['dev_gate']['generation_max_new_tokens'],'generation result denominator/budget differs')
 decision=rec['root_decision'];require(Path(decision['path']).is_file() and sha(decision['path'])==decision['sha256'],'root DEV decision differs');d=json.loads(Path(decision['path']).read_text());require(d.get('schema')=='sepalith.sft11.full-weight-edit-dev-decision.v1' and d.get('status')=='admitted_for_continuation' and d.get('continue_training') is True,'root did not admit continuation');require(d.get('step')==step and d.get('bound_recipe_sha256')==sha(bound_recipe_path) and d.get('checkpoint_manifest_sha256')==sha(manifest) and d.get('generation_result_sha256')==result['sha256'] and d.get('panel_sha256')==PANEL_SHA,'root decision binding differs')
 return rec
def continuation_plan(recipe,bound_recipe_path,resume_checkpoint):
 steps=mandatory_steps(recipe);initial=0
 if resume_checkpoint is not None:
  checkpoint=Path(resume_checkpoint);initial=int(checkpoint.name.removeprefix('checkpoint-'));require(0<initial<steps[-1] or initial in steps,'resume step lies outside the admitted horizon')
  for step in [x for x in steps if x<=initial]:
   archived=checkpoint if step==initial else Path(recipe['outputs']['archive'])/'full'/f'checkpoint-{step}'
   verify_gate_evidence(recipe,bound_recipe_path,step,archived)
 next_steps=[x for x in steps if x>initial];require(next_steps,'no continuation exists beyond terminal gate')
 return {'initial_step':initial,'next_mandatory_stop':next_steps[0],'prior_gates_verified':[x for x in steps if x<=initial]}
