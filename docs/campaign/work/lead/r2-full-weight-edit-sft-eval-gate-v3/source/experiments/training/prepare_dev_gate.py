#!/usr/bin/env python3
"""Bind an actual DEV generation result and explicit root decision for resume."""
import argparse,json,os,tempfile
from pathlib import Path
from milestone_gate import PANEL_SHA,sha,require

def write_new(path,value):
 path=Path(path);require(path.is_absolute() and not path.exists(),'fresh absolute gate evidence required');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def prepare(recipe_path,checkpoint,result_path,decision_path,output):
 recipe_path=Path(recipe_path).resolve();checkpoint=Path(checkpoint).resolve();result_path=Path(result_path).resolve();decision_path=Path(decision_path).resolve();output=Path(output).resolve();recipe=json.loads(recipe_path.read_text());step=int(checkpoint.name.removeprefix('checkpoint-'));require(step in recipe['dev_gate']['mandatory_steps'],'step is not a mandatory gate');require(output==Path(recipe['dev_gate']['evidence_directory'])/f'dev-gate-step-{step}.json','gate output path differs')
 manifest=checkpoint/'campaign-manifest.json';require(manifest.is_file(),'checkpoint manifest missing');msha=sha(manifest);result=json.loads(result_path.read_text());rsha=sha(result_path);require(result.get('schema')=='sepalith.sft11.full-weight-edit-dev-generation.v1' and result.get('status')=='complete' and result.get('step')==step,'generation result incomplete');require(result.get('bound_recipe',{}).get('sha256')==sha(recipe_path) and result.get('checkpoint',{}).get('campaign_manifest_sha256')==msha and result.get('panel',{}).get('sha256')==PANEL_SHA,'generation result identity differs');require(result.get('summary',{}).get('denominators',{}).get('cases')==75,'generation denominator differs')
 decision=json.loads(decision_path.read_text());dsha=sha(decision_path);require(decision.get('schema')=='sepalith.sft11.full-weight-edit-dev-decision.v1' and decision.get('status')=='admitted_for_continuation' and decision.get('continue_training') is True,'root did not admit continuation');require(decision.get('step')==step and decision.get('bound_recipe_sha256')==sha(recipe_path) and decision.get('checkpoint_manifest_sha256')==msha and decision.get('generation_result_sha256')==rsha and decision.get('panel_sha256')==PANEL_SHA,'root decision identity differs')
 record={'schema':'sepalith.sft11.full-weight-edit-dev-gate.v1','status':'accepted_for_continuation','step':step,'bound_recipe':{'path':str(recipe_path),'sha256':sha(recipe_path)},'checkpoint':{'path':str(checkpoint),'campaign_manifest_sha256':msha},'panel':recipe['dev_gate']['panel'],'result':{'path':str(result_path),'sha256':rsha},'root_decision':{'path':str(decision_path),'sha256':dsha}};write_new(output,record);return record
def main():
 p=argparse.ArgumentParser();p.add_argument('--bound-recipe',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--root-decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=prepare(a.bound_recipe,a.checkpoint,a.result,a.root_decision,a.output);print(json.dumps({'status':x['status'],'step':x['step'],'output':str(a.output),'sha256':sha(a.output)}))
if __name__=='__main__':main()
