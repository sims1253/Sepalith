#!/usr/bin/env python3
"""Root-owned exclusive-CUDA DEV75 generation from one full-weight checkpoint."""
import argparse,hashlib,json,os,tempfile,time
from pathlib import Path
from full_weight_edit_sft import load_bound,identity,sha256
from campaign_checkpoint import verify_checkpoint
from milestone_gate import PANEL_SHA,require

def write_new(path,value):
 path=Path(path);require(path.is_absolute() and not path.exists(),'fresh absolute DEV result required');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def main():
 p=argparse.ArgumentParser();p.add_argument('--bound-recipe',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dev-admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 recipe_path=a.bound_recipe.resolve();checkpoint=a.checkpoint.resolve();output=a.output.resolve();recipe=load_bound(recipe_path);steps=recipe['dev_gate']['mandatory_steps'];step=int(checkpoint.name.removeprefix('checkpoint-'));require(step in steps,'checkpoint is not a mandatory DEV milestone')
 manifest=checkpoint/'campaign-manifest.json';verify_checkpoint(checkpoint,identity(recipe),require_full=True,expected_checkpoint_kind='full_weights');manifest_sha=sha256(manifest)
 admission=json.loads(a.dev_admission.read_text());required={'schema','status','purpose','cuda_authorized','final_set_access','bound_recipe_sha256','checkpoint_manifest_sha256','panel_sha256','output','deadline_seconds'};require(set(admission)==required,'DEV admission fields differ');require(admission['schema']=='sepalith.sft11.full-weight-edit-dev-generation-admission.v1' and admission['status']=='admitted' and admission['purpose']=='development_generation','DEV generation not admitted');require(admission['cuda_authorized'] is True and admission['final_set_access'] is False,'DEV CUDA/final admission differs');require(admission['bound_recipe_sha256']==sha256(recipe_path) and admission['checkpoint_manifest_sha256']==manifest_sha and admission['panel_sha256']==PANEL_SHA and Path(admission['output'])==output,'DEV admission identity differs');require(type(admission['deadline_seconds']) is int and 60<=admission['deadline_seconds']<=3600,'DEV deadline differs');require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,''),'root CUDA lease must expose one device')
 started=time.monotonic();os.environ['UNSLOTH_RETURN_LOGITS']='1'
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from campaign_tokenizer_contract import load_pinned_reference_tokenizer
 from trainer_tokenizer_alignment import restore_trainer_eog_alignment,assert_runtime_tokenizer,embedding_identity
 from saved_precision import restore_saved_fp32
 from campaign_eval import development_evaluator
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one root-owned CUDA device required')
 reference=load_pinned_reference_tokenizer(Path(recipe['parent']['path']));tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False);require(sha256(checkpoint/'tokenizer.json')==recipe['parent']['files']['tokenizer.json'],'checkpoint tokenizer bytes differ')
 model=AutoModelForCausalLM.from_pretrained(str(checkpoint),torch_dtype=torch.bfloat16,device_map={'':'cuda'},local_files_only=True,trust_remote_code=False);precision_audit=restore_saved_fp32(model,checkpoint/'model.safetensors');precision_audit.update({'helper_sha256':sha256(Path(__file__).resolve().parent/'saved_precision.py'),'weights_sha256':recipe['parent']['files']['model.safetensors']});require(precision_audit.get('exact_saved_values_verified') is True and precision_audit.get('fp32_tensors_restored') in recipe['precision_policy']['allowed_fp32_tensor_counts'],'saved DEV dtype restoration differs');repair=restore_trainer_eog_alignment(model,tokenizer,reference);embeddings=embedding_identity(model);audit=assert_runtime_tokenizer(model,tokenizer,reference,embeddings,'dev_generation_load');named=list(model.named_parameters());require(len(named)==381 and sum(x.numel() for _,x in named)==2516756480 and not any('lora_' in n.lower() for n,_ in named),'DEV model is not exact dense checkpoint');model.eval()
 output.parent.mkdir(parents=True,exist_ok=True);case_dir=output.with_suffix('.cases');require(not case_dir.exists(),'fresh DEV case directory required');case_dir.mkdir()
 panel=recipe['development']['panel'];cap=recipe['dev_gate']['generation_max_new_tokens'];rows=[json.loads(line) for line in Path(panel['path']).read_text().splitlines() if line];eval_recipe={'development_panel':panel,'renderer_id':'zeta2-prm03-v1','development_case_ids':sorted(row['id'] for row in rows),'development_max_new_tokens':cap,'parameters':{'max_sequence_tokens':recipe['cohort']['max_sequence_tokens']},'evaluation_output_directory':str(case_dir)}
 summary=development_evaluator(eval_recipe)(model,tokenizer,checkpoint,step);require(summary['denominators']['cases']==75 and summary['panel_sha256']==PANEL_SHA,'DEV denominator differs');cases=case_dir/'cases.json';require(cases.is_file() and json.loads(cases.read_text())['status']=='complete','DEV cases incomplete')
 result={'schema':'sepalith.sft11.full-weight-edit-dev-generation.v1','status':'complete','step':step,'bound_recipe':{'path':str(recipe_path),'sha256':sha256(recipe_path)},'checkpoint':{'path':str(checkpoint),'campaign_manifest_sha256':manifest_sha},'panel':panel,'summary':summary,'cases':{'path':str(cases),'sha256':sha256(cases)},'saved_checkpoint_precision_audit':precision_audit,'tokenizer_contract':audit,'tokenizer_repair':repair,'elapsed_seconds':time.monotonic()-started,'final_set_access':False}
 write_new(output,result);print(json.dumps({'status':'complete','step':step,'output':str(output),'sha256':sha256(output)}))
if __name__=='__main__':main()
