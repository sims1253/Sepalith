"""Pure R2 checkpoint/parent and export command contracts; no artifact loading."""
import re
from pathlib import Path
SOURCE='bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384'
RECIPE='2e8038f81c8fee2112b7164e4e0c92c11474459dd1e810f8a86c2bf224ca2316'
PARENT='631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c'
TOK='3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81'
TOKCFG='e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b'
CONVERTER='/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py'
QUANTIZER='/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-quantize'
PYTHON='/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python'
def require(ok,why):
 if not ok:raise ValueError(why)
def digest(value):require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and value not in ('0'*64,'f'*64),'explicit SHA256 required');return value
def path(value):
 p=Path(value);require(p.is_absolute() and '..' not in p.parts,'absolute explicit path required');return p

def check_task_recipe(r, recipe_sha256):
 digest(recipe_sha256);require(recipe_sha256==RECIPE,'exact E recipe bytes required')
 require(r.get('stage')=='expanded_task_sft_v1' and r['identity']['source']==SOURCE,'exact expanded source/stage required')
 parent=r['identity']['parent'];require(parent.get('kind')=='sft_merged' and r['identity']['policy']['initialization']=='new_lora_on_sft_merged_parent','parent kind/init mismatch')
 require(parent.get('weights_sha256')==PARENT and parent.get('previous_checkpoint_step')==500 and parent.get('previous_source_cursor')==8000,'selected merged SFT500 parent differs')
 require(bool(parent.get('revision')),'parent revision required')
 require(r['parameters']['max_steps']==1000 and r['parameters']['lora_rank']==32 and r['parameters']['lora_alpha']==64 and r['parameters'].get('train_max_target_tokens')==1024,'expanded horizon/LoRA/target cap differs')
 require(r['identity']['schedule']==r['parameters'],'schedule identity differs')
 require(r['renderer_id']=='zeta2-prm03-v1' and r['identity']['renderer']['id']==r['renderer_id'],'PRM03 renderer differs')
 require(r.get('development_max_new_tokens')==192,'DEV cap differs')
 require(r.get('resume_from')=='/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-expanded-postsft500-d/full/checkpoint-500','exact D full500 resume required')
 require('resume_identity_compatibility' not in r,'E is an ordinary exact-identity resume')
 require(r.get('milestones')==[250,500,1000] and r.get('mandatory_stop_steps')==[] and r.get('decision_steps')==[],'E terminal1000 control differs')
 require(r.get('checkpoint',{}).get('evaluation_steps')==[250,500,750,1000],'E evaluation cadence differs')
 recovery=r.get('recovery_binding',{})
 require(recovery.get('resume_step')==500 and recovery.get('consumed_draws')==8000 and recovery.get('terminal_step')==1000 and recovery.get('terminal_consumed_draws')==16000,'E recovery cursor differs')
 require(recovery.get('checkpoint_manifest_sha256')=='6a4fe603566af95a09c3fd3bdadea37dac2717db55b08063a0bc9a1d93a1223b' and recovery.get('current_source')==SOURCE and recovery.get('predecessor_source')==SOURCE,'E recovery identity differs')
 base=path(r['model_path']);records={x['path']:x['sha256'] for x in r['inputs']}
 for name,expected in [('model.safetensors',parent['weights_sha256']),('tokenizer.json',TOK),('tokenizer_config.json',TOKCFG)]:require(records.get(str(base/name))==expected,'parent file binding missing '+name)
 return parent

def export_commands(parent,output):
 p=path(parent);o=path(output);require(p!=o,'fresh separate export directory required')
 return [[PYTHON,'-B',CONVERTER,str(p),'--outfile',str(o/'model-F16.gguf'),'--outtype','f16'],[QUANTIZER,'--output-tensor-type','q8_0','--token-embedding-type','q8_0',str(o/'model-F16.gguf'),str(o/'model-Q8_0.gguf'),'Q8_0','2']]
