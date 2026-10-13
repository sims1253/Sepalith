"""Pure R2 checkpoint/parent and export command contracts; no artifact loading."""
import re
from pathlib import Path
SOURCE='8ec908a41904888af647de2ee8b40ba8f73837777509d91159051b25fc72dd3e'
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

def check_task_recipe(r):
 require(r.get('stage')=='task_sft_prm03_v1' and r['identity']['source']==SOURCE,'exact task source/stage required')
 kinds={'midtrain_control':'new_lora_on_midtrain_control','cpt_merged':'new_lora_on_cpt_merged_parent'}
 parent=r['identity']['parent'];require(parent.get('kind') in kinds and r['identity']['policy']['initialization']==kinds[parent['kind']],'parent kind/init mismatch')
 digest(parent['weights_sha256']);require(bool(parent.get('revision')),'parent revision required')
 require(r['parameters']['max_steps']==1000 and r['parameters']['lora_rank']==32 and r['parameters']['lora_alpha']==64,'task horizon/LoRA differs')
 require(r['identity']['schedule']==r['parameters'],'schedule identity differs')
 require(r['renderer_id']=='zeta2-prm03-v1' and r['identity']['renderer']['id']==r['renderer_id'],'PRM03 renderer differs')
 base=path(r['model_path']);records={x['path']:x['sha256'] for x in r['inputs']}
 for name,expected in [('model.safetensors',parent['weights_sha256']),('tokenizer.json',TOK),('tokenizer_config.json',TOKCFG)]:require(records.get(str(base/name))==expected,'parent file binding missing '+name)
 return parent

def export_commands(parent,output):
 p=path(parent);o=path(output);require(p!=o,'fresh separate export directory required')
 return [[PYTHON,'-B',CONVERTER,str(p),'--outfile',str(o/'model-F16.gguf'),'--outtype','f16'],[QUANTIZER,'--output-tensor-type','q8_0','--token-embedding-type','q8_0',str(o/'model-F16.gguf'),str(o/'model-Q8_0.gguf'),'Q8_0','2']]
