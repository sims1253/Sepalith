#!/usr/bin/env python3
"""Root-owned exclusive-CUDA DEV generation from one full-weight checkpoint.

Without the optional flags this is the original DEV75 bf16 run: the same
admission, model load, generation calls and result file.  The flags add:

- ``--dtype {bf16,fp16,fp32}``: generation dtype (bf16 default; fp16 and fp32
  for Kaggle T4s).  Non-bf16 results record ``generation_runtime`` and cannot
  bind a training gate.
- ``--cuda-device N``: pin the Nth visible CUDA device on a multi-GPU host.
- ``--panel {dev75,dev250,both}``: ``dev250`` writes the DEV250 result to
  ``--output``; ``both`` evaluates DEV75 (``--output``) and DEV250
  (``--dev250-output``) in one model load.  The admission then also binds
  ``dev250_panel_sha256`` and ``dev250_output``.
- ``--generation-batch-size N`` and ``--generation-kv-cache {dynamic,static}``:
  batched greedy generation (``campaign_eval.batched_greedy_generate``).  The
  defaults come from the bound recipe's ``dev_gate.generation_batching`` and
  otherwise are 1 and dynamic, the original per-case path.  Batched results
  record ``generation_runtime`` (batch size, ordering, padding side, KV cache);
  a gate whose generation runtime differs from its R1 reference stops for the
  user (``night_arm``).
"""
import argparse,json,os,tempfile,time
from pathlib import Path
from sepalith.training.sft.full_weight_edit_sft import load_bound,identity,sha256
from sepalith.training.checkpoint.campaign_checkpoint import verify_checkpoint
from sepalith.training.eval.milestone_gate import PANEL_SHA,DEV250_PANEL_SHA,DEV250_PANEL_PATH,DEV250_CASES,require
from sepalith.training.eval.campaign_eval import GENERATION_DTYPES,DEFAULT_GENERATION_DTYPE,GENERATION_KV_CACHES,pin_cuda_device,load_development_model,generation_batching

ADMISSION_FIELDS={'schema','status','purpose','cuda_authorized','final_set_access','bound_recipe_sha256','checkpoint_manifest_sha256','panel_sha256','output','deadline_seconds'}
BOTH_PANEL_ADMISSION_FIELDS={'dev250_panel_sha256','dev250_output'}

def write_new(path,value):
 path=Path(path);require(path.is_absolute() and not path.exists(),'fresh absolute DEV result required');path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--bound-recipe',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dev-admission',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True,help='result file: DEV75, or DEV250 with --panel dev250')
 p.add_argument('--panel',choices=('dev75','dev250','both'),default='dev75',help='panel(s) to evaluate in this model load (default: dev75)')
 p.add_argument('--dev250-panel',type=Path,default=Path(DEV250_PANEL_PATH),help='DEV250 panel file; its sha256 is pinned')
 p.add_argument('--dev250-output',type=Path,help='DEV250 result file; required with --panel both')
 p.add_argument('--dtype',choices=tuple(GENERATION_DTYPES),default=DEFAULT_GENERATION_DTYPE,help='generation dtype (default: bf16; fp16/fp32 for Kaggle T4s)')
 p.add_argument('--cuda-device',type=int,help='pin the Nth visible CUDA device (default: the single leased device)')
 p.add_argument('--generation-batch-size',type=int,help="cases per greedy generate call (default: the recipe's dev_gate.generation_batching, else 1 = the original per-case path)")
 p.add_argument('--generation-kv-cache',choices=GENERATION_KV_CACHES,help="KV cache for generation (default: the recipe's, else dynamic)")
 return p.parse_args(argv)
def main(argv=None):
 a=parse_args(argv);both=a.panel=='both';require(both==(a.dev250_output is not None),'--dev250-output goes with --panel both, and only with it')
 recipe_path=a.bound_recipe.resolve();checkpoint=a.checkpoint.resolve();output=a.output.resolve();recipe=load_bound(recipe_path);steps=recipe['dev_gate']['mandatory_steps'];step=int(checkpoint.name.removeprefix('checkpoint-'));require(step in steps,'checkpoint is not a mandatory DEV milestone')
 dev250_output=a.dev250_output.resolve() if both else None;dev250_panel={'path':str(a.dev250_panel.resolve()),'sha256':DEV250_PANEL_SHA}
 manifest=checkpoint/'campaign-manifest.json';verify_checkpoint(checkpoint,identity(recipe),require_full=True,expected_checkpoint_kind='full_weights');manifest_sha=sha256(manifest)
 first_sha=DEV250_PANEL_SHA if a.panel=='dev250' else PANEL_SHA
 admission=json.loads(a.dev_admission.read_text());required=ADMISSION_FIELDS|(BOTH_PANEL_ADMISSION_FIELDS if both else set());require(set(admission)==required,'DEV admission fields differ');require(admission['schema']=='sepalith.sft11.full-weight-edit-dev-generation-admission.v1' and admission['status']=='admitted' and admission['purpose']=='development_generation','DEV generation not admitted');require(admission['cuda_authorized'] is True and admission['final_set_access'] is False,'DEV CUDA/final admission differs');require(admission['bound_recipe_sha256']==sha256(recipe_path) and admission['checkpoint_manifest_sha256']==manifest_sha and admission['panel_sha256']==first_sha and Path(admission['output'])==output,'DEV admission identity differs')
 if both:require(admission['dev250_panel_sha256']==DEV250_PANEL_SHA and Path(admission['dev250_output'])==dev250_output and dev250_output!=output,'DEV admission identity differs')
 require(type(admission['deadline_seconds']) is int and 60<=admission['deadline_seconds']<=3600,'DEV deadline differs')
 if a.cuda_device is not None:pin_cuda_device(a.cuda_device)
 require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,''),'root CUDA lease must expose one device')
 bound_batching=recipe['dev_gate'].get('generation_batching') or {}
 batch_size=a.generation_batch_size if a.generation_batch_size is not None else bound_batching.get('batch_size',1);kv_cache=a.generation_kv_cache or bound_batching.get('kv_cache','dynamic');batching=generation_batching(batch_size,kv_cache)
 # Only non-default runs record their runtime, so default results stay identical to earlier ones.
 runtime={'dtype':a.dtype,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']} if a.dtype!=DEFAULT_GENERATION_DTYPE or a.cuda_device is not None or batching is not None else None
 started=time.monotonic();os.environ['UNSLOTH_RETURN_LOGITS']='1'
 import torch
 from transformers import AutoTokenizer
 from sepalith.training.contracts.campaign_tokenizer_contract import load_pinned_reference_tokenizer
 from sepalith.training.contracts.trainer_tokenizer_alignment import restore_trainer_eog_alignment,assert_runtime_tokenizer,embedding_identity
 from sepalith.training.eval.campaign_eval import development_evaluator
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one root-owned CUDA device required')
 reference=load_pinned_reference_tokenizer(Path(recipe['parent']['path']));tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False);require(sha256(checkpoint/'tokenizer.json')==recipe['parent']['files']['tokenizer.json'],'checkpoint tokenizer bytes differ')
 model=load_development_model(checkpoint,dtype=a.dtype,device='cuda');repair=restore_trainer_eog_alignment(model,tokenizer,reference);embeddings=embedding_identity(model);audit=assert_runtime_tokenizer(model,tokenizer,reference,embeddings,'dev_generation_load');named=list(model.named_parameters());require(len(named)==381 and sum(x.numel() for _,x in named)==2516756480 and not any('lora_' in n.lower() for n,_ in named),'DEV model is not exact dense checkpoint');model.eval()
 runs=[]
 if a.panel!='dev250':runs.append((output,recipe['development']['panel'],PANEL_SHA,75))
 if a.panel!='dev75':runs.append((dev250_output or output,dev250_panel,DEV250_PANEL_SHA,DEV250_CASES))
 for destination,*_ in runs[1:]:require(not destination.exists() and not destination.with_suffix('.cases').exists(),'fresh DEV case directory required')
 for destination,panel,panel_sha,denominator in runs:
  destination.parent.mkdir(parents=True,exist_ok=True);case_dir=destination.with_suffix('.cases');require(not case_dir.exists(),'fresh DEV case directory required');case_dir.mkdir()
  cap=recipe['dev_gate']['generation_max_new_tokens'];rows=[json.loads(line) for line in Path(panel['path']).read_text().splitlines() if line];eval_recipe={'development_panel':panel,'renderer_id':'zeta2-prm03-v1','development_case_ids':sorted(row['id'] for row in rows),'development_max_new_tokens':cap,'parameters':{'max_sequence_tokens':recipe['cohort']['max_sequence_tokens']},'evaluation_output_directory':str(case_dir),'generation_batch_size':batch_size,'generation_kv_cache':kv_cache}
  summary=development_evaluator(eval_recipe)(model,tokenizer,checkpoint,step);require(summary['denominators']['cases']==denominator and summary['panel_sha256']==panel_sha,'DEV denominator differs');cases=case_dir/'cases.json';require(cases.is_file() and json.loads(cases.read_text())['status']=='complete','DEV cases incomplete')
  result={'schema':'sepalith.sft11.full-weight-edit-dev-generation.v1','status':'complete','step':step,'bound_recipe':{'path':str(recipe_path),'sha256':sha256(recipe_path)},'checkpoint':{'path':str(checkpoint),'campaign_manifest_sha256':manifest_sha},'panel':panel,'summary':summary,'cases':{'path':str(cases),'sha256':sha256(cases)},'tokenizer_contract':audit,'tokenizer_repair':repair,'elapsed_seconds':time.monotonic()-started,'final_set_access':False}
  if runtime is not None:result['generation_runtime']={**runtime,**(summary.get('generation_runtime') or {})}
  write_new(destination,result);print(json.dumps({'status':'complete','step':step,'output':str(destination),'sha256':sha256(destination)}))
if __name__=='__main__':main()
