#!/usr/bin/env python3
"""Capture exact Trainer resume state before the first resumed forward pass."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path

UPSTREAM=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-full-weight-resume-proof-v3/source')
sys.path.insert(0,str(UPSTREAM)); import full_weight_resume_proof as proof

class RestorationCaptured(RuntimeError): pass

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def run(recipe_path,resume,expected_report,output_dir,result_path):
 recipe=proof.load_recipe(recipe_path);resume=Path(resume);expected=json.loads(Path(expected_report).read_text())
 from campaign_checkpoint import verify_checkpoint
 manifest=verify_checkpoint(resume,proof.identity(recipe),require_full=True,expected_checkpoint_kind=proof.FULL_KIND)
 if manifest['step']!=1 or expected['global_step']!=1 or str(resume)!=expected['checkpoint']:raise ValueError('checkpoint/report step or path differs')
 os.environ['UNSLOTH_RETURN_LOGITS']='0'
 from unsloth import FastLanguageModel
 import torch
 from torch.utils.data import Dataset,SequentialSampler
 from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
 from full_weight_optimizer import FullWeightOptimizerTrainerMixin,OptimizerConfig
 from campaign_tokenizer_contract import load_pinned_reference_tokenizer,restore_pinned_tokenizer_contract
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('requires exactly one root-owned CUDA device')
 seed=int(recipe['seed']);set_seed(seed)
 rows=[json.loads(x) for x in Path(recipe['data']['path']).read_text().splitlines() if x]
 class Data(Dataset):
  def __len__(self):return len(rows)
  def __getitem__(self,i):return {k:rows[i][k] for k in ('input_ids','attention_mask','labels')}
 model,tokenizer=FastLanguageModel.from_pretrained(model_name=recipe['model']['path'],max_seq_length=2048,dtype=torch.bfloat16,load_in_4bit=False,full_finetuning=True,float32_mixed_precision=False,fast_inference=False,trust_remote_code=False,use_gradient_checkpointing=True)
 reference=load_pinned_reference_tokenizer(Path(recipe['model']['path']));restore_pinned_tokenizer_contract(model,tokenizer,reference_tokenizer=reference,prompt_rows=())
 model.config.use_cache=False
 try:FastLanguageModel.for_training(model,use_gradient_checkpointing=True)
 except TypeError:FastLanguageModel.for_training(model)
 model.train();dispatch=json.loads(Path(recipe['optimizer']['expected_dispatch_path']).read_text())
 class Capture(TrainerCallback):
  def on_step_begin(self,args,state,control,**kwargs):
   if int(state.global_step)!=1:raise ValueError('capture did not occur at restored step1')
   observed={'model_state_sha256':proof.tensor_mapping_digest(model.state_dict()),'optimizer_state_sha256':proof.tensor_mapping_digest(trainer.optimizer.state_dict()),'scheduler_state_sha256':hashlib.sha256(proof.canonical(trainer.lr_scheduler.state_dict())).hexdigest(),'cpu_rng_sha256':hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),'cuda_rng_sha256':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),'optimizer_dispatch_sha256':hashlib.sha256(proof.canonical(trainer.full_weight_optimizer_manifest)).hexdigest()}
   comparisons={k:observed[k]==expected[k] for k in observed}
   result={'schema':'sepalith.sft11.preupdate-restoration-proof.v1','status':'exact_preupdate_restoration' if all(comparisons.values()) else 'restoration_mismatch','checkpoint':str(resume),'checkpoint_manifest_step':manifest['step'],'global_step':int(state.global_step),'capture_boundary':'on_step_begin after Trainer _prepare_for_training and resumed dataloader skip/RNG restore; before training_step/forward/backward/optimizer','comparisons':comparisons,'observed':observed,'expected_report':str(Path(expected_report)),'optimizer_update_executed':False}
   write_json(result_path,result);raise RestorationCaptured()
 class T(FullWeightOptimizerTrainerMixin,Trainer):
  def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
  def create_optimizer(self):
   x=super().create_optimizer()
   for k in ('arm','ordered_rows','ordered_rows_sha256','parameter_objects','parameters','counts'):
    if self.full_weight_optimizer_manifest.get(k)!=dispatch.get(k):raise ValueError('optimizer dispatch differs:'+k)
   return x
 vals=proof.training_arguments_kwargs(Path(output_dir),seed);vals['learning_rate']=recipe['optimizer']['config']['hidden_lr'];args=TrainingArguments(**vals)
 trainer=T(model=model,args=args,train_dataset=Data(),processing_class=tokenizer,callbacks=[Capture()]);trainer.full_weight_optimizer_config=OptimizerConfig(**recipe['optimizer']['config'])
 try:trainer.train(resume_from_checkpoint=str(resume))
 except RestorationCaptured:pass
 else:raise ValueError('restoration capture did not stop before forward')
 result=json.loads(Path(result_path).read_text())
 if result['status']!='exact_preupdate_restoration':raise ValueError('restored in-memory state differs before forward')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--resume',type=Path,required=True);p.add_argument('--expected-report',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.recipe,a.resume,a.expected_report,a.output_dir,a.result),sort_keys=True))
if __name__=='__main__':main()
