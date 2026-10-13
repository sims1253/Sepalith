#!/usr/bin/env python3
"""Actual CPU TRL 0.24 + campaign mixins full-weight update/resume proof."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
SRC=HERE/'source'/'experiments'/'training'
sys.path.insert(0,str(SRC))


def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def tensor_hash(model):
 import torch
 h=hashlib.sha256()
 for n,p in sorted(model.state_dict().items()):
  h.update(n.encode()+b'\0'); h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def state_hash(value):
 import torch
 h=hashlib.sha256()
 def walk(x):
  if torch.is_tensor(x): h.update(str(x.dtype).encode()+str(tuple(x.shape)).encode()+x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str): h.update(str(k).encode()+b'\0'); walk(x[k])
  elif isinstance(x,(list,tuple)):
   for v in x: walk(v)
  else: h.update(repr(x).encode()+b'\0')
 walk(value); return h.hexdigest()

def identity(): return {k:{'proof':'tiny-real-trl'} for k in ('parent','tokenizer','renderer','data','source','policy','schedule')}

def make_tokenizer():
 from tokenizers import Tokenizer
 from tokenizers.models import WordLevel
 from tokenizers.pre_tokenizers import Whitespace
 from transformers import PreTrainedTokenizerFast
 vocab={'<bos>':0,'<eos>':1,'<c2>':2,'<c3>':3,'<c4>':4,'<c5>':5,'<c6>':6,'<unk>':7}
 vocab.update({f't{i}':i for i in range(8,32)})
 backend=Tokenizer(WordLevel(vocab=vocab,unk_token='<unk>')); backend.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<bos>',eos_token='<eos>',pad_token='<eos>',unk_token='<unk>')
 tok.padding_side='left'; return tok

def model_types():
 import torch
 from transformers import PreTrainedModel, PretrainedConfig
 from transformers.modeling_outputs import CausalLMOutput
 class TinyConfig(PretrainedConfig):
  model_type='sepalith_tiny_rl_proof'
  def __init__(self,**kw): super().__init__(vocab_size=32,bos_token_id=0,eos_token_id=1,pad_token_id=1,**kw)
 class Attention(torch.nn.Module):
  def __init__(self):
   super().__init__(); self.q_proj=torch.nn.Linear(4,4,bias=False); self.k_proj=torch.nn.Linear(4,4,bias=False); self.v_proj=torch.nn.Linear(4,4,bias=False); self.o_proj=torch.nn.Linear(4,4,bias=False)
  def forward(self,x): return self.o_proj(torch.tanh(self.q_proj(x)+self.k_proj(x)+self.v_proj(x)))
 class MLP(torch.nn.Module):
  def __init__(self):
   super().__init__(); self.gate_proj=torch.nn.Linear(4,8,bias=False); self.up_proj=torch.nn.Linear(4,8,bias=False); self.down_proj=torch.nn.Linear(8,4,bias=False)
  def forward(self,x): return self.down_proj(torch.tanh(self.gate_proj(x))*torch.sigmoid(self.up_proj(x)))
 class Layer(torch.nn.Module):
  def __init__(self): super().__init__(); self.self_attn=Attention(); self.mlp=MLP()
  def forward(self,x): return x+self.self_attn(x)+self.mlp(x)
 class Core(torch.nn.Module):
  def __init__(self): super().__init__(); self.embed_tokens=torch.nn.Embedding(32,4); self.layers=torch.nn.ModuleList([Layer()]); self.norm=torch.nn.LayerNorm(4,elementwise_affine=True,bias=False)
  def forward(self,ids):
   x=self.embed_tokens(ids)
   for layer in self.layers:x=layer(x)
   return self.norm(x)
 class TinyModel(PreTrainedModel):
  config_class=TinyConfig
  def __init__(self,config):
   super().__init__(config); self.model=Core(); self.lm_head=torch.nn.Linear(4,32,bias=False); self.post_init()
   # TRL uses this ordinary Transformers bookkeeping field during init.
   self.warnings_issued={}
  def forward(self,input_ids=None,attention_mask=None,logits_to_keep=None,**kwargs):
   return CausalLMOutput(logits=self.lm_head(self.model(input_ids)))
  def generate(self,input_ids=None,attention_mask=None,**kwargs):
   # Deliberately deterministic candidate diversity. The real policy forward
   # and log-probability/loss below still use this model's logits.
   tails=[]
   for i in range(input_ids.shape[0]): tails.append([24+(i%4),1])
   tail=torch.tensor(tails,dtype=torch.long,device=input_ids.device)
   return SimpleNamespace(sequences=torch.cat((input_ids,tail),dim=1))
 return TinyConfig,TinyModel

def build_trainer(output:Path,max_steps:int,stop_after_one:bool,trace:list):
 import torch
 from datasets import Dataset
 from transformers import TrainerCallback,set_seed
 import campaign_rl_train as rl
 from full_weight_optimizer import FullWeightOptimizerTrainerMixin,OptimizerConfig
 from campaign_checkpoint import seal_checkpoint
 TinyConfig,TinyModel=model_types(); set_seed(1729); random.seed(1729); torch.manual_seed(1729)
 model=TinyModel(TinyConfig()); tokenizer=make_tokenizer()
 GRPO=rl.campaign_grpo_trainer_class()
 class FullWeightCampaignGRPO(FullWeightOptimizerTrainerMixin,GRPO): pass
 config,geometry=rl.make_grpo_config(candidate_count=4,rollout_rows_per_update=4,
   per_device_train_batch_size=4,gradient_accumulation_steps=1,output_dir=output,
   max_steps=max_steps,save_steps=1,learning_rate=2e-3,warmup_steps=0,use_cpu=True,
   generation_kwargs={'do_sample':True,'temperature':1.0,'top_p':1.0,'repetition_penalty':1.0},
   extra={'disable_tqdm':True,'dataloader_pin_memory':False,'gradient_checkpointing':False})
 dataset=Dataset.from_list([
  {'id':'row0','prompt':{'ids':[0,22]}},
  {'id':'row1','prompt':{'ids':[0,23]}},
 ])
 reward_joins=[]
 def exact_reward(completion_ids,id,**kwargs):
  values=[]
  for ids,row in zip(completion_ids,id):
   assert ids[-1]==1 and len(ids)==2
   digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode('ascii')).hexdigest()
   reward_joins.append({'row_id':row,'completion_ids_sha256':digest})
   values.append(float(ids[0]))
  return values
 exact_reward.__name__='exact_id_reward'
 class Boundary(TrainerCallback):
  def on_step_end(self,args,state,control,**kwargs):
   trace.append({'event':'optimizer_boundary','step':int(state.global_step),'model_sha256':tensor_hash(kwargs['model'])})
   if stop_after_one and int(state.global_step)==1: control.should_save=True; control.should_training_stop=True
   return control
  def on_save(self,args,state,control,**kwargs):
   step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
   sampler=kwargs['model']._proof_trainer.campaign_sampler_state()
   sampler.update({'rollout_buffer_state':'empty','rollout_group_cursor':step,'optimizer_global_step':step})
   seal_checkpoint(checkpoint,identity(),step,full=True,sampler=sampler,checkpoint_kind='full_weights')
   trace.append({'event':'full_checkpoint','step':step,'sampler':sampler})
   return control
 trainer=FullWeightCampaignGRPO(model=model,args=config,reward_funcs=exact_reward,train_dataset=dataset,processing_class=tokenizer,callbacks=[Boundary()])
 model._proof_trainer=trainer
 trainer.full_weight_optimizer_config=OptimizerConfig(arm='aurora_mix',hidden_lr=2e-3,side_lr=2e-3,state_dtype='float32',bf16_update_policy='stochastic_round')
 trainer.configure_campaign_runtime(generation_guard_factory=lambda model: __import__('contextlib').nullcontext(),prompt_max_tokens=8,completion_max_tokens=2,context_max_tokens=10,generation_kwargs={'do_sample':True,'temperature':1.0,'top_p':1.0,'repetition_penalty':1.0},generation_groups_per_call=1)
 return trainer,model,geometry,reward_joins

def lane(output,max_steps,stop,resume=None):
 import torch
 from campaign_checkpoint import verify_checkpoint
 trace=[]; trainer,model,geometry,joins=build_trainer(output,max_steps,stop,trace)
 initial=tensor_hash(model)
 result=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
 terminal=tensor_hash(model); opt=state_hash(trainer.optimizer.state_dict()); sched=state_hash(trainer.lr_scheduler.state_dict())
 floating_state_dtypes=sorted({str(value.dtype) for state in trainer.optimizer.state.values()
   for value in state.values() if torch.is_tensor(value) and value.is_floating_point()})
 generation_joins=[hashlib.sha256(json.dumps(record['generated_tokens'],separators=(',',':')).encode('ascii')).hexdigest()
   for record in trainer._campaign_last_generation['records']]
 assert generation_joins==[record['completion_ids_sha256'] for record in joins[-len(generation_joins):]]
 step=int(trainer.state.global_step); cp=output/f'checkpoint-{step}'
 manifest=verify_checkpoint(cp,identity(),require_full=True,expected_checkpoint_kind='full_weights')
 return {'step':step,'initial_model_sha256':initial,'model_sha256':terminal,'optimizer_sha256':opt,'scheduler_sha256':sched,'optimizer_dispatch':trainer.full_weight_optimizer_manifest,'optimizer_floating_state_dtypes':floating_state_dtypes,'trace':trace,'generation_last_group_sha256':generation_joins,'reward_joins':joins,'generation_reward_exact_join':True,'geometry':geometry.to_dict(),'manifest':{'step':manifest['step'],'kind':manifest['checkpoint_kind']},'checkpoint':str(cp),'train_loss':float(result.training_loss)}

def run(root:Path):
 import torch
 os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['TOKENIZERS_PARALLELISM']='false'
 direct=lane(root/'direct',2,False)
 interrupted=lane(root/'interrupted',2,True)
 resumed=lane(root/'resumed',2,False,Path(interrupted['checkpoint']))
 assert direct['initial_model_sha256']==interrupted['initial_model_sha256']==resumed['initial_model_sha256']
 assert direct['step']==resumed['step']==2 and interrupted['step']==1
 assert direct['model_sha256']==resumed['model_sha256'],(direct['model_sha256'],resumed['model_sha256'])
 assert direct['optimizer_sha256']==resumed['optimizer_sha256']
 assert direct['scheduler_sha256']==resumed['scheduler_sha256']
 assert interrupted['trace'][-1]['sampler']['rollout_buffer_state']=='empty'
 assert interrupted['trace'][-1]['sampler']['consumed_rows']==4
 assert resumed['trace'][0]['step']==2
 assert interrupted['model_sha256']!=interrupted['initial_model_sha256']
 assert direct['model_sha256']!=direct['initial_model_sha256']
 assert direct['optimizer_dispatch']['arm']=='aurora_mix'
 assert all(dtypes==['torch.float32'] for dtypes in (direct['optimizer_floating_state_dtypes'],interrupted['optimizer_floating_state_dtypes'],resumed['optimizer_floating_state_dtypes']))
 report={'schema':'sepalith.rl11.full-weight-rl-runtime-proof.v1','status':'PASS','versions':{},'direct':direct,'interrupted':interrupted,'resumed':resumed,'parity':{'model':True,'optimizer':True,'scheduler':True,'resume_first_boundary':2,'source_checkpoint_empty_rollout':True}}
 import importlib.metadata as md
 for n in ('torch','transformers','trl','accelerate','datasets'): report['versions'][n]=md.version(n)
 out=root/'runtime-proof.json'; out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,sort_keys=True)); return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
if __name__=='__main__':main()
