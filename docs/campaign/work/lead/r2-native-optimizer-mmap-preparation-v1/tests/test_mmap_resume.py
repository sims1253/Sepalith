import copy,fcntl,gc,json,os,sys,tempfile,unittest
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACKET/'source'))
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,SequentialSampler
from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
from full_weight_optimizer import FullWeightOptimizerTrainerMixin,OptimizerConfig
from mmap_optimizer_resume import MmapOptimizerResumeTrainerMixin,fingerprint,mmap_load_optimizer_state,assert_materialized_without_alias
class Attention(nn.Module):
 def __init__(self):super().__init__();self.q_proj=nn.Linear(4,4,bias=False)
class MLP(nn.Module):
 def __init__(self):super().__init__();self.gate_proj=nn.Linear(4,8,bias=False);self.up_proj=nn.Linear(4,8,bias=False);self.down_proj=nn.Linear(8,4,bias=False)
class Block(nn.Module):
 def __init__(self):super().__init__();self.self_attn=Attention();self.mlp=MLP()
class Body(nn.Module):
 def __init__(self):super().__init__();self.embed_tokens=nn.Embedding(17,4);self.layers=nn.ModuleList([Block()])
class Model(nn.Module):
 accepts_loss_kwargs=True
 def __init__(self):super().__init__();self.model=Body();self.final_norm=nn.LayerNorm(4);self.lm_head=nn.Linear(4,17,bias=False)
 def forward(self,input_ids,labels,num_items_in_batch=None,**_):
  b=self.model.layers[0];x=self.model.embed_tokens(input_ids);x=b.self_attn.q_proj(x);x=x+b.mlp.down_proj(torch.relu(b.mlp.gate_proj(x))+b.mlp.up_proj(x));logits=self.lm_head(self.final_norm(x));loss=nn.functional.cross_entropy(logits[:,:-1].reshape(-1,17),labels[:,1:].reshape(-1),reduction='sum')/num_items_in_batch;return {'loss':loss,'logits':logits}
class Rows(Dataset):
 def __init__(self,start=0,count=4):self.start=start;self.count=count
 def __len__(self):return self.count
 def __getitem__(self,i):
  i+=self.start;ids=[i+2,i+3,i+4];return {'input_ids':ids,'labels':[-100]+ids[1:]}
def collate(items):return {k:torch.tensor([x[k] for x in items])for k in ('input_ids','labels')}
def arguments(path):return TrainingArguments(output_dir=str(path),per_device_train_batch_size=1,gradient_accumulation_steps=2,max_steps=2,learning_rate=3e-4,lr_scheduler_type='constant_with_warmup',warmup_steps=1,max_grad_norm=0.,use_cpu=True,bf16=False,save_strategy='no',logging_strategy='no',eval_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=False,train_sampling_strategy='sequential',ignore_data_skip=True,seed=73,data_seed=73)
class Stop(TrainerCallback):
 def __init__(self,step,stop):self.step=step;self.stop=stop
 def on_step_end(self,args,state,control,**_):
  if int(state.global_step)==self.step:control.should_save=True;control.should_training_stop=self.stop
  return control
class Base(FullWeightOptimizerTrainerMixin,Trainer):
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset or self.train_dataset)
 def get_train_dataloader(self):
  generator=torch.Generator(device='cpu');generator.manual_seed(int(self.args.data_seed));loader=DataLoader(self.train_dataset,batch_size=self._train_batch_size,sampler=SequentialSampler(self.train_dataset),collate_fn=self.data_collator,num_workers=0,pin_memory=False,drop_last=False,generator=generator);return self.accelerator.prepare(loader)
class Mmap(MmapOptimizerResumeTrainerMixin,FullWeightOptimizerTrainerMixin,Trainer):
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset or self.train_dataset)
 def get_train_dataloader(self):return Base.get_train_dataloader(self)
def config():return OptimizerConfig(arm='aurora_mix',hidden_lr=3e-4,side_lr=3e-5,weight_decay=0.,ns_steps=2,aurora_K=1,stochastic_round_chunk_elements=128)
def build(klass,path,state,callback,start=0,count=4):
 m=Model();m.load_state_dict(copy.deepcopy(state));t=klass(model=m,args=arguments(path),train_dataset=Rows(start,count),data_collator=collate,callbacks=[callback]);t.full_weight_optimizer_config=config();return t
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
 return a==b
class Proof(unittest.TestCase):
 def setUp(self):self.old=os.environ.get('SEPALITH_NATIVE_STAGE_ATTESTATION')
 def tearDown(self):
  if self.old is None:os.environ.pop('SEPALITH_NATIVE_STAGE_ATTESTATION',None)
  else:os.environ['SEPALITH_NATIVE_STAGE_ATTESTATION']=self.old
 def attest(self,path):
  fd=os.open(path,os.O_RDONLY);fcntl.flock(fd,fcntl.LOCK_SH);s=os.fstat(fd);os.environ['SEPALITH_NATIVE_STAGE_ATTESTATION']=json.dumps({'files':[{'path':str(Path(path).resolve()),'fd':fd,'bytes':s.st_size,'sha256':'a'*64,'fingerprint':fingerprint(s)}]});return fd
 def test_actual_trainer_mmap_resume_matches_stock_state_scheduler_rng_and_model(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);torch.manual_seed(11);state=Model().state_dict();set_seed(101);direct=build(Base,root/'direct',state,Stop(2,True));direct.train();direct_rng=torch.get_rng_state().clone();set_seed(101);split=build(Base,root/'split',state,Stop(1,True));split.train();cp=root/'split/checkpoint-1';fd=self.attest(cp/'optimizer.pt')
   try:
    set_seed(999);stock=build(Base,root/'stock',state,Stop(2,True),2,2);stock.train(resume_from_checkpoint=str(cp));stock_rng=torch.get_rng_state().clone()
    set_seed(555);mapped=build(Mmap,root/'mapped',state,Stop(2,True),2,2);mapped.train(resume_from_checkpoint=str(cp));mapped_rng=torch.get_rng_state().clone()
   finally:os.close(fd)
   self.assertTrue(equal(direct.model.state_dict(),stock.model.state_dict()),'direct/resumed model');self.assertTrue(equal(direct.optimizer.state_dict(),stock.optimizer.state_dict()),'direct/resumed optimizer');self.assertEqual(direct.lr_scheduler.state_dict(),stock.lr_scheduler.state_dict(),'direct/resumed scheduler');self.assertTrue(torch.equal(direct_rng,stock_rng),'direct/resumed RNG');self.assertTrue(equal(stock.model.state_dict(),mapped.model.state_dict()),'stock/mmap model');self.assertTrue(equal(stock.optimizer.state_dict(),mapped.optimizer.state_dict()),'stock/mmap optimizer');self.assertEqual(stock.lr_scheduler.state_dict(),mapped.lr_scheduler.state_dict(),'stock/mmap scheduler');self.assertTrue(torch.equal(stock_rng,mapped_rng),'stock/mmap RNG');self.assertEqual(mapped.state.global_step,stock.state.global_step);self.assertEqual(mapped.state.global_step,2);self.assertEqual(mapped.mmap_optimizer_load_report['mapped_storage_aliases'],0);self.assertGreater(mapped.mmap_optimizer_load_report['mapped_tensor_bytes'],0)
 def test_materialized_state_survives_mapped_file_rename_and_bad_attestation_fails(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);torch.manual_seed(7);state=Model().state_dict();t=build(Base,root/'source',state,Stop(1,True));t.train();cp=root/'source/checkpoint-1';opt=cp/'optimizer.pt';fd=self.attest(opt)
   try:
    target=build(Mmap,root/'target',state,Stop(2,False));target.create_optimizer_and_scheduler(2);loaded,_,mapped=mmap_load_optimizer_state(opt);target.optimizer.load_state_dict(loaded);audit=assert_materialized_without_alias(target.optimizer,mapped);snapshot=copy.deepcopy(target.optimizer.state_dict());del loaded,mapped;gc.collect();moved=cp/'optimizer.renamed';opt.rename(moved);self.assertTrue(equal(snapshot,target.optimizer.state_dict()));moved.rename(opt);self.assertEqual(audit['mapped_storage_aliases'],0)
   finally:os.close(fd)
   os.environ['SEPALITH_NATIVE_STAGE_ATTESTATION']=json.dumps({'files':[]})
   with self.assertRaisesRegex(ValueError,'unique native attestation'):mmap_load_optimizer_state(opt)
if __name__=='__main__':unittest.main(verbosity=2)
