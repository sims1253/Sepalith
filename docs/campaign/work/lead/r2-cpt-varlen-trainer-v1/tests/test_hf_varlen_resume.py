"""Real Transformers 5.5 Trainer test of the logical-update seam and resume."""
import copy,os,sys,tempfile,unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKET/'source/experiments/training'))
from varlen_update_adapter import (PackedOptimizerWindowDataset,
    VarlenUpdateTrainerMixin,logical_update_collator)

import torch
from torch import nn
from torch.utils.data import SequentialSampler
from transformers import Trainer,TrainerCallback,TrainingArguments


class Rows:
 def __init__(self,positions):self.positions=list(positions)
 def __len__(self):return len(self.positions)
 def __getitem__(self,i):
  p=self.positions[i];length=2+(p%4);ids=[(p+j)%13 for j in range(length)]
  return {'input_ids':ids,'labels':[-100]+ids[1:],'attention_mask':[1]*length,'_draw_position':p}


class TinyCausal(nn.Module):
 accepts_loss_kwargs=True
 def __init__(self):
  super().__init__();self.embed=nn.Embedding(13,5);self.head=nn.Linear(5,13)
 def forward(self,input_ids,labels,position_ids=None,packed_seq_lengths=None,num_items_in_batch=None,**_):
  assert num_items_in_batch is not None
  logits=self.head(self.embed(input_ids));shift=logits[:,:-1].reshape(-1,13);gold=labels[:,1:].reshape(-1)
  numerator=nn.functional.cross_entropy(shift,gold,ignore_index=-100,reduction='sum')
  return {'loss':numerator/num_items_in_batch,'logits':logits}


class LogicalTrainer(VarlenUpdateTrainerMixin,Trainer):
 def __init__(self,*a,seen=None,**kw):self.seen=seen if seen is not None else [];self.denominators=[];self.pack_counts=[];super().__init__(*a,**kw)
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
 def before_varlen_update(self,positions,denominator,physical_packs):self.seen.extend(positions);self.denominators.append(denominator);self.pack_counts.append(physical_packs)


class OrdinarySourceTrainer(Trainer):
 def __init__(self,*a,seen=None,**kw):self.seen=seen if seen is not None else [];super().__init__(*a,**kw)
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
 def compute_loss(self,model,inputs,*a,**kw):self.seen.extend(int(x) for x in inputs.pop('_draw_position').tolist());return super().compute_loss(model,inputs,*a,**kw)


class SaveStop(TrainerCallback):
 def __init__(self,step,stop=False):self.step=step;self.stop=stop
 def on_step_end(self,args,state,control,**kw):
  if int(state.global_step)==self.step:
   control.should_save=True
   if self.stop:control.should_training_stop=True
  return control


def arguments(path,max_steps,gradient_accumulation=1):
 return TrainingArguments(output_dir=str(path),per_device_train_batch_size=1,gradient_accumulation_steps=gradient_accumulation,max_steps=max_steps,learning_rate=2e-3,lr_scheduler_type='constant_with_warmup',warmup_steps=1,max_grad_norm=0.0,use_cpu=True,bf16=False,save_strategy='no',save_only_model=False,logging_strategy='no',eval_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=False,train_sampling_strategy='sequential',ignore_data_skip=True,seed=515,data_seed=515)


def state_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b


def trainer(root,name,state,positions,max_steps,callback,seen,token_cap=7):
 model=TinyCausal();model.load_state_dict(copy.deepcopy(state))
 data=PackedOptimizerWindowDataset(Rows(positions),first_position=positions[0],token_cap=token_cap,effective_batch=4)
 return LogicalTrainer(model=model,args=arguments(root/name,max_steps),train_dataset=data,data_collator=logical_update_collator,callbacks=[callback],seen=seen)


def source_trainer(root,state,positions,seen):
 model=TinyCausal();model.load_state_dict(copy.deepcopy(state))
 def collate(items):
  assert len(items)==1;row=items[0];length=len(row['input_ids'])
  return {'input_ids':torch.tensor([row['input_ids']]),'labels':torch.tensor([row['labels']]),
          '_draw_position':torch.tensor([row['_draw_position']])}
 return OrdinarySourceTrainer(model=model,args=arguments(root/'source',2,4),train_dataset=Rows(positions),
   data_collator=collate,callbacks=[SaveStop(2)],seen=seen)


class ActualTrainerTests(unittest.TestCase):
 def test_one_optimizer_update_matches_single_pack_reference(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
   root=Path(td);torch.manual_seed(77);initial=TinyCausal().state_dict()
   reference_seen=[];reference=trainer(root,'one-pack',initial,list(range(4)),1,SaveStop(1),reference_seen,100);reference.train()
   varlen_seen=[];varlen=trainer(root,'many-packs',initial,list(range(4)),1,SaveStop(1),varlen_seen,7);varlen.train()
   self.assertEqual(reference_seen,varlen_seen);self.assertEqual(reference.pack_counts,[1]);self.assertGreater(varlen.pack_counts[0],1)
   for expected,actual in zip(reference.model.parameters(),varlen.model.parameters()):
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
   self.assertEqual(reference.lr_scheduler.state_dict(),varlen.lr_scheduler.state_dict())

 def test_interrupted_resume_matches_uninterrupted_and_starts_exact_cursor(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
   root=Path(td);torch.manual_seed(99);initial=TinyCausal().state_dict()
   source_seen=[];source=source_trainer(root,initial,list(range(100,108)),source_seen);source.train();source_checkpoint=root/'source/checkpoint-2'
   self.assertTrue((source_checkpoint/'optimizer.pt').is_file() and (source_checkpoint/'scheduler.pt').is_file())
   direct_seen=[];direct=trainer(root,'direct',initial,list(range(12)),5,SaveStop(5),direct_seen);direct.train(resume_from_checkpoint=str(source_checkpoint))
   first_seen=[];first=trainer(root,'split',initial,list(range(12)),5,SaveStop(3,True),first_seen);first.train(resume_from_checkpoint=str(source_checkpoint));checkpoint=root/'split/checkpoint-3'
   self.assertTrue((checkpoint/'optimizer.pt').is_file() and (checkpoint/'scheduler.pt').is_file())
   resumed_seen=[];resumed=trainer(root,'resumed',initial,list(range(4,12)),5,SaveStop(5),resumed_seen);resumed.train(resume_from_checkpoint=str(checkpoint))
   self.assertEqual(source_seen,list(range(100,108)));self.assertEqual(direct_seen,list(range(12)));self.assertEqual(first_seen,list(range(4)));self.assertEqual(resumed_seen,list(range(4,12)))
   self.assertEqual(source.args.gradient_accumulation_steps,4);self.assertEqual(direct.args.gradient_accumulation_steps,1)
   self.assertEqual(int(direct.state.global_step),5);self.assertEqual(int(resumed.state.global_step),5)
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(direct.model.state_dict().values(),resumed.model.state_dict().values())))
   self.assertTrue(state_equal(direct.optimizer.state_dict(),resumed.optimizer.state_dict()))
   self.assertEqual(direct.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict())
   self.assertTrue(all(x>0 for x in direct.denominators));self.assertTrue(all(x>=2 for x in direct.pack_counts))


if __name__=='__main__':unittest.main(verbosity=2)
