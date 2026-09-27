"""Actual Transformers Trainer transition/resume control on a tiny CPU model."""
import copy,os,sys,tempfile,unittest
from pathlib import Path

PACKET=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKET/'source/experiments/training'))
from streaming_trainer_adapter import StageCursorView

import torch
from torch import nn
from transformers import Trainer,TrainerCallback,TrainingArguments

class Rows:
 def __init__(self,ids):self.ids=list(ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  row=self.ids[i];return {'input_ids':torch.tensor([row%7,(row+1)%7]),'labels':torch.tensor(row%3),'_draw_position':torch.tensor(row)}
 def close(self):pass

class Tiny(nn.Module):
 def __init__(self):super().__init__();self.embed=nn.Embedding(7,4);self.drop=nn.Identity();self.head=nn.Linear(4,3)
 def forward(self,input_ids,labels,**_):
  logits=self.head(self.drop(self.embed(input_ids).mean(1)));return {'loss':nn.functional.cross_entropy(logits,labels),'logits':logits}

class TrackTrainer(Trainer):
 def __init__(self,*a,seen=None,**kw):self.seen=seen if seen is not None else [];super().__init__(*a,**kw)
 def compute_loss(self,model,inputs,*a,**kw):self.seen.extend(int(x) for x in inputs.pop('_draw_position').tolist());return super().compute_loss(model,inputs,*a,**kw)

class SaveStop(TrainerCallback):
 def __init__(self,save_step,stop=False):self.save_step=save_step;self.stop=stop;self.lrs=[]
 def on_step_end(self,args,state,control,optimizer=None,**kw):
  self.lrs.append([float(x['lr']) for x in optimizer.param_groups])
  if int(state.global_step)==self.save_step:
   control.should_save=True
   if self.stop:control.should_training_stop=True
  return control

def collate(rows):return {k:torch.stack([x[k] for x in rows]) for k in rows[0]}

def arguments(path,max_steps,ignore_data_skip):
 return TrainingArguments(output_dir=str(path),per_device_train_batch_size=1,gradient_accumulation_steps=1,max_steps=max_steps,learning_rate=1e-3,lr_scheduler_type='constant_with_warmup',warmup_steps=1,max_grad_norm=0.0,use_cpu=True,bf16=False,save_strategy='no',save_only_model=False,logging_strategy='no',eval_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=False,train_sampling_strategy='sequential',ignore_data_skip=ignore_data_skip,seed=818,data_seed=818)

def state_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b

class TestActualTrainerTransition(unittest.TestCase):
 def test_source_to_new_rows_then_stage_local_resume_matches_uninterrupted(self):
  with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR','/mnt/e')) as td:
   root=Path(td);torch.manual_seed(818)
   source_cb=SaveStop(2);source=TrackTrainer(model=Tiny(),args=arguments(root/'source',2,False),train_dataset=Rows([100,101]),data_collator=collate,callbacks=[source_cb]);source.train();checkpoint=root/'source/checkpoint-2';self.assertTrue((checkpoint/'optimizer.pt').is_file());self.assertEqual(int(source.state.global_step),2)
   def destination(name,dataset,resume,stop_at=None):
    seen=[];cb=SaveStop(stop_at or 5,stop_at is not None);trainer=TrackTrainer(model=Tiny(),args=arguments(root/name,5,True),train_dataset=dataset,data_collator=collate,callbacks=[cb],seen=seen);trainer.train(resume_from_checkpoint=str(resume));return trainer,seen,cb
   direct,direct_rows,direct_cb=destination('direct',StageCursorView(Rows([0,1,2]),0),checkpoint)
   first,first_rows,first_cb=destination('split',StageCursorView(Rows([0,1,2]),0),checkpoint,3);mid=root/'split/checkpoint-3';self.assertTrue((mid/'optimizer.pt').is_file())
   resumed,resumed_rows,resumed_cb=destination('resumed',StageCursorView(Rows([0,1,2]),1),mid)
   self.assertEqual(direct_rows,[0,1,2]);self.assertEqual(first_rows,[0]);self.assertEqual(resumed_rows,[1,2]);self.assertEqual(int(direct.state.global_step),5);self.assertEqual(int(resumed.state.global_step),5)
   self.assertTrue(all(x==1e-3 for x in direct_cb.lrs[0]));self.assertTrue(all(x==1e-3 for x in first_cb.lrs[0]))
   self.assertTrue(all(torch.equal(x,y) for x,y in zip(direct.model.state_dict().values(),resumed.model.state_dict().values())))
   self.assertTrue(state_equal(direct.optimizer.state_dict(),resumed.optimizer.state_dict()));self.assertEqual(direct.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict())

if __name__=='__main__':unittest.main(verbosity=2)
