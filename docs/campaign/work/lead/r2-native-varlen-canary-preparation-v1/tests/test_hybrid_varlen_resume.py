"""Actual Transformers 5.5 unequal-length denominator and hybrid resume proof."""
import copy,os,sys,tempfile,unittest
from pathlib import Path
PACKET=Path(__file__).resolve().parents[1];TRAINING=PACKET/'source/experiments/training';sys.path.insert(0,str(TRAINING))
import torch
from torch import nn
from torch.utils.data import SequentialSampler
from transformers import Trainer,TrainerCallback,TrainingArguments
from full_weight_optimizer import FullWeightOptimizerTrainerMixin,OptimizerConfig
from varlen_update_adapter import PackedOptimizerWindowDataset,VarlenUpdateTrainerMixin,logical_update_collator

class Attention(nn.Module):
 def __init__(self):super().__init__();self.q_proj=nn.Linear(4,4,bias=False)
class MLP(nn.Module):
 def __init__(self):super().__init__();self.gate_proj=nn.Linear(4,8,bias=False);self.up_proj=nn.Linear(4,8,bias=False);self.down_proj=nn.Linear(8,4,bias=False)
class Block(nn.Module):
 def __init__(self):super().__init__();self.self_attn=Attention();self.mlp=MLP()
class Body(nn.Module):
 def __init__(self):super().__init__();self.embed_tokens=nn.Embedding(17,4);self.layers=nn.ModuleList([Block()])
class TinyCausalAurora(nn.Module):
 accepts_loss_kwargs=True
 def __init__(self):super().__init__();self.model=Body();self.final_norm=nn.LayerNorm(4);self.lm_head=nn.Linear(4,17,bias=False)
 def forward(self,input_ids,labels,position_ids=None,packed_seq_lengths=None,num_items_in_batch=None,**_):
  assert num_items_in_batch is not None
  b=self.model.layers[0];x=self.model.embed_tokens(input_ids);x=b.self_attn.q_proj(x);x=x+b.mlp.down_proj(torch.relu(b.mlp.gate_proj(x))+b.mlp.up_proj(x));logits=self.lm_head(self.final_norm(x))
  numerator=nn.functional.cross_entropy(logits[:,:-1].reshape(-1,17),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
  return {'loss':numerator/num_items_in_batch,'logits':logits}
class Rows:
 def __init__(self,positions):self.positions=list(positions)
 def __len__(self):return len(self.positions)
 def __getitem__(self,i):
  p=self.positions[i];n=(3,6,4,8)[p%4];ids=[(p+j+2)%17 for j in range(n)];return {'input_ids':ids,'labels':[-100]+ids[1:],'attention_mask':[1]*n,'_draw_position':p,'_row_id':f'row-{p}'}
def collate(items):
 n=max(len(x['input_ids'])for x in items);ids=[];labels=[];pos=[]
 for x in items:
  k=len(x['input_ids']);ids.append(x['input_ids']+[1]*(n-k));labels.append(x['labels']+[-100]*(n-k));pos.append(x['_draw_position'])
 return {'input_ids':torch.tensor(ids),'labels':torch.tensor(labels),'_draw_position':torch.tensor(pos)}
def args(path,steps,ga):return TrainingArguments(output_dir=str(path),per_device_train_batch_size=1,gradient_accumulation_steps=ga,max_steps=steps,learning_rate=3e-4,lr_scheduler_type='constant_with_warmup',warmup_steps=1,max_grad_norm=0.,use_cpu=True,bf16=False,save_strategy='no',logging_strategy='no',eval_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,dataloader_num_workers=0,dataloader_pin_memory=False,train_sampling_strategy='sequential',ignore_data_skip=True,seed=73,data_seed=73)
class Stop(TrainerCallback):
 def __init__(self,step,stop=False):self.step=step;self.stop=stop
 def on_step_end(self,args,state,control,**_):
  if int(state.global_step)==self.step:control.should_save=True;control.should_training_stop=self.stop
  return control
class Dispatch:
 def create_optimizer(self):
  out=super().create_optimizer();assert self.full_weight_optimizer_manifest['arm']=='aurora_mix';return out
class Ordinary(Dispatch,FullWeightOptimizerTrainerMixin,Trainer):
 def __init__(self,*a,seen=None,denoms=None,**k):self.seen=seen;self.denoms=denoms;super().__init__(*a,**k)
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset or self.train_dataset)
 def compute_loss(self,model,inputs,*a,**kw):
  self.seen.extend(int(x)for x in inputs.pop('_draw_position').tolist());d=kw['num_items_in_batch'];self.denoms.append(int(d.item()if hasattr(d,'item')else d));return super().compute_loss(model,inputs,*a,**kw)
class Packed(VarlenUpdateTrainerMixin,Dispatch,FullWeightOptimizerTrainerMixin,Trainer):
 def __init__(self,*a,seen=None,denoms=None,**k):self.seen=seen;self.denoms=denoms;super().__init__(*a,**k)
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset or self.train_dataset)
 def before_varlen_update(self,positions,row_ids,denominator,physical_packs):self.seen.extend(positions);self.denoms.append(denominator)
def config():return OptimizerConfig(arm='aurora_mix',hidden_lr=3e-4,side_lr=3e-5,weight_decay=0.,ns_steps=2,aurora_K=1,stochastic_round_chunk_elements=128)
def build(root,name,state,positions,steps,packed,stop):
 model=TinyCausalAurora();model.load_state_dict(copy.deepcopy(state));seen=[];den=[]
 if packed:data=PackedOptimizerWindowDataset(Rows(positions),first_position=positions[0],token_cap=12,effective_batch=4);klass=Packed;co=logical_update_collator;ga=1
 else:data=Rows(positions);klass=Ordinary;co=collate;ga=4
 t=klass(model=model,args=args(root/name,steps,ga),train_dataset=data,data_collator=co,callbacks=[stop],seen=seen,denoms=den);t.full_weight_optimizer_config=config();return t,seen,den
def eq(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys()and all(eq(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(eq(x,y)for x,y in zip(a,b))
 return a==b
class Proof(unittest.TestCase):
 def test_ordinary_unequal_lengths_uses_one_global_shifted_token_denominator(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);torch.manual_seed(11);state=TinyCausalAurora().state_dict();ordinary,seen,den=build(root,'ordinary',state,range(4),1,False,Stop(1));ordinary.train()
   expected=sum(len(Rows(range(4))[i]['labels'])-1 for i in range(4));self.assertEqual(seen,list(range(4)));self.assertEqual(den,[expected]*4)
 def test_one_update_preserves_global_token_weighting_and_hybrid_dispatch(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);torch.manual_seed(17);state=TinyCausalAurora().state_dict();ordinary,os_,od=build(root,'ordinary',state,range(4),1,False,Stop(1));torch.manual_seed(91);ordinary.train();packed,ps,pd=build(root,'packed',state,range(4),1,True,Stop(1));torch.manual_seed(91);packed.train()
   expected=sum(len(Rows(range(4))[i]['labels'])-1 for i in range(4));self.assertEqual((os_,ps),(list(range(4)),list(range(4))));self.assertEqual((od[0],pd[0]),(expected,expected))
   for a,b in zip(ordinary.model.parameters(),packed.model.parameters()):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
   self.assertEqual(ordinary.lr_scheduler.state_dict(),packed.lr_scheduler.state_dict());self.assertEqual(ordinary.full_weight_optimizer_manifest['counts'],packed.full_weight_optimizer_manifest['counts'])
 def test_packed_hybrid_update_and_resume_are_exact(self):
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as td:
   root=Path(td);torch.manual_seed(19);state=TinyCausalAurora().state_dict();direct,ds,dd=build(root,'direct',state,range(8),2,True,Stop(2));direct.train();split,ss,sd=build(root,'split',state,range(8),2,True,Stop(1,True));split.train();cp=root/'split/checkpoint-1';resumed,rs,rd=build(root,'resumed',state,range(4,8),2,True,Stop(2));resumed.train(resume_from_checkpoint=str(cp))
   self.assertEqual((ds,ss,rs),(list(range(8)),list(range(4)),list(range(4,8))));self.assertEqual(dd,sd+rd);self.assertTrue(eq(direct.model.state_dict(),resumed.model.state_dict()));self.assertTrue(eq(direct.optimizer.state_dict(),resumed.optimizer.state_dict()));self.assertEqual(direct.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict());self.assertGreater(direct.full_weight_optimizer_manifest['counts']['aurora'],0);self.assertGreater(direct.full_weight_optimizer_manifest['counts']['adamw'],0)
if __name__=='__main__':unittest.main(verbosity=2)
