"""Unchanged task trainer plus one explicit, non-updating backward profile."""
import hashlib, heapq, json, sys, time
from pathlib import Path
import campaign_sft
import campaign_task_sft
from campaign_checkpoint import preserve_random_state,write_json

def backward_profile(trainer,model,*,device_stats=None):
 import torch
 started=time.monotonic()
 if device_stats is None:
  if not torch.cuda.is_available():raise ValueError('cloud backward profile requires CUDA')
  device_stats=lambda:{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 if trainer.args.per_device_train_batch_size!=2:raise ValueError('cloud profile microbatch differs')
 indices=heapq.nlargest(2,range(len(trainer.train_dataset)),key=lambda i:len(trainer.train_dataset[i]['input_ids']))
 if len(indices)!=2:raise ValueError('two real dataset rows required')
 rows=[dict(trainer.train_dataset[i]) for i in indices]
 batch=trainer._prepare_inputs(trainer.data_collator(rows));denominator=batch['labels'][:,1:].ne(-100).sum()
 parameters=[p for p in model.parameters() if p.requires_grad]
 versions=[p._version for p in parameters]
 if any(p.grad is not None for p in parameters):raise ValueError('profile started with gradients')
 configs=[(m,m.config.use_cache) for m in model.modules() if hasattr(getattr(m,'config',None),'use_cache')]
 try:
  with preserve_random_state(model),torch.enable_grad():
   model.train()
   for module,_ in configs:module.config.use_cache=False
   with trainer.compute_loss_context_manager():loss=trainer.compute_loss(model,dict(batch),num_items_in_batch=denominator)
   if loss.numel()!=1 or not bool(torch.isfinite(loss)):raise ValueError('profile loss not finite')
   loss.backward()
   gradients=[p.grad for p in parameters if p.grad is not None]
   if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('profile gradients missing or nonfinite')
   result={'status':'pass','selected_dataset_indices':indices,'real_input_lengths':[len(r['input_ids']) for r in rows],
    'maximum_real_dataset_length':len(rows[0]['input_ids']),'target_tokens':int(denominator),'loss':float(loss.detach()),
    'trainable_gradients_observed':len(gradients),'optimizer_updates':0,'input_sha256':hashlib.sha256(json.dumps([r['input_ids'] for r in rows],separators=(',',':')).encode()).hexdigest(),
    'memory':device_stats(),'memory_scope':'Process peak includes model load, startup gate and worst-real-length microbatch2 backward; optimizer-state allocation still belongs to first update.'}
 finally:
  model.zero_grad(set_to_none=True)
  for module,value in configs:module.config.use_cache=value
 if any(p._version!=version or p.grad is not None for p,version in zip(parameters,versions)):raise ValueError('backward profile changed weights or left gradients')
 result['seconds']=time.monotonic()-started;return result

def main():
 recipe=json.loads(Path(sys.argv[1]).read_text());original=campaign_sft.target_only_startup_gate
 def gate(trainer,model,dataloader,**kwargs):
  result=original(trainer,model,dataloader,**kwargs)
  profile=backward_profile(trainer,model)
  write_json(Path(recipe['output_dir'])/'cloud-backward-profile.json',profile)
  return result
 campaign_sft.target_only_startup_gate=gate
 try:campaign_task_sft.run(recipe)
 finally:campaign_sft.target_only_startup_gate=original
if __name__=='__main__':main()
