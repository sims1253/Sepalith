import json,sys,pathlib,os
S=pathlib.Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/runner-r2-cpt-v1/snapshots/8df4c01e161fddb4f0e3c6d3b7b0899c6683b51971253806f9455dea66ec2b1d/source');sys.path[:0]=[str(S/'experiments/training'),str(S/'packages/sepalith/src')]
from campaign_cpt import preflight
from campaign_cpt_data import causal_lm_collator,verify_causal_batch
from datasets import Dataset
from torch.utils.data import DataLoader,SequentialSampler
import torch
torch.set_num_threads(1)
r=json.loads(pathlib.Path(sys.argv[1]).read_text());c=preflight(r,verify_model_files=True);rows=c['train_rows'];ix={x['id']:i for i,x in enumerate(rows)};draws=[ix[x] for x in c['schedule']['row_ids']];raw=[{k:x[k] for k in ['input_ids','labels','attention_mask']} for x in rows];ds=Dataset.from_list(raw).select(draws);loader=DataLoader(ds,batch_size=2,sampler=SequentialSampler(ds),collate_fn=causal_lm_collator);den=verify_causal_batch(next(iter(loader)),[raw[i] for i in draws[:2]]);print(json.dumps({'status':'actual_frozen_preflight_Dataset_SequentialSampler_collator_pass','first_loss_tokens':den,'draws':len(draws),'CUDA_initialized':torch.cuda.is_initialized()}));assert not torch.cuda.is_initialized()
