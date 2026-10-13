import copy, json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

SOURCE=Path(__file__).parents[1]/"source"/"experiments"/"training"
sys.path.insert(0,str(SOURCE))
from a100_distributed import (DistributedFullWeightCompositeOptimizer,
    global_supervised_denominator, validate_ddp_window, validate_gathered_window)
from full_weight_optimizer import OptimizerConfig, FullWeightCompositeOptimizer, inventory_full_weight

rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo")
assert world==8

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.scale=torch.nn.Parameter(torch.tensor([0.75,-0.25,0.5],dtype=torch.bfloat16))
    def forward(self,x): return self.scale.float().mul(x).sum()

cfg=OptimizerConfig(arm="adamw_anchor",hidden_lr=3e-3,side_lr=3e-3,
                    stochastic_round_chunk_elements=2)

def setup(step):
    torch.manual_seed(111)
    model=DDP(Tiny())
    inv,manifest=inventory_full_weight(model.module,"adamw_anchor")
    opt=DistributedFullWeightCompositeOptimizer(inv,manifest,cfg,rounding_base_seed=3407,
                                                completed_global_step=step,world_size=8)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
    return model,opt,sched

def update(model,opt,sched,local_update):
    pos=[384+local_update*16+rank*2,385+local_update*16+rank*2]
    validate_ddp_window(pos,initial_cursor=384,local_update=local_update,rank=rank)
    gathered=[None]*world; dist.all_gather_object(gathered,pos)
    validate_gathered_window(gathered,first=384+local_update*16)
    counts=torch.tensor([sum(1+(p%3) for p in pos)],dtype=torch.int64)
    denom=global_supervised_denominator(counts).float()
    numerator=sum((model(torch.tensor([1.0,p%5/7,0.25]))-(p%4)/10).square()*(1+(p%3)) for p in pos)
    loss=numerator/denom*world
    loss.backward()
    rng=torch.get_rng_state().clone(); opt.step(); assert torch.equal(rng,torch.get_rng_state())
    sched.step(); opt.zero_grad(set_to_none=True)

direct,dopt,dsched=setup(90)
update(direct,dopt,dsched,0)
checkpoint={"model":copy.deepcopy(direct.module.state_dict()),"optimizer":copy.deepcopy(dopt.state_dict()),
            "scheduler":copy.deepcopy(dsched.state_dict()),"rng":torch.get_rng_state().clone()}
update(direct,dopt,dsched,1)
expected=direct.module.scale.detach().clone(); expected_opt=dopt.state_dict()

resumed,ropt,rsched=setup(91)
resumed.module.load_state_dict(checkpoint["model"])
ropt.load_state_dict(checkpoint["optimizer"])
rsched.load_state_dict(checkpoint["scheduler"]); torch.set_rng_state(checkpoint["rng"])
update(resumed,ropt,rsched,1)
assert torch.equal(expected,resumed.module.scale)
assert expected_opt["_sepalith_distributed"]["completed_global_step"]==ropt.completed_global_step==92

replicas=[torch.empty_like(expected) for _ in range(world)]
dist.all_gather(replicas,resumed.module.scale.detach())
assert all(torch.equal(replicas[0],x) for x in replicas)

# A later single-device optimizer accepts the same base state after the DDP metadata is removed.
plain=Tiny(); inv,manifest=inventory_full_weight(plain,"adamw_anchor")
popt=FullWeightCompositeOptimizer(inv,manifest,cfg)
portable=dict(ropt.state_dict()); portable.pop("_sepalith_distributed")
popt.load_state_dict(portable)
assert all(v.dtype==torch.float32 for state in popt.state.values() for k,v in state.items()
           if k in ("exp_avg","exp_avg_sq"))
if rank==0: print(json.dumps({"status":"pass","world_size":world,"steps":[91,92],
    "global_windows":[list(range(384,400)),list(range(400,416))],
    "replicas_exact":True,"resume_exact":True,"single_device_optimizer_load":True}))
dist.destroy_process_group()
