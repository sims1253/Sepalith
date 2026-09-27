"""Small tensor-comparison helpers for the proposed actual-model CUDA probe."""
from __future__ import annotations
import math

def relative_metrics(reference, candidate, torch):
 if reference.shape!=candidate.shape: raise ValueError('comparison shapes differ')
 if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()): raise ValueError('nonfinite comparison')
 a,b=reference.detach().float(),candidate.detach().float(); d=a-b
 den=max(float(torch.linalg.vector_norm(a)),float(torch.linalg.vector_norm(b)),1e-12)
 return {'max_abs':float(d.abs().max()),'relative_l2':float(torch.linalg.vector_norm(d))/den,'elements':a.numel()}

def split_concat_projection(linear, members, torch):
 if not members or any(x.ndim!=2 or x.shape[1]!=members[0].shape[1] for x in members): raise ValueError('projection members differ')
 split=torch.cat([linear(x) for x in members],dim=0)
 combined=linear(torch.cat(members,dim=0))
 return relative_metrics(split,combined,torch)

def threshold_decision(result, ordinary_batch_control):
 """Conservative gate tied to a measured ordinary batching control."""
 required=('isolation_max_abs','repeat_relative_l2','gradient_median_relative_l2','gradient_max_relative_l2','optimizer_update_relative_l2','optimizer_update_cosine')
 if any(k not in result for k in required): raise ValueError('candidate metrics incomplete')
 if ordinary_batch_control is None or any(k not in ordinary_batch_control for k in ('gradient_median_relative_l2','gradient_max_relative_l2')): raise ValueError('ordinary batch control missing')
 vals=list(result.values())+list(ordinary_batch_control.values())
 if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals): raise ValueError('nonfinite metric')
 ceilings={'gradient_median_relative_l2':max(0.01,1.25*ordinary_batch_control['gradient_median_relative_l2']),
           'gradient_max_relative_l2':max(0.05,1.25*ordinary_batch_control['gradient_max_relative_l2'])}
 passed=(result['isolation_max_abs']==0 and result['repeat_relative_l2']<=1e-7 and
         result['gradient_median_relative_l2']<=ceilings['gradient_median_relative_l2'] and
         result['gradient_max_relative_l2']<=ceilings['gradient_max_relative_l2'] and
         result['optimizer_update_relative_l2']<=0.02 and result['optimizer_update_cosine']>=0.999)
 return {'pass':passed,'ceilings':ceilings}
