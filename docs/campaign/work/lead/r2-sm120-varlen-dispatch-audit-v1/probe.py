#!/usr/bin/env python3
"""Root-admitted, no-model attention numerics/timing probe. This file was not run during preparation."""
import argparse,hashlib,json,os,random,statistics
from pathlib import Path
PACKET=Path(__file__).resolve().parent
PROFILES={
 "multi_7260":[3356,809,2485,548,62],
 "multi_16040":[3478,1191,4972,647,291,4287,1174],
 "single_16384":[16384],
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--admission',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 adm=json.loads(a.admission.read_text());manifest=PACKET/'source-manifest.json'
 if adm.get('schema')!='sepalith.sft11.sm120-varlen-probe-admission.v1' or adm.get('status')!='admitted' or adm.get('cuda_authorized') is not True or adm.get('packet_source_manifest_sha256')!=sha(manifest) or adm.get('maximum_seconds')!=600:raise ValueError('probe is not root-admitted')
 if a.output.exists():raise ValueError('fresh output required')
 import torch
 from torch_varlen_candidate import torch_varlen_attention
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('exact single sm120 device required')
 import xformers.ops as xops
 from xformers.ops.fmha.common import Inputs
 from xformers.ops.fmha.dispatch import _dispatch_fw,_dispatch_bw
 if not str(Path(__import__('xformers').__file__).resolve()).startswith('/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-overlay/'):raise RuntimeError('xformers overlay differs')
 torch.manual_seed(20260915);random.seed(20260915);device=torch.device('cuda');results=[]
 for name,lengths in PROFILES.items():
  total=sum(lengths);base=[torch.randn((1,h,total,128),device=device,dtype=torch.bfloat16) for h in (16,2,2)]
  def current(q,k,v):
   bias=xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths)
   qt=q.transpose(1,2);kt=k.transpose(1,2).view(1,total,2,1,128).expand(1,total,2,8,128).reshape(1,total,16,128);vt=v.transpose(1,2).view(1,total,2,1,128).expand(1,total,2,8,128).reshape(1,total,16,128)
   inp=Inputs(qt,kt,vt,attn_bias=bias);fw=_dispatch_fw(inp,True);bw=_dispatch_bw(inp,fw.VARLEN_LSE_PACKED)
   return xops.memory_efficient_attention(qt,kt,vt,attn_bias=bias),(fw.NAME,bw.NAME)
  def invoke(kind):
   q,k,v=[x.detach().clone().requires_grad_(True) for x in base]
   if kind=='xformers_fa2':out,ops=current(q,k,v)
   else:out=torch_varlen_attention(q,k,v,lengths);ops=('aten::_flash_attention_forward','aten::_flash_attention_backward')
   # Identical scalar exposes all output positions and all Q/K/V gradients.
   loss=out.sum(dtype=torch.float32)/out.numel();loss.backward();torch.cuda.synchronize()
   return out.detach(),[x.grad.detach() for x in (q,k,v)],ops
  ref,refg,ops=invoke('xformers_fa2');cand,candg,cops=invoke('torch_varlen')
  def metric(x,y):
   delta=(x.float()-y.float());rms=float(delta.square().mean().sqrt());base=float(x.float().square().mean().sqrt())
   return {'max_abs':float(delta.abs().max()),'rms':rms,'relative_rms':rms/max(base,1e-12)}
  numeric={'output':metric(ref,cand),'gradients':{n:metric(x,y) for n,x,y in zip(('q','k','v'),refg,candg)}}
  del ref,refg,cand,candg
  raw={k:{'milliseconds':[],'peak_allocated_bytes':[]} for k in ('xformers_fa2','torch_varlen')}
  for i in range(7):
   order=('xformers_fa2','torch_varlen') if i%2==0 else ('torch_varlen','xformers_fa2')
   for kind in order:
    torch.cuda.reset_peak_memory_stats();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();value=invoke(kind);end.record();torch.cuda.synchronize()
    if i>=2:raw[kind]['milliseconds'].append(start.elapsed_time(end));raw[kind]['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated())
    del value
  timings={k:{'milliseconds':v['milliseconds'],'median_milliseconds':statistics.median(v['milliseconds']),'peak_allocated_bytes':max(v['peak_allocated_bytes'])} for k,v in raw.items()}
  results.append({'profile':name,'lengths':lengths,'total_tokens':total,'xformers_selected_ops':ops,'torch_ops':cops,'numerics':numeric,'timing':timings,'xformers_over_torch_speedup':timings['xformers_fa2']['median_milliseconds']/timings['torch_varlen']['median_milliseconds']})
 report={'schema':'sepalith.sft11.sm120-varlen-attention-probe.v1','status':'measurement_only_no_training_admission','device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'torch':torch.__version__,'results':results}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
