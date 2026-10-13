#!/usr/bin/env python3
"""Root-admitted no-model attention probe; never run during CPU preparation."""
import argparse,hashlib,json,os,random,statistics
from pathlib import Path
PACKET=Path(__file__).resolve().parent
PROFILES={
 "multi_7260":[3356,809,2485,548,62],
 "multi_16040":[3478,1191,4972,647,291,4287,1174],
 "single_16384":[16384],
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def req(value,message):
 if not value:raise ValueError(message)
def numeric_metric(reference,candidate):
 delta=reference.float()-candidate.float();rms=float(delta.square().mean().sqrt());base=float(reference.float().square().mean().sqrt())
 return {'max_abs':float(delta.abs().max()),'rms':rms,'relative_rms':rms/max(base,1e-12)}
def profiler_capture(torch,invoke,kind,inferred):
 """Untimed CPU-activity capture. Inferred names never enter observed fields."""
 try:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,profile_memory=False)as profile:value=invoke(kind,'scalar',None)
  del value
  keys=sorted({event.key for event in profile.key_averages() if isinstance(event.key,str)})
  req(keys,'profiler returned no operator keys')
  return {'status':'captured','activity':'CPU','actual_operator_keys':keys,'actual_aten_operator_keys':[key for key in keys if key.startswith('aten::')],'actual_attention_operator_keys':[key for key in keys if any(word in key.lower()for word in ('attention','flash','fmha'))],'inferred_operator_names':[]}
 except Exception as error:
  return {'status':'capture_unavailable','activity':'CPU','actual_operator_keys':[],'actual_aten_operator_keys':[],'actual_attention_operator_keys':[],'inferred_operator_names':list(inferred),'error_type':type(error).__name__}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();admission=json.loads(args.admission.read_text());manifest=PACKET/'source-manifest.json'
 req(admission.get('schema')=='sepalith.sft11.sm120-varlen-probe-admission.v2' and admission.get('status')=='admitted' and admission.get('cuda_authorized')is True and admission.get('packet_source_manifest_sha256')==sha(manifest) and admission.get('maximum_seconds')==600,'probe is not root-admitted');req(not args.output.exists(),'fresh output required')
 import torch
 from torch_varlen_candidate import torch_varlen_attention
 req(torch.cuda.is_available()and torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0),'exact single sm120 device required')
 import xformers.ops as xops
 from xformers.ops.fmha.common import Inputs
 from xformers.ops.fmha.dispatch import _dispatch_fw,_dispatch_bw
 req(str(Path(__import__('xformers').__file__).resolve()).startswith('/mnt/e/sepalith/campaign-20260915/build-work/sm120-varlen-v1/cu130-overlay/'),'xformers overlay differs')
 torch.manual_seed(20260915);random.seed(20260915);device=torch.device('cuda');results=[]
 for profile_name,lengths in PROFILES.items():
  total=sum(lengths);base=[torch.randn((1,heads,total,128),device=device,dtype=torch.bfloat16)for heads in (16,2,2)]
  def forward(kind,tensors):
   q,k,v=tensors
   if kind=='xformers_fa2':
    bias=xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths);qt=q.transpose(1,2);kt=k.transpose(1,2).view(1,total,2,1,128).expand(1,total,2,8,128).reshape(1,total,16,128);vt=v.transpose(1,2).view(1,total,2,1,128).expand(1,total,2,8,128).reshape(1,total,16,128);inp=Inputs(qt,kt,vt,attn_bias=bias);fw=_dispatch_fw(inp,True);bw=_dispatch_bw(inp,fw.VARLEN_LSE_PACKED);return xops.memory_efficient_attention(qt,kt,vt,attn_bias=bias),(fw.NAME,bw.NAME)
   return torch_varlen_attention(q,k,v,lengths),None
  def invoke(kind,backward_mode='scalar',cotangent=None,tensors=None):
   values=[x.detach().clone().requires_grad_(True)for x in (base if tensors is None else tensors)];out,dispatch=forward(kind,values)
   if backward_mode=='scalar':(out.sum(dtype=torch.float32)/out.numel()).backward()
   elif backward_mode=='cotangent':req(cotangent is not None and cotangent.shape==out.shape,'cotangent layout differs');out.backward(cotangent)
   else:raise ValueError('backward mode differs')
   torch.cuda.synchronize();return out.detach(),[x.grad.detach()for x in values],dispatch
  reference,reference_grads,xdispatch=invoke('xformers_fa2');candidate,candidate_grads,_=invoke('torch_varlen');scalar={'output':numeric_metric(reference,candidate),'gradients':{name:numeric_metric(x,y)for name,x,y in zip(('q','k','v'),reference_grads,candidate_grads)}}
  generator=torch.Generator(device=device);generator.manual_seed(20260915+total);cotangent=torch.randn(candidate.shape,device=device,dtype=torch.bfloat16,generator=generator);_,random_reference_grads,_=invoke('xformers_fa2','cotangent',cotangent);_,random_candidate_grads,_=invoke('torch_varlen','cotangent',cotangent);random_gradients={name:numeric_metric(x,y)for name,x,y in zip(('q','k','v'),random_reference_grads,random_candidate_grads)}
  if len(lengths)>1:
   document_index=len(lengths)//2;start=sum(lengths[:document_index]);end=start+lengths[document_index];perturbed=[x.detach().clone()for x in base];perturbed[0][:,:,start:end,:]+=0.125;perturbed[1][:,:,start:end,:]-=0.0625;perturbed[2][:,:,start:end,:]+=0.03125
   isolation={}
   for kind,baseline in (('xformers_fa2',reference),('torch_varlen',candidate)):
    changed,_=forward(kind,perturbed);outside=torch.cat((baseline[:,:start],baseline[:,end:]),dim=1);changed_outside=torch.cat((changed[:,:start],changed[:,end:]),dim=1);inside_delta=float((baseline[:,start:end]-changed[:,start:end]).float().abs().max());outside_delta=float((outside-changed_outside).float().abs().max());isolation[kind]={'perturbed_document_index':document_index,'token_range':[start,end],'outside_max_abs':outside_delta,'inside_max_abs':inside_delta,'passed':outside_delta==0.0 and inside_delta>0.0};req(isolation[kind]['passed'],kind+' document isolation failed');del changed
  else:isolation={'status':'not_applicable_single_document'}
  inferred_x=list(xdispatch);inferred_torch=['aten::_flash_attention_forward','aten::_flash_attention_backward'];operator_capture={'xformers_fa2':profiler_capture(torch,invoke,'xformers_fa2',inferred_x),'torch_varlen':profiler_capture(torch,invoke,'torch_varlen',inferred_torch),'dispatcher_selected_implementation_names':inferred_x,'implementation_declared_expected_names':{'torch_varlen':inferred_torch},'timing_included':False}
  del reference,reference_grads,candidate,candidate_grads,random_reference_grads,random_candidate_grads,cotangent
  raw={kind:{'milliseconds':[],'peak_allocated_bytes':[]}for kind in ('xformers_fa2','torch_varlen')}
  for repetition in range(7):
   order=('xformers_fa2','torch_varlen')if repetition%2==0 else('torch_varlen','xformers_fa2')
   for kind in order:
    torch.cuda.reset_peak_memory_stats();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record();value=invoke(kind);end_event.record();torch.cuda.synchronize()
    if repetition>=2:raw[kind]['milliseconds'].append(start_event.elapsed_time(end_event));raw[kind]['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated())
    del value
  timings={kind:{'milliseconds':value['milliseconds'],'median_milliseconds':statistics.median(value['milliseconds']),'peak_allocated_bytes':max(value['peak_allocated_bytes'])}for kind,value in raw.items()}
  results.append({'profile':profile_name,'lengths':lengths,'total_tokens':total,'operator_capture':operator_capture,'numerics':{'scalar_loss':scalar,'seeded_random_cotangent_gradients':random_gradients,'cotangent_seed':20260915+total,'document_isolation':isolation},'timing':timings,'xformers_over_torch_speedup':timings['xformers_fa2']['median_milliseconds']/timings['torch_varlen']['median_milliseconds']})
 report={'schema':'sepalith.sft11.sm120-varlen-attention-probe.v2','status':'measurement_only_no_training_admission','device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'torch':torch.__version__,'profiler_contract':{'activity':'CPU','untimed':True,'actual_keys_distinct_from_inferred_names':True,'cupti_required':False},'timing_contract':{'warmups_each':2,'alternating_timed_repetitions_each':5,'profiler_excluded':True},'results':results}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x')as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
