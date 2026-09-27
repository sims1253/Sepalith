#!/usr/bin/env python3
"""Root-run, kernel-only native FlexAttention fallback diagnostic for sm120."""
from __future__ import annotations

import argparse, json, os, tempfile, time
from pathlib import Path


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as s:
            json.dump(value, s, indent=2, sort_keys=True); s.write("\n"); s.flush(); os.fsync(s.fileno())
        os.replace(name, path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    if a.report.exists(): raise ValueError("report path must be fresh")
    import torch
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("requires one root-leased CUDA device")
    if torch.cuda.get_device_capability()!=(12,0): raise RuntimeError("diagnostic is pinned to sm120")
    lengths=(127,129); total=sum(lengths); boundary=lengths[0]
    def same_segment_causal(_b,_h,q,k):
        same=((q < boundary) == (k < boundary)); return same & (k <= q)
    block=create_block_mask(same_segment_causal,B=None,H=None,Q_LEN=total,KV_LEN=total,device="cuda")
    torch.manual_seed(20260915); torch.cuda.manual_seed_all(20260915)
    q=torch.randn((1,16,total,128),device="cuda",dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn((1,2,total,128),device="cuda",dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    out=flex_attention(q,k,v,block_mask=block,enable_gqa=True)
    out.float().square().mean().backward(); torch.cuda.synchronize(); elapsed=time.monotonic()-started
    if not all(torch.isfinite(x).all() for x in (out,q.grad,k.grad,v.grad)): raise RuntimeError("nonfinite flex output or gradient")
    with torch.no_grad():
        changed_k=k.detach().clone(); changed_v=v.detach().clone()
        changed_k[:,:,1]+=1; changed_v[:,:,1]+=1
        changed_out=flex_attention(q.detach(),changed_k,changed_v,block_mask=block,enable_gqa=True)
        isolation=float((out[:,:,boundary:]-changed_out[:,:,boundary:]).abs().max())
    if isolation != 0.0: raise RuntimeError("cross-document attention leakage")
    report={"schema":"sepalith.sft11.sm120_flex_kernel_probe.v1","status":"kernel_pass_requires_unsloth_integration_and_model_parity","device_capability":[12,0],"dtype":"bfloat16","head_dim":128,"query_heads":16,"kv_heads":2,"logical_lengths":list(lengths),"forward_backward_seconds":elapsed,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"cross_document_max_abs_delta":isolation,"finite_forward_backward":True,"no_model_loaded":True,"no_optimizer_step":True}
    write_atomic(a.report,report); print(json.dumps(report,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
