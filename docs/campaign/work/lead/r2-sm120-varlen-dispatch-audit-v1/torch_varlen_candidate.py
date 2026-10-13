#!/usr/bin/env python3
"""Isolated candidate adapter for Torch 2.11 native variable-length attention."""
from __future__ import annotations


def validate_layout(q_shape, k_shape, v_shape, lengths, *, q_heads=16, kv_heads=2, head_dim=128):
    if tuple(q_shape[:2]) != (1, q_heads) or tuple(k_shape[:2]) != (1, kv_heads) or tuple(v_shape[:2]) != (1, kv_heads):
        raise ValueError("GQA head layout differs")
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("Q/K/V must use [1,H,T,D]")
    total = q_shape[2]
    if k_shape[2] != total or v_shape[2] != total or q_shape[3] != head_dim or k_shape[3] != head_dim or v_shape[3] != head_dim:
        raise ValueError("token or head-dimension layout differs")
    if not lengths or any(type(x) is not int or x <= 0 for x in lengths) or sum(lengths) != total:
        raise ValueError("sequence boundary coverage differs")
    if q_heads % kv_heads:
        raise ValueError("query heads not divisible by KV heads")
    return {"total_tokens": total, "max_seqlen": max(lengths), "gqa_groups": q_heads // kv_heads}


def torch_varlen_attention(q, k, v, lengths):
    """Return [1,T,Hq,D], preserving native GQA and exact causal boundaries."""
    import torch
    from torch.nn.attention.varlen import varlen_attn
    info = validate_layout(q.shape, k.shape, v.shape, list(lengths))
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("candidate is CUDA-only")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("candidate requires common BF16 Q/K/V")
    lens = torch.tensor(lengths, dtype=torch.int32, device=q.device)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=q.device)
    torch.cumsum(lens, dim=0, out=cu[1:])
    # Torch documents (-1, 0) as causal. It forwards these exact cumulative
    # boundaries to aten::_flash_attention_{forward,backward}.
    out = varlen_attn(
        q.transpose(1, 2).reshape(info["total_tokens"], q.shape[1], q.shape[3]),
        k.transpose(1, 2).reshape(info["total_tokens"], k.shape[1], k.shape[3]),
        v.transpose(1, 2).reshape(info["total_tokens"], v.shape[1], v.shape[3]),
        cu, cu, info["max_seqlen"], info["max_seqlen"], window_size=(-1, 0),
    )
    if tuple(out.shape) != (info["total_tokens"], q.shape[1], q.shape[3]):
        raise RuntimeError("Torch varlen output layout differs")
    return out.view(1, info["total_tokens"], q.shape[1], q.shape[3])
