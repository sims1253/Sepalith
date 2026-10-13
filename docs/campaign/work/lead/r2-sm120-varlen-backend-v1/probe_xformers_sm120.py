#!/usr/bin/env python3
"""Root-run, kernel-only xFormers block-diagonal diagnostic for sm120."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from backend_contract import validate


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.report.exists(): raise ValueError("report path must be fresh")
    static = validate()
    import torch
    import xformers
    import xformers.ops as xops
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one root-leased CUDA device")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("diagnostic is pinned to sm120")
    if not str(Path(xformers.__file__).resolve()).startswith(str(Path(static["overlay"]["path"]).resolve())):
        raise RuntimeError("xformers did not import from the isolated cu130 overlay")
    torch.manual_seed(20260915); torch.cuda.manual_seed_all(20260915)
    device = torch.device("cuda")
    lengths = [127, 129]
    bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths)
    q = torch.randn((1, sum(lengths), 16, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    out = xops.memory_efficient_attention(q, k, v, attn_bias=bias, p=0.0)
    loss = out.float().square().mean(); loss.backward(); torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not all(torch.isfinite(x).all() for x in (out, q.grad, k.grad, v.grad)):
        raise RuntimeError("nonfinite xformers output or gradient")
    with torch.no_grad():
        # Change keys and values in member zero. Member one's queries must be
        # unable to observe either change when the block diagonal mask works.
        changed_k = k.detach().clone(); changed_v = v.detach().clone()
        changed_k[:, 1] += 1; changed_v[:, 1] += 1
        changed_out = xops.memory_efficient_attention(q.detach(), changed_k, changed_v, attn_bias=bias, p=0.0)
        isolation = float((out[:, lengths[0]:] - changed_out[:, lengths[0]:]).abs().max())
    if isolation != 0.0: raise RuntimeError("cross-document attention leakage")
    report = {
        "schema": "sepalith.sft11.sm120_xformers_kernel_probe.v1",
        "status": "kernel_pass_model_parity_still_required",
        "device_capability": [12, 0], "dtype": "bfloat16", "head_dim": 128,
        "logical_lengths": lengths, "forward_backward_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cross_document_max_abs_delta": isolation,
        "finite_forward_backward": True, "no_model_loaded": True, "no_optimizer_step": True,
        "static_binding": static,
    }
    write_atomic(a.report, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
