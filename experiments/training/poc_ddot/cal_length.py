#!/usr/bin/env python3
"""Task 5 baseline (b): CAL-style length search on the POC-DIFF twin.

The honest non-OT control for length handling (pre-registered in the
DDOT plan): NO retrain — one first-step denoising pass over the full max
window, read the per-slot confidence, pick the span length from it.
After CAL (arXiv:2602.00476): diffusion models can approximate optimal
infilling lengths implicitly via first-step denoising confidence.

Pre-registered v1 rule (kept deliberately simple, frozen before any eval):
  - c_l = max token prob at window slot l (temperature 0), in window order;
  - p_empty = P([EMPTY]) at window slot 0 — the length-0 candidate;
  - peak = max(p_empty, max_l c_l);
  - if p_empty == peak -> L = 0 (the empty span wins);
  - else L = the largest prefix whose LAST slot still has c_{L-1} >=
    peak / 2 ("plateau extent at half-peak confidence").

Batched over prompt rows; context positions are never written; one
forward per batch (that is the point — length costs one pass, not a
search over L re-samples).
"""
import torch

from experiments.training.poc_diff import EMPTY_ID
from experiments.training.poc_diff.sample import _chunked_probs


@torch.no_grad()
def pick_length(model, prompt_ids, window, temperature=0.0, chunk=1024):
    """First-step-confidence length pick. Returns list[int] length per row
    (0 = empty span). model: poc_diff MDGQA in eval mode."""
    device = next(model.parameters()).device
    B = len(prompt_ids)
    T_ctx = max(len(p) for p in prompt_ids)
    T = T_ctx + window
    x = torch.full((B, T), 1, dtype=torch.long, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, p in enumerate(prompt_ids):
        x[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        valid[b, :len(p)] = True
    span_rows = torch.arange(T_ctx, T, device=device)
    x[:, T_ctx:] = model.mask_id
    valid[:, T_ctx:] = True

    h = model.trunk(x, probe=False, attn_mask=valid[:, None, None, :])
    win_h = h[:, T_ctx:, :]                      # (B, W, d)
    # temperature 0 = greedy: read RAW probs (dividing by a floor of 1e-6
    # saturates the softmax and flattens the confidence signal)
    probs = _chunked_probs(win_h.reshape(-1, win_h.size(-1)),
                           model.embed.weight, chunk=chunk,
                           temperature=temperature if temperature > 0 else 1.0)
    probs = probs.reshape(B, window, -1)
    c = probs.max(dim=-1).values                 # (B, W) per-slot confidence
    empty_id = getattr(model, "empty_id", EMPTY_ID)
    p_empty = probs[:, 0, empty_id]              # length-0 candidate

    lengths = []
    for b in range(B):
        peak = max(float(p_empty[b]), float(c[b].max()))
        if float(p_empty[b]) >= peak:
            lengths.append(0)
            continue
        thresh = peak / 2.0
        L = 0
        for l in range(window, 0, -1):
            if float(c[b, l - 1]) >= thresh:
                L = l
                break
        lengths.append(L)
    return lengths
