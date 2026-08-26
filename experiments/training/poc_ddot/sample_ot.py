#!/usr/bin/env python3
"""Task 5: joint (value, position) span sampler — POC-DDOT.

Samples values AND positions together over a fixed MAX window, then snaps
positions to token slots at the final step — spans grow/shrink/re-anchor
instead of being pinned to the window (docs/research/
poc-ddot-ot-coupling-plan-2026-08-26.md Task 5). The OT coupling itself
is a TRAINING-time mechanism (objective_ot.py); per the plan, sampling is
joint denoising + snap, with no Sinkhorn in the loop.

Mechanism (pre-registered v1):
  - iterate `steps` remasking passes exactly like poc_diff's sampler
    (freeze the top-[open/steps] confident window slots per pass; context
    positions are NEVER written);
  - every frozen slot also records its predicted slot coordinate from the
    model's position head (contract: `model.predict_positions(h)` maps
    trunk features (B,T,d) -> (B,T) slot predictions in [0,1]; the Task-4
    OT twin trains this head with objective_ot.position_mse);
  - final step: snap_slots() quantizes predicted coordinates to the window
    grid, merges collisions keeping the higher-confidence token — so the
    SPAN LENGTH emerges from how the predicted positions cluster (dense
    cluster -> short span, spread -> long), and the absolute coordinates
    re-anchor the insertion point.

decode_ot_span() follows poc_diff's convention: predicted [EMPTY] tokens
strip before decoding; all-[EMPTY] decodes to ''.

Returns the poc_diff sample_spans dict shape (pred_ids padded with eos,
conf, n_fwd, latency_ms) plus `slots` (B, Lmax) long — snapped slot index
per predicted token, 0-padded — and `raw_positions` for telemetry.
"""
import torch
import torch.nn.functional as F

try:
    from . import ot_coupling as OT          # package import (unused v1; kept
except ImportError:                          # for the Task-5 eval coupling hook)
    import ot_coupling as OT

from experiments.training.poc_diff import EMPTY_ID
from experiments.training.poc_diff.sample import _chunked_probs


def snap_slots(values, positions, conf, window):
    """Final-step snap: positions -> token slots.

    slot = round(p * (window-1)); tokens colliding on a slot merge, the
    higher-confidence one wins; output ordered by slot (the span's token
    order). Returns dict(ids, slots).
    """
    if len(values) == 0:
        return dict(ids=[], slots=[])
    taken = {}
    for v, p, c in zip(values, positions, conf):
        sl = int(round(float(p) * (window - 1)))
        sl = max(0, min(window - 1, sl))
        if sl not in taken or c > taken[sl][1]:
            taken[sl] = (v, c)
    slots_sorted = sorted(taken)
    return dict(ids=[taken[sl][0] for sl in slots_sorted], slots=slots_sorted)


def decode_ot_span(region_ids, tok=None):
    """Region ids -> span text, poc_diff convention ([EMPTY] strips;
    all-[EMPTY] -> '')."""
    ids = [i for i in region_ids if i != EMPTY_ID]
    if not ids:
        return ""
    return tok.decode(ids) if tok is not None else " ".join(map(str, ids))


@torch.no_grad()
def sample_ot_spans(model, prompt_ids, window, steps, temperature=0.0,
                    generator=None, chunk=1024):
    """Joint value+position sampling over a max window; snap at the end.

    model: the OT twin in eval mode — poc_diff's MDGQA contract plus
    `predict_positions(h) -> (B, T)` (Task 4 trains it). prompt_ids:
    list[list[int]]. window: max span slots (<= 256). steps/temperature/
    generator/chunk: as poc_diff.sample.sample_spans.
    """
    device = next(model.parameters()).device
    B = len(prompt_ids)
    T_ctx = max(len(p) for p in prompt_ids)
    T = T_ctx + window
    x = torch.full((B, T), 1, dtype=torch.long, device=device)  # eos pad
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, p in enumerate(prompt_ids):
        x[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        valid[b, :len(p)] = True
    span_pos = torch.zeros(B, T, dtype=torch.bool, device=device)
    span_pos[:, T_ctx:] = True
    x[:, T_ctx:] = model.mask_id
    valid[:, T_ctx:] = True
    attn_mask = valid[:, None, None, :]

    per_step = max(1, -(-window // steps))  # ceil
    frozen = torch.zeros(B, T, dtype=torch.bool, device=device)
    conf_out = torch.zeros(B, T, device=device)
    pred = torch.zeros(B, T, dtype=torch.long, device=device)
    pos_out = torch.zeros(B, T, device=device)

    start_ev = torch.cuda.Event(enable_timing=True) if device.type == "cuda" \
        else None
    end_ev = torch.cuda.Event(enable_timing=True) if start_ev else None
    if start_ev:
        start_ev.record()
    n_fwd = 0
    for _ in range(steps):
        open_pos = span_pos & ~frozen
        if not open_pos.any():
            break
        n_fwd += 1
        h = model.trunk(x, probe=False, attn_mask=attn_mask)
        positions = model.predict_positions(h)
        bidx, pidx = open_pos.nonzero(as_tuple=True)
        # temperature 0 = greedy on raw probs (a 1e-6 floor saturates the
        # softmax to all-1.0 confidences and breaks the freeze ordering)
        probs = _chunked_probs(h[bidx, pidx], model.embed.weight,
                               chunk=chunk,
                               temperature=temperature if temperature > 0
                               else 1.0)
        if temperature and temperature > 0:
            picks = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            pconf = probs.gather(1, picks[:, None]).squeeze(1)
        else:
            pconf, picks = probs.max(dim=-1)
        for b in range(B):
            sel = bidx == b
            if not sel.any():
                continue
            k = min(per_step, int(sel.sum().item()))
            order = pconf[sel].argsort(descending=True)[:k]
            rows = pidx[sel][order]
            frozen[b, rows] = True
            pred[b, rows] = picks[sel][order]
            conf_out[b, rows] = pconf[sel][order]
            pos_out[b, rows] = positions[b, rows].float()
            x[b, rows] = picks[sel][order]  # context untouched by construction
    if end_ev:
        end_ev.record()
        torch.cuda.synchronize()
        latency_ms = start_ev.elapsed_time(end_ev)
    else:
        latency_ms = 0.0

    ids_rows, slot_rows = [], []
    for b in range(B):
        rows = span_pos[b].nonzero(as_tuple=True)[0]
        snapped = snap_slots(pred[b, rows].tolist(), pos_out[b, rows].tolist(),
                             conf_out[b, rows].tolist(), window)
        ids_rows.append(snapped["ids"])
        slot_rows.append(snapped["slots"])
    Lmax = max((len(r) for r in ids_rows), default=0)
    L_ids = torch.full((B, max(Lmax, 1)), 1, dtype=torch.long, device=device)
    L_slots = torch.zeros(B, max(Lmax, 1), dtype=torch.long, device=device)
    L_conf = torch.zeros(B, max(Lmax, 1), device=device)
    L_raw = torch.zeros(B, window, device=device)
    for b in range(B):
        n = len(ids_rows[b])
        if n:
            L_ids[b, :n] = torch.tensor(ids_rows[b], dtype=torch.long,
                                        device=device)
            L_slots[b, :n] = torch.tensor(slot_rows[b], dtype=torch.long,
                                          device=device)
        L_raw[b] = pos_out[b, T_ctx:]
    return dict(pred_ids=L_ids, slots=L_slots, conf=L_conf, n_fwd=n_fwd,
                lengths=torch.tensor([len(r) for r in ids_rows],
                                     dtype=torch.long),
                latency_ms=latency_ms, raw_positions=L_raw)
