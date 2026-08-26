"""Task 5: low-confidence-remasking span sampler.

Iterative MDLM sampling: start from context + [MASK]*L; each of `steps`
forward passes computes logits at the still-masked span positions only,
then freezes the ⌈L/steps⌉ highest-confidence predictions (argmax at
temperature 0, multinomial at temperature > 0 with the sampled prob as its
confidence). Positions not frozen stay [MASK] — i.e. low-confidence
predictions are implicitly re-masked and re-predicted next step. Context
positions are NEVER written. [EMPTY] predictions decode to the empty
string (empty_id is outside the tokenizer vocab; predicted [EMPTY] tokens
are stripped before decoding).

Batched with right padding + bool attn_mask; logits materialize only at
selected masked positions in small chunks (vocab 130k never expands to
(B,T,V) here).
"""
import torch
import torch.nn.functional as F


def _chunked_probs(h_sel, head_w, chunk=1024, temperature=1.0):
    """(N,V) softmax rows for selected positions, chunked. bf16 matmul,
    fp32 softmax. temperature <= 0 means GREEDY PICK, not zero-temperature
    softmax: dividing by ~0 saturates every row to an exact one-hot, all
    confidences read 1.0 and confidence-ranked freezing degenerates to
    index order — so the distribution is computed at temperature 1.0 and
    the caller argmaxes it (found by zcode-ddot-poc, board 00:35)."""
    temp = temperature if temperature > 0 else 1.0
    probs = []
    for c in range(0, h_sel.size(0), chunk):
        logits = F.linear(h_sel[c:c + chunk], head_w).float()
        probs.append(F.softmax(logits / temp, dim=-1))
    return torch.cat(probs) if probs else h_sel.new_zeros(0, head_w.size(0))


@torch.no_grad()
def sample_spans(model, prompt_ids, span_lens, steps, temperature=0.0,
                 generator=None, chunk=1024):
    """Sample span regions for a batch of triples.

    model: MDGQA in eval mode. prompt_ids: list[list[int]] (variable
    length, the shared context). span_lens: region length L per row
    (>=1; empty spans use the 1-token [EMPTY] region). steps: sampling
    steps (grid {8,16,32,64}). temperature: 0 = greedy confidence, >0 =
    multinomial sampling with sampled-prob confidence.

    Returns dict with:
      pred_ids  (B, Lmax) long — predicted region ids, padded with eos
      conf      (B, Lmax) float — frozen-prediction confidence, 0 on pads
      n_fwd    — forward passes actually used (early exit when all frozen)
      latency_ms — CUDA-event wall time of the sampling loop (0 on CPU)
    """
    device = next(model.parameters()).device
    B = len(prompt_ids)
    assert all(l >= 1 for l in span_lens)
    T_ctx = max(len(p) for p in prompt_ids)
    Lmax = max(span_lens)
    T = T_ctx + Lmax
    x = torch.full((B, T), 1, dtype=torch.long, device=device)  # eos pad
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, p in enumerate(prompt_ids):
        x[b, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        valid[b, :len(p)] = True
    span_pos = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, l in enumerate(span_lens):
        span_pos[b, T_ctx:T_ctx + l] = True
        x[b, T_ctx:T_ctx + l] = model.mask_id
        valid[b, T_ctx:T_ctx + l] = True
    attn_mask = valid[:, None, None, :]

    per_step = [max(1, -(-l // steps)) for l in span_lens]  # ceil
    frozen = torch.zeros(B, T, dtype=torch.bool, device=device)
    conf_out = torch.zeros(B, T, device=device)
    pred = torch.zeros(B, T, dtype=torch.long, device=device)

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
        bidx, pidx = open_pos.nonzero(as_tuple=True)  # (N,), (N,)
        probs = _chunked_probs(h[bidx, pidx], model.embed.weight,
                               chunk=chunk, temperature=temperature)
        if temperature and temperature > 0:
            picks = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            pconf = probs.gather(1, picks[:, None]).squeeze(1)
        else:
            pconf, picks = probs.max(dim=-1)
        # freeze the top-⌈L/steps⌉ confident OPEN positions per row
        for b in range(B):
            sel = bidx == b
            if not sel.any():
                continue
            k = min(per_step[b], int(sel.sum().item()))
            order = pconf[sel].argsort(descending=True)[:k]
            rows = pidx[sel][order]
            frozen[b, rows] = True
            pred[b, rows] = picks[sel][order]
            conf_out[b, rows] = pconf[sel][order]
            x[b, rows] = picks[sel][order]  # context untouched by construction
    if end_ev:
        end_ev.record()
        torch.cuda.synchronize()
        latency_ms = start_ev.elapsed_time(end_ev)
    else:
        latency_ms = 0.0

    L_ids = torch.full((B, Lmax), 1, dtype=torch.long, device=device)
    L_conf = torch.zeros(B, Lmax, device=device)
    for b, l in enumerate(span_lens):
        L_ids[b, :l] = pred[b, T_ctx:T_ctx + l]
        L_conf[b, :l] = conf_out[b, T_ctx:T_ctx + l]
    return dict(pred_ids=L_ids, conf=L_conf, n_fwd=n_fwd,
                latency_ms=latency_ms)


def decode_span(tok, region_ids, empty_id):
    """Region ids -> span text. Predicted [EMPTY] tokens are outside the
    tokenizer vocab: strip them; all-[EMPTY] decodes to ''."""
    ids = [i for i in region_ids.tolist() if i != empty_id]
    return tok.decode(ids) if ids else ""
