"""Task 3: MDLM span objective (timestep-free, SMDM-style).

Per SMDM: no timestep conditioning anywhere. Per example, sample a mask
rate t ~ U(0,1); mask span-region tokens i.i.d. Bernoulli(t); the loss is
-log p(original token | masked input) at the masked positions, weighted
1/t (eps-clamped, SMDM practice against 1/t blowup at small t). CONTEXT
(prompt) tokens are never masked and never carry loss weight.

Normalization (recorded decision): per example,
    contribution_i = (1/t_i) * sum_{j masked} CE_ij / S_i
whose expectation over the mask is the mean per-token CE of the span
(E[#masked] = t*S). The batch loss is the mean of contributions (zero-
contribution examples — no span token happened to be masked — stay in the
mean; they are part of the unbiased estimator). This keeps the loss on the
same per-token-nats scale as the AR twin's chunked CE, so the poc_twin
Muon lr 0.01 / embed 4e-3 recipe transfers.

Loss weight only ever lands on masked span positions; gradients still
flow INTO context representations through attention (that is the point of
bidirectional context) — see test (c) for the precise property.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

T_MIN = 0.01  # clamp on t for the 1/t weight (plan risk: MDLM instability)


def sample_rates(n, generator=None, device="cpu"):
    """t ~ U(0,1), per example."""
    return torch.rand(n, generator=generator, device=device)


def bernoulli_mask(span_pos, t, generator=None):
    """(B,T) bool: True where a SPAN position is masked this draw.
    Context positions are never in span_pos, hence never masked."""
    draw = torch.rand(span_pos.shape, generator=generator, device=span_pos.device)
    return span_pos & (draw < t[:, None])


def _ce_chunk_weighted(hc, w, tc, wc):
    logits = F.linear(hc, w)
    return F.cross_entropy(logits, tc, reduction="none") * wc


def mdlm_loss(h, x, m, t, span_len, head_w, chunk=4096):
    """Vectorized weighted MDLM loss (see module docstring for the exact
    estimator). All tensors on the same device; h/x/m (B,T,·), t/span_len (B,).

    The vocab-sized matmul runs in checkpointed chunks on the SELECTED
    (masked span) positions only — memory O(chunk), same discipline as
    poc_twin.model.chunked_ce (kept out of torch.compile for the same
    reason: inductor materializes fp32 vocab intermediates).
    """
    idx = m.nonzero(as_tuple=False)  # (N, 2): [b, pos]
    if idx.numel() == 0:
        return h.sum() * 0.0  # keep the graph (all-zero batch edge case)
    h_sel = h[idx[:, 0], idx[:, 1]]
    tgt = x[idx[:, 0], idx[:, 1]]
    w_pos = (1.0 / t.clamp(min=T_MIN))[idx[:, 0]]
    n = h_sel.size(0)
    total = h_sel.new_zeros((), dtype=torch.float32)
    for c in range(0, n, chunk):
        total = total + checkpoint(
            _ce_chunk_weighted, h_sel[c:c + chunk], head_w,
            tgt[c:c + chunk], w_pos[c:c + chunk], use_reentrant=False)
    per_pos = total  # sum_i w_i * CE_i
    # per-example (1/t)-weighted sum, normalized by span length S_i
    bidx = idx[:, 0]
    ex_sum = torch.zeros(h.size(0), device=h.device, dtype=torch.float32)
    ex_sum.scatter_add_(0, bidx, per_pos)
    contrib = ex_sum / span_len.clamp(min=1).float()
    return contrib.mean()


def naive_mdlm_loss(h, x, m, t, span_len, head_w):
    """Loop implementation of the identical estimator — the brute-force
    reference for test (a). Do not use in training."""
    B = h.size(0)
    logits = F.linear(h, head_w)
    logp = F.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for b in range(B):
        s = 0.0
        tw = 1.0 / max(t[b].item(), T_MIN)
        for j in range(h.size(1)):
            if m[b, j]:
                s = s - logp[b, j, x[b, j]].item() * tw
        total = total + s / max(1, span_len[b].item())
    return total / B


def span_batch_to_loss(model, x, span_pos, valid, t=None, generator=None,
                       chunk=4096):
    """Compose mask sampling + input masking + trunk forward + MDLM loss.
    Returns (loss, telemetry dict). x: (B,T) ids with TRUE span tokens;
    span_pos/valid: (B,T) bool; model: MDGQA."""
    B = x.size(0)
    if t is None:
        t = sample_rates(B, generator=generator, device=x.device)
    m = bernoulli_mask(span_pos, t, generator=generator)
    x_in = x.masked_fill(m, model.mask_id)
    attn_mask = valid[:, None, None, :] if valid is not None else None
    h = model.trunk(x_in, probe=False, attn_mask=attn_mask)
    loss = mdlm_loss(h, x, m, t, span_pos.sum(dim=1), model.embed.weight,
                     chunk=chunk)
    stats = dict(t_mean=float(t.mean()),
                 mask_rate=float(m.sum() / span_pos.sum().clamp(min=1)),
                 n_masked=int(m.sum()))
    return loss, stats
