"""X5-S1: recurrent self-conditioning sampler + banked-ckpt load gate.

FRM (arXiv 2606.29150 v2) inference scheme, ported to the remasking
sampler: "a sampler can alternate between advancing the flow state x_t
and applying additional recurrent updates at the current flow time."
Here: each remasking schedule step (a freeze of ceil(L/steps) top-
confidence OPEN positions, sample.py verbatim) first runs `depth` k
recurrent forwards on the CURRENT input, the carry feeding back the
previous forward's raw span logits (as the soft clean-prediction
embedding, `logits_to_carry`); the first forward of a run carries None.
Freezing decisions come from the LAST recurrent forward (deepest
refinement). NFE accounting: n_fwd = forwards actually run (nominal
steps*depth; the loop early-exits when all positions are frozen, exactly
as sample.py).

Load gate (B8 pattern, pre-registered health check): a banked pre-X5
ckpt lacks carry_proj; load_md_x5 loads it strict=False, asserts the
missing-key set is EXACTLY {"carry_proj.weight"}, asserts the loaded
W_c stays all-zero, and the trunk's carry=None path never touches it —
so the loaded model is byte-identical to the banked one until a carry
is passed (unit-tested in test_x5_s1.py).
"""
import torch

from experiments.training.poc_diff.model_md import MDGQA, logits_to_carry
from experiments.training.poc_diff.sample import _chunked_probs


def load_md_x5(path, device):
    """MDGQA from a ckpt; pre-X5 ckpts (no carry_proj) load with the
    zero-init carry channel intact. Pre-registered gates: the only
    missing key may be carry_proj.weight, and then it must be zero."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDGQA(ck["cfg"]).to(device).eval()
    sd = {k: v.float() for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in ckpt: {unexpected}"
    assert set(missing) <= {"carry_proj.weight"}, \
        f"missing keys beyond the carry channel: {missing}"
    if "carry_proj.weight" in missing:
        assert torch.count_nonzero(model.carry_proj.weight) == 0, \
            "carry_proj must be zero-init at load (pre-X5 ckpt)"
    return model


@torch.no_grad()
def sample_spans_rc(model, prompt_ids, span_lens, steps, depth=1,
                    temperature=0.0, generator=None, chunk=1024):
    """sample.sample_spans + recurrent self-conditioning carry.

    Same batch/pad/freeze conventions as sample.sample_spans (which see);
    per schedule step the model runs `depth` recurrent forwards with the
    carry feeding back its previous raw span logits. depth=1 with the
    carry active is the minimal recurrent sampler (NFE = steps, the
    anchor accounting); depth=k>1 multiplies NFE by k.

    Returns the sample_spans dict plus n_fwd_nominal = steps*depth.
    (Residual trajectories are measured by the eval-side instrumented
    replay in x5_s1_eval.py, mirroring x5_s0_residual.replay_row.)
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
        carry = None
        probs = None
        bidx = pidx = None
        for _j in range(depth):
            h = model.trunk(x, probe=False, attn_mask=attn_mask, carry=carry)
            n_fwd += 1
            carry = logits_to_carry(h, span_pos, model.embed.weight,
                                    chunk=chunk)
            bidx, pidx = open_pos.nonzero(as_tuple=True)
            probs = _chunked_probs(h[bidx, pidx], model.embed.weight,
                                   chunk=chunk, temperature=temperature)
        # freeze from the LAST recurrent forward's distributions
        if temperature and temperature > 0:
            picks = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            pconf = probs.gather(1, picks[:, None]).squeeze(1)
        else:
            pconf, picks = probs.max(dim=-1)
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
    out = dict(pred_ids=L_ids, conf=L_conf, n_fwd=n_fwd,
               n_fwd_nominal=steps * depth, latency_ms=latency_ms)
    return out
