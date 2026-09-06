"""Task 2: bidirectional TinyGQA-MD model.

The diffusion twin = the poc_twin TinyGQA shape with the causal mask
removed, one extra [MASK] embedding row and one [EMPTY] class row, tied
output head resized to match. Everything else (GQA 16Q/2KV, SwiGLU,
pre-RMSNorm, RoPE theta 500k, init discipline, QK-Clip telemetry path) is
the AR twin's — imported and subclassed, never copy-pasted.

Bidirectionality note: RoPE encodes relative position, so permuting input
tokens changes rotary phases and outputs do NOT simply permute — the
no-causal-leakage property is therefore tested by suffix perturbation
(perturbing tokens AFTER position i must change the output AT position i,
which a causal mask forbids).

Padding: batches of variable-length triples are padded; `attn_mask` (bool,
True = attend, broadcastable to (B,1,Tq,Tk)) masks pad KEYS out of every
attention. Pad-row outputs are garbage by construction and never read.
The QK probe runs over pad positions too (telemetry-only inflation,
tau=100 QK-Clip is a soft mechanism; accepted for the POC).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.training.poc_twin import model as twin
from experiments.training.poc_diff import BASE_VOCAB

# With a bool attn_mask the flash backend is unavailable; MATH materializes
# (B,H,T,T) scores and OOMs a 13.7GB budget. Memory-efficient handles bool
# masks without materialization; MATH stays as the CPU/fallback path.
_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class BiAttention(twin.Attention):
    """TinyGQA attention with is_causal dropped and an optional padding
    mask. GQA/enable_gqa/QK probe inherited unchanged."""

    def forward(self, x, cos, sin, probe=True, attn_mask=None):
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.n_q, self.head_dim)
        k = self.Wk(x).view(B, T, self.n_kv, self.head_dim)
        v = self.Wv(x).view(B, T, self.n_kv, self.head_dim)
        q = twin.apply_rope(q, cos, sin)
        k = twin.apply_rope(k, cos, sin)
        if probe:
            self.qk_smax = self._max_qk_logit(q, k)
        with sdpa_kernel(_SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=False, attn_mask=attn_mask, enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, T, self.n_q * self.head_dim)
        return self.Wo(y)


class BiBlock(twin.Block):
    """twin.Block with the attention submodule swapped for BiAttention."""

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg["d_model"]
        self.attn = BiAttention(d, cfg["n_q"], cfg["n_kv"], cfg["head_dim"])

    def forward(self, x, cos, sin, probe=True, attn_mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, probe=probe, attn_mask=attn_mask)
        xb = self.ln2(x)
        h = F.silu(self.Wg(xb)) * self.Wu(xb)
        return x + self.Wd(h)


class MDGQA(twin.TinyGQA):
    """Bidirectional masked-diffusion twin of TinyGQA.

    cfg additions vs twin.model_config: vocab is the EXTENDED size
    (base+2); mask_id/empty_id pin the two new rows. Init for the new rows:
    [MASK] = mean of the base embeddings, [EMPTY] = mean + N(0, 0.02)
    (plan Task 2 step 2)."""

    def __init__(self, cfg):
        super().__init__(cfg)  # builds causal blocks + base embed; we rebuild both
        base = cfg["mask_id"]
        # swap the block stack for bidirectional blocks, then run the SAME
        # init discipline the parent applied to its (causal) blocks
        self.blocks = nn.ModuleList(BiBlock(cfg) for _ in range(cfg["n_layers"]))
        self.blocks.apply(self._init)
        for b in self.blocks:
            for w in (b.attn.Wo.weight, b.Wd.weight):
                nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layers"]))
        # extend the (tied) embedding table by the two new rows (the parent
        # constructor already sized it to base+2; keep its base rows)
        old = self.embed.weight.data[:base]
        new_embed = nn.Embedding(cfg["vocab"], cfg["d_model"])
        with torch.no_grad():
            new_embed.weight[:base] = old
            mean_row = old.mean(dim=0)
            new_embed.weight[base] = mean_row                    # [MASK]
            new_embed.weight[base + 1] = mean_row + 0.02 * torch.randn_like(mean_row)
        self.embed = new_embed
        self.mask_id = base
        self.empty_id = base + 1
        # --- X5-S1 carry channel (2026-09-06; additive, default-off) ---
        # Recurrent self-conditioning per FRM (arXiv 2606.29150 v2, App-A
        # + Analog-Bits lineage): the trunk ingests the previous pass's
        # RAW span logits as a carry. Their carry is "a previous
        # clean-solution prediction" (the full categorical distribution);
        # the ingestion projection is unspecified in the paper, and a
        # raw-logit (V,d) Linear at V=130,562 would add 100.3M params
        # (+49% model), so the port ingests the raw logits through the
        # TIED embedding table — c = softmax(logits) @ E, a soft
        # clean-prediction embedding (information-preserving up to the
        # softmax row-shift) — then adds W_c(c) at the input embedding,
        # with W_c ZERO-INIT. Zero-init + the None guard below keep every
        # pre-X5 code path (banked ckpts, sample.py, eval_spans.py)
        # byte-identical: loading a banked ckpt with strict=False leaves
        # W_c at 0 and trunk(carry=None) never touches it.
        self.carry_proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)
        nn.init.zeros_(self.carry_proj.weight)

    def trunk(self, idx, probe=True, attn_mask=None, carry=None):
        """Embed -> bidirectional blocks -> final norm.

        carry: optional (B,T,d_model) soft-embedding carry — the previous
        pass's raw span logits ingested via softmax @ embed.weight (see
        logits_to_carry); rows are ZERO at positions without a previous
        prediction (their null carry). Added through the zero-init
        carry_proj; carry=None reproduces the pre-X5 trunk exactly."""
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        x = self.embed(idx)
        if carry is not None:
            x = x + self.carry_proj(carry.to(x.dtype))
        for b in self.blocks:
            x = b(x, cos, sin, probe=probe, attn_mask=attn_mask)
        return self.ln_f(x)

    def forward(self, idx, attn_mask=None, probe=True, carry=None):
        """Full-vocab logits via the tied head (training computes the loss
        in objective.py against trunk() hidden states; this path is for
        sampling/eval where full logits are fine)."""
        h = self.trunk(idx, probe=probe, attn_mask=attn_mask, carry=carry)
        return F.linear(h, self.embed.weight)


def logits_to_carry(h, span_pos, head_w, chunk=1024):
    """X5-S1: previous-pass RAW span logits -> dense soft-embedding carry.

    For every span position (span_pos (B,T) bool), computes
    c_i = softmax(logits_i) @ head_w over the full extended vocab (fp32,
    `_chunked_probs` numerics: temperature-1.0 softmax, chunked so the
    (N,V) logits never all materialize), and scatters the rows into a
    zero (B,T,d) fp32 tensor — non-span rows stay exactly zero, the null
    carry. Returns the (B,T,d) carry for MDGQA.trunk(carry=...).

    Dtype-robust: h may arrive bf16 (from a trunk pass under autocast)
    while head_w is fp32 — the selected rows are cast to head_w's dtype
    so the linear runs fp32 in any context (matching eval numerics)."""
    B, T, d = h.shape[0], h.shape[1], head_w.shape[1]
    bidx, pidx = span_pos.nonzero(as_tuple=True)
    out = h.new_zeros(B, T, d, dtype=torch.float32)
    if bidx.numel() == 0:
        return out
    h_sel = h[bidx, pidx].to(head_w.dtype)
    rows = []
    for c in range(0, h_sel.size(0), chunk):
        logits = F.linear(h_sel[c:c + chunk], head_w).float()
        probs = F.softmax(logits, dim=-1)
        rows.append(probs @ head_w.float())   # (chunk, d) fp32
    out[bidx, pidx] = torch.cat(rows)
    return out


def model_config_md(**over):
    """twin.model_config + the two new vocabulary rows.

    The extended vocab convention: pass the BASE vocab (default 130,560);
    mask_id/empty_id/vocab are derived. Tests pass small vocabs."""
    cfg = dict(vocab=BASE_VOCAB, d_model=768, n_layers=12, n_q=16, n_kv=2,
               head_dim=64, ffn_hidden=3072, rope_theta=500000.0, max_seq=1024)
    cfg.update(over)
    base = cfg["vocab"]
    cfg.update(vocab=base + 2, mask_id=base, empty_id=base + 1, base_vocab=base)
    return cfg


# --- M1 micro config (2026-09-01 micro-specialist probe plan, additive) ---
# ~75M-class sibling of the 206M anchor for the fixed 0.5B-token budget:
# HALF the width, SAME depth/GQA-KV/head-dim/ffn-ratio, vocab PINNED at
# BASE_VOCAB+2 = 130,562 so tokenizer + harness are unchanged. Frozen at
# prep (M1_PREP.md); the anchor defaults above are untouched.
MICRO = dict(d_model=384, n_layers=12, n_q=6, n_kv=2, head_dim=64,
             ffn_hidden=1536)
# printed at prep: total=76,097,664 (76.10M) = embed 50.14M + hidden 25.96M
# = 0.369x the anchor's 206,459,136; inside the plan's 70-80M band.


def model_config_micro(**over):
    """model_config_md with the MICRO shape merged in (vocab stays BASE)."""
    cfg = dict(MICRO)
    cfg.update(over)
    return model_config_md(**cfg)


def count_params_md(cfg=None):
    m = MDGQA(cfg or model_config_md())
    n_all = sum(p.numel() for p in m.parameters())
    n_emb = m.embed.weight.numel()
    return n_all, n_emb, n_all - n_emb


if __name__ == "__main__":
    cfg = model_config_md()
    n_all, n_emb, n_hid = count_params_md(cfg)
    base = twin.count_params(twin.model_config())
    print(f"MDGQA: total={n_all/1e6:.1f}M embed={n_emb/1e6:.1f}M "
          f"hidden={n_hid/1e6:.1f}M")
    print(f"TinyGQA base: total={base[0]/1e6:.1f}M  delta={n_all - base[0]} "
          f"(expected 2*768=1536)")
    # X5-S1: delta = 2 new vocab rows + the d^2 carry_proj (2026-09-06)
    assert (n_all - base[0]) == 2 * cfg["d_model"] + cfg["d_model"] ** 2, \
        "param audit (2 rows + carry channel)"
    within = abs(n_all - (206.5e6 + 2 * 768)) / (206.5e6 + 2 * 768)
    print(f"|total - (206.5M + 1536)| / total = {within:.5f} (plan band: <0.01)")
    assert within < 0.01, "plan param audit ±1% of 206.5M + 2*768"
    print("param audit OK")

    # M1 micro audit (the prep param printout)
    cfg_m = model_config_micro()
    n_m, n_emb_m, n_hid_m = count_params_md(cfg_m)
    print(f"MDGQA-micro: d={cfg_m['d_model']} L={cfg_m['n_layers']} "
          f"n_q={cfg_m['n_q']} n_kv={cfg_m['n_kv']} ffn={cfg_m['ffn_hidden']} "
          f"vocab={cfg_m['vocab']}")
    print(f"MDGQA-micro: total={n_m:,} ({n_m/1e6:.2f}M) "
          f"embed={n_emb_m/1e6:.2f}M hidden={n_hid_m/1e6:.2f}M "
          f"({n_m/n_all:.3f}x anchor)")
    assert cfg_m["vocab"] == 130_562, "micro vocab pinned at BASE_VOCAB+2"
    assert 70e6 <= n_m <= 80e6, "micro param band 70-80M"
    print("micro param audit OK")
