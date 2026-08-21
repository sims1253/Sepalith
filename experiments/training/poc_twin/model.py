"""
TinyGQA: the POC from-scratch transformer for the Muon-vs-AdamW twin.

Shape follows docs/research/fromscratch-design-A2.md scaled down to the POC
band per the twin brief: dense GQA 16Q/2KV x head_dim 64, d_model 768,
12 layers, SwiGLU hidden 4*d (3072x768 tall-MLP geometry, the 8192x2048 of
the 1.5B scaled proportionally), pre-RMSNorm eps 1e-6, RoPE theta 500k,
tied embeddings, MiniCPM5 tokenizer (vocab 130,560 padded).

QK stabilization: NO QK-norm (sweep OPT-4a shape) + stateless QK-Clip
(Kimi K2 MuonClip): the attention forward records the per-head max QK
logit S_max^h (chunked, no_grad); the trainer, AFTER the optimizer step,
rescales W_Q/W_K row-blocks by (tau/S_max^h)^0.5 for heads with
S_max^h > tau (alpha=0.5 split across Q and K so the product scales by
gamma=tau/S_max^h). With GQA the shared K head takes the max gamma of its
Q-head group (guarantees every head's product lands under tau).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_chunk(hc, w, tc):
    """CE (sum) over a chunk of tokens; run under checkpoint so the
    vocab-130k logits are recomputed instead of stored for backward.
    Kept EAGER (not compiled): inductor materializes fp32 vocab-sized
    intermediates; the eager fused CE kernel does not."""
    logits = F.linear(hc, w)
    return F.cross_entropy(logits, tc, reduction="sum")


def chunked_ce(h, w, targets, chunk=4096):
    """Training loss: mean CE over all tokens, computed in checkpointed
    chunks (memory O(chunk), compute +1 lm_head forward)."""
    h2 = h.view(-1, h.size(-1))
    t2 = targets.reshape(-1)
    n = h2.size(0)
    total = h2.new_zeros((), dtype=torch.float32)
    for c in range(0, n, chunk):
        total = total + checkpoint(_ce_chunk, h2[c:c + chunk], w, t2[c:c + chunk],
                                   use_reentrant=False)
    return total / n


def rope_cache(max_seq: int, head_dim: int, theta: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, T, H, hd); cos/sin: (T, hd/2)
    x1, x2 = x.float().chunk(2, dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d_model, n_q, n_kv, head_dim):
        super().__init__()
        self.n_q, self.n_kv, self.head_dim = n_q, n_kv, head_dim
        self.Wq = nn.Linear(d_model, n_q * head_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.Wo = nn.Linear(n_q * head_dim, d_model, bias=False)
        self.qk_smax = None  # (n_q,) fp32 per-head max QK logit from last forward

    def forward(self, x, cos, sin, probe=True):
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.n_q, self.head_dim)
        k = self.Wk(x).view(B, T, self.n_kv, self.head_dim)
        v = self.Wv(x).view(B, T, self.n_kv, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if probe:
            self.qk_smax = self._max_qk_logit(q, k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, T, self.n_q * self.head_dim)
        return self.Wo(y)

    @torch.no_grad()
    def _max_qk_logit(self, q, k):
        """Per-q-head max pre-softmax logit q.k/sqrt(hd), chunked over the
        query axis to bound memory. fp32 for telemetry fidelity."""
        B, T, H, hd = q.shape
        qf = q.float()
        kf = k.float().repeat_interleave(H // k.shape[2], dim=2)  # (B,T,H,hd)
        smax = torch.full((H,), -1e30, device=q.device)
        chunk = 256
        for i in range(0, T, chunk):
            logits = torch.einsum('bqhd,bkhd->bhqk', qf[:, i:i + chunk], kf)
            logits = logits / math.sqrt(hd)
            smax = torch.maximum(smax, logits.amax(dim=(0, 2, 3)))
            del logits
        return smax

    @torch.no_grad()
    def qk_clip(self, tau=100.0, alpha=0.5):
        """Stateless post-step clip; returns (n_clipped_heads, max_smax)."""
        smax = self.qk_smax
        if smax is None:
            return 0, float('nan')
        over = smax > tau
        n = int(over.sum().item())
        if n > 0:
            gamma = (tau / smax.clamp_min(1e-6)) ** alpha  # (n_q,)
            gq = torch.where(over, gamma, torch.ones_like(gamma))
            # shared KV head takes the most aggressive gamma of its Q-group
            grp = self.n_q // self.n_kv
            gamma_kv = gq.view(self.n_kv, grp).min(dim=1).values  # (n_kv,)
            scale_q = torch.ones(self.n_q, 1, device=self.Wq.weight.device,
                                 dtype=self.Wq.weight.dtype)
            scale_q[over] = gq[over].to(scale_q.dtype).unsqueeze(1)
            self.Wq.weight.mul_(scale_q.repeat_interleave(self.head_dim, dim=0))
            Wk = self.Wk.weight  # (n_kv*hd, d)
            scale_k = torch.where(gamma_kv < 1.0, gamma_kv, torch.ones_like(gamma_kv))
            Wk.mul_(scale_k.to(Wk.dtype).unsqueeze(1).repeat_interleave(self.head_dim, dim=0))
        return n, float(smax.max().item())


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.RMSNorm(d, eps=1e-6)
        self.attn = Attention(d, cfg["n_q"], cfg["n_kv"], cfg["head_dim"])
        self.ln2 = nn.RMSNorm(d, eps=1e-6)
        h = cfg["ffn_hidden"]
        self.Wg = nn.Linear(d, h, bias=False)
        self.Wu = nn.Linear(d, h, bias=False)
        self.Wd = nn.Linear(h, d, bias=False)

    def forward(self, x, cos, sin, probe=True):
        x = x + self.attn(self.ln1(x), cos, sin, probe=probe)
        xb = self.ln2(x)
        h = F.silu(self.Wg(xb)) * self.Wu(xb)
        return x + self.Wd(h)


class TinyGQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab"], cfg["d_model"])
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
        self.ln_f = nn.RMSNorm(cfg["d_model"], eps=1e-6)
        cos, sin = rope_cache(cfg["max_seq"], cfg["head_dim"],
                              cfg["rope_theta"], "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # residual-projection scaling (GPT-2 trick)
        for b in self.blocks:
            for w in (b.attn.Wo.weight, b.Wd.weight):
                nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layers"]))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def trunk(self, idx, probe=True):
        """Embed -> blocks -> final norm. Returns hidden states h."""
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, cos, sin, probe=probe)
        return self.ln_f(x)

    def forward(self, idx, targets=None, probe=True):
        h = self.trunk(idx, probe=probe)
        if targets is None:
            # generation path: full logits are fine (no backward)
            return F.linear(h, self.embed.weight), None  # tied head
        return None, chunked_ce(h, self.embed.weight, targets)

    @torch.no_grad()
    def qk_clip_all(self, tau=100.0, alpha=0.5):
        """Apply QK-Clip on every layer from the last forward's telemetry.
        Returns (n_heads_clipped, global_max_smax)."""
        n_tot, gmax = 0, -float("inf")
        for b in self.blocks:
            n, m = b.attn.qk_clip(tau, alpha)
            n_tot += n
            gmax = max(gmax, m)
        return n_tot, gmax

    @torch.no_grad()
    def generate(self, idx, n_new, greedy=True):
        """Naive cacheless greedy decode (POC sanity only)."""
        out = idx.clone()
        for _ in range(n_new):
            idx_in = out[:, -self.cfg["max_seq"]:]
            logits, _ = self(idx_in, probe=False)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out


def model_config(**over):
    cfg = dict(vocab=130560, d_model=768, n_layers=12, n_q=16, n_kv=2,
               head_dim=64, ffn_hidden=3072, rope_theta=500000.0, max_seq=1024)
    cfg.update(over)
    return cfg


def count_params(cfg):
    m = TinyGQA(cfg)
    n_all = sum(p.numel() for p in m.parameters())
    n_emb = m.embed.weight.numel()
    return n_all, n_emb, n_all - n_emb
