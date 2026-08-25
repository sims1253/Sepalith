"""A2-structure model: TinyGQA + matryoshka exits + Markov MTP.

The A2-prime spec's training structure, built on the verified twin
components (Block/Muon/QK-Clip untouched):

  MATRYOSHKA EXITS: per-exit final RMSNorms at the configured exit
  layers (below the top); the loss is
      L = CE_top + sum_e w_e * CE_e        (spec: 0.25@16L, 0.125@8L)
  — one forward pass trains all tiers (the ~1.04x economy). The
  stochastic-depth variant (LayerSkip dropout of upper layers) is a
  trainer flag; the deterministic multi-exit loss is the spec's formula.

  MTP (1 layer, Markov-conditioned): input at position t is
      RMSNorm(h_top(t)) (pre-final-norm trunk output) ⊕ E[t+1]
  predicting token t+2 — the DeepSeek-style draft head; λ ramps in the
  trainer (spec: 0.5 from 10% of budget).

  THE INSTRUMENT: per-exit held-out BPB — "16L ≈ 24L within 2%" is the
  ship-M flagship gate (design §1.2).

Cluster notes: single-file model, no custom kernels; DDP-ready (the
trainer documents the launch); micro-bs auto-shaped from free VRAM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Block, TinyGQA, chunked_ce, rope_cache


class A2Model(TinyGQA):
    """TinyGQA + exit taps + MTP. cfg keys: exit_layers (list of ints <
    n_layers), exit_weights (parallel list), use_mtp (bool)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.exit_layers = list(cfg.get("exit_layers") or [])
        self.exit_weights = list(cfg.get("exit_weights")
                                  or [0.25] * len(self.exit_layers))
        assert len(self.exit_weights) == len(self.exit_layers)
        for e in self.exit_layers:
            assert 0 < e < cfg["n_layers"], "exits sit below the top"
        self.exit_norms = nn.ModuleList(
            nn.RMSNorm(cfg["d_model"], eps=1e-6)
            for _ in self.exit_layers)
        self.use_mtp = bool(cfg.get("use_mtp"))
        if self.use_mtp:
            self.mtp_proj = nn.Linear(2 * cfg["d_model"], cfg["d_model"],
                                       bias=False)
            self.mtp_block = Block(cfg)
            self.mtp_norm = nn.RMSNorm(cfg["d_model"], eps=1e-6)
            # same residual-projection scaling init as the trunk blocks
            for w in (self.mtp_block.attn.Wo.weight,
                      self.mtp_block.Wd.weight):
                nn.init.normal_(
                    w, mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg["n_layers"]))

    def trunk_taps(self, idx, probe=True):
        """Returns (h_top_preln, {exit_layer: exit_normed_hidden})."""
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        x = self.embed(idx)
        taps = {}
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, probe=probe)
            layer_no = i + 1
            if layer_no in self.exit_layers:
                taps[layer_no] = self.exit_norms[
                    self.exit_layers.index(layer_no)](x)
        return x, taps

    def forward(self, idx, targets=None, probe=True, mtp_weight=0.0):
        """Returns (logits_or_None, loss_dict). Takes the FULL overlap-1
        block (1025) for idx/targets; inputs are shifted internally to
        max_seq (1024) — the rope cache bounds sequence length."""
        inp = idx[:, :-1]
        x_pre, taps = self.trunk_taps(inp, probe=probe)
        if targets is None:
            h = self.ln_f(x_pre)
            return F.linear(h, self.embed.weight), None
        loss_dict = {}
        # top exit (the normal LM loss): predict t+1 from h(t)
        h_top = self.ln_f(x_pre)
        # h(t) -> token t+1 for ALL t in 0..T-1: FULL hidden vs the
        # shifted targets (the twin convention — no :-1 slice)
        loss_dict["ce_top"] = chunked_ce(h_top.contiguous(),
                                         self.embed.weight, targets[:, 1:])
        for e, w, tap in zip(self.exit_layers, self.exit_weights, taps.values()):
            # taps[e] is the normed hidden AT position t (post-block-e);
            # its prediction target is also t+1
            loss_dict[f"ce_{e}"] = chunked_ce(
                tap.contiguous(), self.embed.weight, targets[:, 1:])
        if self.use_mtp and mtp_weight > 0.0:
            # MTP: at position t (< T-1): input mtp_norm(h_pre(t)) ⊕
            # E[idx[t+1]], predict idx[t+2]
            # explicit alignment: t = 0..Tm-1 with Tm = T-1
            #   input  h(t) (t<=Tm-1) ⊕ E[inp[t+1]]  (inp has T tokens)
            #   target idx[t+2]                       (idx has T+1 tokens)
            h_norm = self.mtp_norm(x_pre)
            Tm = inp.size(1) - 1
            e_next = self.embed(inp[:, 1:Tm + 1])
            mtp_in = self.mtp_proj(
                torch.cat([h_norm[:, :Tm], e_next], dim=-1))
            cos = self.rope_cos[:Tm].to(idx.device)
            sin = self.rope_sin[:Tm].to(idx.device)
            mtp_h = self.mtp_block(mtp_in, cos, sin, probe=False)
            loss_dict["ce_mtp"] = chunked_ce(
                mtp_h.contiguous(), self.embed.weight,
                targets[:, 2:Tm + 2])
        return None, loss_dict

    @torch.no_grad()
    def qk_clip_all(self, tau=100.0, alpha=0.5):
        n_tot, gmax = super().qk_clip_all(tau, alpha)
        if self.use_mtp:
            n, m = self.mtp_block.attn.qk_clip(tau, alpha)
            n_tot += n
            gmax = max(gmax, m)
        return n_tot, gmax


def a2_config(**over):
    from model import model_config
    cfg = model_config(exit_layers=[8, 16] if (over.get("n_layers") or 12) > 16
                       else [4, 8],
                       exit_weights=[0.25, 0.125], use_mtp=True, **over)
    return cfg
