#!/usr/bin/env python3
"""MLV: EBT verifier v1 (NCE contrastive energy head over (prompt, completion)).

Readouts: chosen-vs-rejected AUROC (KILL if < 0.65), pairwise accuracy vs
V1D-style labels, ECE, coarse-to-fine routing win. NO inner loop in v1.
AG1 target restated for a learned gate: >=20% FP suppression at >=99%
accepted retention on the V1a test split.

DESIGN DECISIONS (queue row: NCE head + head-MRL nesting {128/256/512}):

1. TRUNK: the 206M poc twin (experiments/training/poc_twin, TinyGQA d=768 /
12L / GQA, tied embeddings, MiniCPM5 130k tokenizer, muon_final.pt) is an
AR-LM-shaped decoder trunk: causal attention, no bidirectional pair
encoding, vocab-tied 130k head, and a checkpoint + tokenizer dependency
that is too heavy for a CPU v1 script. A frozen b4-base encoder pass is
likewise too heavy for v1 (GGUF serve + full-width forward per pair).
FALL BACK (honest): a SMALL trainable-from-scratch text encoder -- shared
two-tower, byte-level ids (0-255 + pad 256), 2-layer TransformerEncoder
d=256, mean-pool, linear up to a 512-d pooled rep. The MRL nesting
{128, 256, 512} is applied to that pooled rep (first-k dims). If v1
clears the AUROC gate, a b4-distilled trunk is the justified v2 lever.

2. LABELS: positive = accepted (V1a point decision) or scenario-exact
(requests/logprob rows); negative = false_suggestion / dismissed /
noop-FP (proposal=True, expectation=no_proposal) / scenario non-exact.
noop/correct_stop rows are abstentions, not completion judgments: skipped.

3. SPLIT: group-disjoint train/test. Group key = episode key for V1a
points (all points of one trajectory stay on one side), scenario family
(or row id) for scenario rows. Deterministic hash split (default 80/20).

4. LOSS: per-nesting-level binary NCE (BCE on pos/neg pair energies) plus
in-batch InfoNCE over the positives of the batch (negatives = other
positives' completions); level losses combined with uniform MRL weights.
Energy e_k(p, c) = dot(z_p[:k], z_c[:k]) / tau_k, tau learnable per level.

DATA (all banked): V1a episodes.jsonl carries the (prompt, proposal)
texts; confirm.jsonl carries the matching per-point labels/decisions
(joined by row index); requests.jsonl + logprob-features.jsonl carry
scenario labels but NO prompt text, so they contribute labels only where
joinable and are otherwise skipped for pair training (counted + reported).

Run: cd <worktree root> && .venv/bin/python experiments/eval/mlv_verifier.py
  train --episodes <ep.jsonl> --confirm <c.jsonl> --out <ckpt.pt> [--smoke]
  eval  --ckpt <ckpt.pt> --episodes <ep.jsonl> --confirm <c.jsonl>
  train --smoke  (tiny CPU mock smoke: synthetic pairs, d=32, 2 epochs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NEST_LEVELS = (128, 256, 512)
REP_DIM = 512
PAD_ID = 256
N_TOKENS = 257
AUROC_KILL = 0.65
MAX_LEN = 256


# ---------------------------------------------------------------- tokenize
def tokenize(text: str, max_len: int = MAX_LEN) -> list[int]:
    ids = list(text.encode("utf-8", errors="ignore")[:max_len])
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def batch_encode(texts: list[str], max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([tokenize(t, max_len) for t in texts], dtype=torch.long)
    mask = (ids != PAD_ID).float()
    return ids, mask


# ---------------------------------------------------------------- encoder
class PairEncoder(nn.Module):
    """Shared two-tower byte-level encoder -> 512-d pooled rep."""

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(N_TOKENS, d_model, padding_idx=PAD_ID)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, REP_DIM)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids)
        h = self.tr(h, src_key_padding_mask=(mask == 0))
        h = self.norm(h)
        denom = mask.sum(1).clamp_min(1).unsqueeze(1)
        pooled = (h * mask.unsqueeze(2)).sum(1) / denom
        return self.proj(pooled)


class MLVVerifier(nn.Module):
    """Two-tower energy model with per-level learnable temperature."""

    def __init__(self, d_model: int = 256, levels: tuple = NEST_LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        self.enc = PairEncoder(d_model=d_model)
        self.log_tau = nn.Parameter(torch.zeros(len(self.levels)))

    def reps(self, p_ids, p_mask, c_ids, c_mask):
        return self.enc(p_ids, p_mask), self.enc(c_ids, c_mask)

    def energies(self, zp: torch.Tensor, zc: torch.Tensor) -> dict[int, torch.Tensor]:
        out = {}
        for i, k in enumerate(self.levels):
            tau = torch.exp(self.log_tau[i]).clamp(1e-3, 100.0)
            out[k] = (zp[:, :k] * zc[:, :k]).sum(1) / tau
        return out


# ---------------------------------------------------------------- loss
def mrl_nce_loss(model: MLVVerifier, zp, zc, labels: torch.Tensor,
                 w_info: float = 0.5) -> tuple[torch.Tensor, dict]:
    """BCE per level on pos/neg pairs + in-batch InfoNCE over positives."""
    en = model.energies(zp, zc)
    y = labels.float()
    total, parts = 0.0, {}
    pos = (labels == 1).nonzero(as_tuple=True)[0]
    for i, k in enumerate(model.levels):
        s = en[k]
        bce = F.binary_cross_entropy_with_logits(s, y)
        info = s.new_zeros(())
        if len(pos) >= 2:
            sp = s[pos]
            tau = torch.exp(model.log_tau[i]).clamp(1e-3, 100.0)
            sim = (zp[pos, :k] @ zc[pos, :k].T) / tau
            info = F.cross_entropy(sim, torch.arange(len(pos), device=s.device))
        lvl = bce + w_info * info
        total = total + lvl
        parts[k] = float(lvl.item())
    return total / len(model.levels), parts


# ---------------------------------------------------------------- metrics (stdlib only)
def auroc(scores: list[float], labels: list[int]) -> float:
    pos = sorted((s for s, y in zip(scores, labels) if y == 1))
    neg = sorted((s for s, y in zip(scores, labels) if y == 0))
    n, m = len(pos), len(neg)
    if n == 0 or m == 0:
        return float("nan")
    wins = sum(1 for s in pos for t in neg if s > t)
    ties = sum(1 for s in pos for t in neg if s == t)
    return (wins + 0.5 * ties) / (n * m)


def pairwise_accuracy(scores: list[float], labels: list[int], seed: int = 0) -> float:
    """V1D-style proxy: each pos paired with a random neg; frac pos wins."""
    pi = [i for i, y in enumerate(labels) if y == 1]
    ni = [i for i, y in enumerate(labels) if y == 0]
    if not pi or not ni:
        return float("nan")
    rng = random.Random(seed)
    w = sum(1 for i in pi if scores[i] > scores[rng.choice(ni)])
    return w / len(pi)


def ece(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    probs = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    tot, acc = 0.0, 0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (p >= lo if b == n_bins - 1 else lo <= p < hi)]
        if idx:
            tot += len(idx) / len(probs) * abs(sum(probs[i] for i in idx) / len(idx)
                                               - sum(labels[i] for i in idx) / len(idx))
            acc += len(idx)
    return tot if acc else float("nan")


def retention_gate(train_scores, train_labels, test_scores, test_labels,
                   retention: float = 0.99) -> dict:
    """Threshold = score quantile holding `retention` of train positives;
    report test accepted-retention and FP suppression (1 - FPR)."""
    tp = sorted((s for s, y in zip(train_scores, train_labels) if y == 1))
    if not tp:
        return {"threshold": float("nan"), "retention": float("nan"), "fp_suppression": float("nan")}
    thr = tp[max(0, math.ceil((1 - retention) * len(tp)) - 1)]
    tpos = [s for s, y in zip(test_scores, test_labels) if y == 1]
    tneg = [s for s, y in zip(test_scores, test_labels) if y == 0]
    ret = sum(1 for s in tpos if s >= thr) / len(tpos) if tpos else float("nan")
    fpr = sum(1 for s in tneg if s >= thr) / len(tneg) if tneg else float("nan")
    return {"threshold": thr, "retention": ret,
            "fp_suppression": (1 - fpr) if tneg else float("nan")}


def all_metrics(scores, labels, tr_scores=None, tr_labels=None) -> dict:
    m = {"auroc": auroc(scores, labels),
         "pairwise_accuracy": pairwise_accuracy(scores, labels),
         "ece": ece(scores, labels)}
    if tr_scores is not None:
        m["gate"] = retention_gate(tr_scores, tr_labels, scores, labels)
    return m


# ---------------------------------------------------------------- data
POS_DECISIONS = {"accepted"}
NEG_DECISIONS = {"false_suggestion", "dismissed"}
SKIP_DECISIONS = {"correct_stop"}


def load_episode_pairs(ep_path: str | Path, confirm_path: str | Path | None = None) -> list[dict]:
    """Join episode (prompt, proposal) texts with per-point decisions.

    confirm.jsonl (if given) is the flattened label stream joined by row
    index; otherwise the embedded point decision/gt fields are used.
    Returns rows: {prompt, completion, label, group}."""
    eps = [json.loads(l) for l in open(ep_path)]
    confirm = None
    if confirm_path:
        confirm = [json.loads(l) for l in open(confirm_path)]
    rows, skipped, ci = [], 0, 0
    for ep in eps:
        group = ep.get("key", "unknown")
        for pt in ep.get("points", []):
            dec = pt.get("decision")
            if confirm is not None and ci < len(confirm):
                c = confirm[ci]
                dec = c.get("decision", dec)
            ci += 1
            if dec in SKIP_DECISIONS or dec is None:
                skipped += 1
                continue
            if dec in POS_DECISIONS:
                y = 1
            elif dec in NEG_DECISIONS:
                y = 0
            else:
                skipped += 1
                continue
            prop = pt.get("proposal") or ""
            if not prop.strip():
                skipped += 1
                continue
            rows.append({"prompt": pt.get("prompt") or "", "completion": prop,
                         "label": y, "group": str(group)})
    rows.sort(key=lambda r: (r["group"], r["label"]))
    return rows


def load_scenario_labels(req_path: str | Path) -> dict:
    """Auxiliary scenario labels (id -> 1/0); no prompt text in this file,
    so these rows cannot train (prompt, completion) pairs -- counted only."""
    out = {}
    for l in open(req_path):
        r = json.loads(l)
        if r.get("kind") != "scenario":
            continue
        out[str(r.get("id"))] = 1 if r.get("exact") else 0
    return out


def group_split(rows: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[list, list]:
    groups = sorted({r["group"] for r in rows})
    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: hashlib.md5(f"{seed}:{g}".encode()).hexdigest())
    n_test = max(1, round(len(order) * test_frac)) if len(order) > 1 else 0
    test_g = set(rng.sample(order, n_test) if n_test else [])
    # deterministic: take lowest-hash groups as test
    test_g = set(order[:n_test])
    return [r for r in rows if r["group"] not in test_g], [r for r in rows if r["group"] in test_g]


def synthetic_rows(n: int = 64, seed: int = 0) -> list[dict]:
    """Mock pairs with a learnable signal: pos completions echo the prompt
    head; neg completions are unrelated. Group-disjoint by construction."""
    rng = random.Random(seed)
    vocab = ["mean", "filter", "mutate", "ggplot", "lm(", "aes(", "data", "na.rm"]
    rows = []
    for i in range(n):
        w = [rng.choice(vocab) for _ in range(6)]
        prompt = " ".join(w)
        if i % 2 == 0:
            comp, y = " ".join(w[:3]) + " ...", 1
        else:
            comp, y = " ".join(rng.choice(vocab) for _ in range(3)) + " ???", 0
        rows.append({"prompt": prompt, "completion": comp, "label": y,
                     "group": f"g{i % 8}"})
    return rows


# ---------------------------------------------------------------- train/eval core
def collate(rows: list[dict], max_len: int = MAX_LEN):
    p_ids, p_m = batch_encode([r["prompt"] for r in rows], max_len)
    c_ids, c_m = batch_encode([r["completion"] for r in rows], max_len)
    y = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    return p_ids, p_m, c_ids, c_m, y


def run_train(rows: list[dict], out: str | Path | None = None, epochs: int = 3,
              lr: float = 1e-3, batch: int = 32, test_frac: float = 0.2,
              seed: int = 0, d_model: int = 256, max_len: int = MAX_LEN,
              device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    tr_rows, te_rows = group_split(rows, test_frac, seed)
    if not tr_rows or not te_rows:
        raise ValueError(f"degenerate split: {len(tr_rows)} train / {len(te_rows)} test rows")
    model = MLVVerifier(d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        rng = random.Random(seed + _)
        idx = list(range(len(tr_rows)))
        rng.shuffle(idx)
        model.train()
        for s in range(0, len(idx), batch):
            b = [tr_rows[i] for i in idx[s:s + batch]]
            p_ids, p_m, c_ids, c_m, y = (t.to(device) for t in collate(b, max_len))
            zp, zc = model.reps(p_ids, p_m, c_ids, c_m)
            loss, _ = mrl_nce_loss(model, zp, zc, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    rep = {"n_train": len(tr_rows), "n_test": len(te_rows),
           "train_groups": len({r["group"] for r in tr_rows}),
           "test_groups": len({r["group"] for r in te_rows})}
    with torch.no_grad():
        model.eval()
        out_m = {}
        for tag, rs in (("train", tr_rows), ("test", te_rows)):
            p_ids, p_m, c_ids, c_m, y = (t.to(device) for t in collate(rs, max_len))
            zp, zc = model.reps(p_ids, p_m, c_ids, c_m)
            en = model.energies(zp, zc)[max(model.levels)]
            out_m[tag] = (en.cpu().tolist(), y.cpu().tolist())
    tr_s, tr_y = out_m["train"]
    te_s, te_y = out_m["test"]
    rep["metrics_test"] = all_metrics(te_s, te_y, tr_s, tr_y)
    rep["kill"] = (rep["metrics_test"]["auroc"] is not None
                   and not math.isnan(rep["metrics_test"]["auroc"])
                   and rep["metrics_test"]["auroc"] < AUROC_KILL)
    if out:
        torch.save({"state": model.state_dict(), "d_model": d_model,
                    "levels": list(model.levels), "max_len": max_len}, str(out))
        Path(str(out) + ".json").write_text(json.dumps(rep, indent=1))
    return rep


def run_eval(ckpt: str | Path, rows: list[dict], test_frac: float = 0.0,
             seed: int = 0, max_len: int | None = None, device: str = "cpu") -> dict:
    blob = torch.load(str(ckpt), map_location=device, weights_only=True)
    model = MLVVerifier(d_model=blob.get("d_model", 256),
                        levels=tuple(blob.get("levels", NEST_LEVELS))).to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    ml = max_len or blob.get("max_len", MAX_LEN)
    if test_frac > 0:
        _, rows = group_split(rows, test_frac, seed)
    with torch.no_grad():
        p_ids, p_m, c_ids, c_m, y = (t.to(device) for t in collate(rows, ml))
        zp, zc = model.reps(p_ids, p_m, c_ids, c_m)
        en = model.energies(zp, zc)[max(model.levels)]
    return {"n": len(rows), "metrics": all_metrics(en.cpu().tolist(), y.cpu().tolist())}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MLV EBT verifier v1 (train/eval).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--episodes", default=None)
        p.add_argument("--confirm", default=None)
        p.add_argument("--test-frac", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--smoke", action="store_true",
                       help="synthetic-data CPU mock smoke (ignores data paths)")
    sub.choices["train"].add_argument("--out", default="mlv_verifier.pt")
    sub.choices["train"].add_argument("--epochs", type=int, default=3)
    sub.choices["train"].add_argument("--d-model", type=int, default=256)
    sub.choices["eval"].add_argument("--ckpt", default="mlv_verifier.pt")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.smoke:
        rows = synthetic_rows()
        seed, d_model, epochs = 0, 32, 2
    elif not a.episodes:
        ap.error("--episodes required (or --smoke)")
    else:
        rows = load_episode_pairs(a.episodes, a.confirm)
        seed, d_model, epochs = a.seed, 256, 3
    if a.cmd == "train":
        if not a.smoke:
            epochs, d_model = a.epochs, a.d_model
            seed = a.seed
        rep = run_train(rows, a.out if not a.smoke else None, epochs=epochs,
                        seed=seed, d_model=d_model, test_frac=a.test_frac)
        print(json.dumps(rep, indent=1))
        return 2 if rep.get("kill") else 0
    rep = run_eval(a.ckpt, rows, test_frac=0.0 if a.smoke else a.test_frac, seed=a.seed)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
