"""X5-S0: convergence-residual probe on the banked MD span head (CPU-only).

Queue docs/EXPERIMENT-QUEUE.md §2c X5, S0 leg (pre-registration verbatim):
rerun the banked `md_final.pt` on the 216-row harness with the existing
remasking schedule instrumented — per-row adjacent-step residual
r_k = SKL(p_k, p_{k-1}) (FRM paper Fig 6 instrument, arXiv 2606.29150 v2);
readouts = (a) AUROC of final-step residual vs span-exact correctness,
(b) commit-when-stable early-stop fraction (rows whose argmax stops
changing >=2 steps before the schedule ends). Diagnostic either way:
AUROC >= 0.65 OR >= 20% early-stoppable rows = confidence signal +
free-latency lever exist TODAY on the banked head; ~chance on both =
stability is FPF-created, S1 carries the whole claim (flag, not kill).

Operationalization (fixed before running; the FRM instrument re-derived
for the remasking sampler per intel doc §3 M3 / non-transfer #2):

* The remasking sampler freezes top-confidence positions per step, so a
  naive "successive distributions" over OPEN positions degenerates as the
  open set shrinks. The faithful FRM analog: every step, probe the
  model's full predictive distribution p_k at ALL span-region positions
  (frozen ones included — the trunk is bidirectional, so their
  distributions keep evolving as neighbors freeze; the probe is
  read-only and never alters the schedule). p_k is the softmax over the
  full extended vocab (130,562 incl [MASK]/[EMPTY] rows) at temperature
  1.0, exactly `_chunked_probs` numerics.
* r_k = mean_i SKL(p_k[i], p_{k-1}[i]) over all span positions, SKL =
  KL(p||q)+KL(q||p), nats; defined for k >= 2. Secondary r_open_k: the
  same mean restricted to positions still OPEN at step k (the shrinking
  set).
* final-step residual R = r_{n_fwd} (n_fwd = forwards actually run; the
  sampler early-exits when all positions are frozen, as in sample.py).
* AUROC(R) is reported with INCORRECT as the positive class (paper:
  residual separates correct from incorrect; under FPF AUROC 1.00,
  vanilla 0.50). Rank statistic w/ tie handling + percentile bootstrap.
* argmax stability: a_k = argmax_v p_k over span positions. k* = first
  step of the stable tail (a_k* == a_j for all j >= k*). Early-stoppable
  iff k* <= n_fwd - 2 (the pre-registered ">=2 steps before the schedule
  ends", against the schedule AS RUN for that row); also reported
  against the nominal step count. STRICT early-stop (latency-honest):
  additionally the committed output == a_k* (stopping at k* and
  committing the all-position argmax would reproduce the banked output
  token-for-token). fixed-by-k: k* <= k (paper: fixed point by k<=8 in
  97.5% under FPF).
* Validity anchor: temperature-0 greedy replay should reproduce the
  banked exact counts (MD@32 = 15/216, MD@64 = 16/216) up to fp32
  CPU-vs-GPU reduction-order drift (~+-1 row).

Usage (CPU only — CUDA_VISIBLE_DEVICES="" enforced by assertion):
  .venv/bin/python -m experiments.training.poc_diff.x5_s0_residual \
      --steps 64 32 --out results_x5_s0
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.training.poc_diff import TMP
from experiments.training.poc_diff.eval_spans import (
    BUCKETS, load_md, norm_lines, point_metrics,
)
from experiments.training.poc_diff.sample import _chunked_probs, decode_span

DEFAULT_CKPT = "/mnt/h/sepalith/runs/poc_diff/md_final.pt"
BUCKET_NAMES = [b[0] for b in BUCKETS]


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def skl_rows(p, q):
    """Row-wise symmetric KL between two (N,V) prob matrices (nats)."""
    lp, lq = p.log(), q.log()
    kl_pq = (p * (lp - lq)).sum(-1)
    kl_qp = (q * (lq - lp)).sum(-1)
    return kl_pq + kl_qp


@torch.no_grad()
def replay_row(model, tok, row, steps, chunk=512):
    """One row's remasking schedule at temperature 0 (greedy confidence),
    instrumented per the module docstring. Mirrors sample.sample_spans
    batch-1 semantics exactly: freeze the top-ceil(L/steps) confident OPEN
    positions per step; probe-all-positions is read-only."""
    device = next(model.parameters()).device
    prompt_ids, L = row["prompt_ids"], max(1, len(row["span_ids"]))
    T_ctx, T = len(prompt_ids), len(prompt_ids) + L
    x = torch.full((1, T), 1, dtype=torch.long, device=device)  # eos pad
    x[0, :T_ctx] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    span_pos = torch.zeros(1, T, dtype=torch.bool, device=device)
    span_pos[0, T_ctx:T_ctx + L] = True
    x[0, T_ctx:T_ctx + L] = model.mask_id
    valid = torch.ones(1, T, dtype=torch.bool, device=device)
    attn_mask = valid[:, None, None, :]

    per_step = max(1, -(-L // steps))  # ceil, sample.py convention
    frozen = torch.zeros(1, T, dtype=torch.bool, device=device)
    pred = torch.zeros(T, dtype=torch.long, device=device)
    prev, prev_open_mask = None, None
    r_all, r_open, conf_traj, argmaxes = [], [], [], []
    n_fwd = 0
    for step in range(1, steps + 1):
        open_pos = span_pos & ~frozen
        if not open_pos.any():
            break
        n_fwd += 1
        h = model.trunk(x, probe=False, attn_mask=attn_mask)
        h_span = h[0, T_ctx:T_ctx + L]                     # (L, d)
        probs = _chunked_probs(h_span, model.embed.weight, chunk=chunk,
                               temperature=0.0)            # (L, V) fp32
        amax = probs.argmax(-1)
        argmaxes.append(amax.to(torch.int32))
        conf_traj.append(float((probs.max(-1).values).mean()))
        if prev is not None:
            skl = skl_rows(probs, prev)                    # (L,)
            r_all.append(float(skl.mean()))
            if prev_open_mask is not None and prev_open_mask.any():
                r_open.append(float(skl[prev_open_mask].mean()))
            else:
                r_open.append(None)
        prev = probs
        prev_open_mask = (open_pos[0, T_ctx:T_ctx + L]).clone()
        # freeze the top-`per_step` confident OPEN positions (greedy pick)
        open_local = (~frozen[0, T_ctx:T_ctx + L]).nonzero(as_tuple=True)[0]
        pconf, picks = probs.max(-1)                       # greedy: conf=argmax prob
        k = min(per_step, int(open_local.numel()))
        order = pconf[open_local].argsort(descending=True)[:k]
        sel = open_local[order]
        frozen[0, T_ctx + sel] = True
        pred[T_ctx + sel] = picks[sel]
        x[0, T_ctx + sel] = picks[sel]

    committed = pred[T_ctx:T_ctx + L].tolist()
    final_amax = argmaxes[-1].tolist() if argmaxes else []
    # k*: first step of the stable argmax tail (a_k* == a_j for all j >= k*)
    k_star = None
    if len(argmaxes) >= 1:
        tail = argmaxes[-1]
        for j in range(len(argmaxes) - 1, -1, -1):
            if torch.equal(argmaxes[j], tail):
                k_star = j + 1                             # steps are 1-based
            else:
                break
    n = len(argmaxes)
    text = decode_span(tok, pred[T_ctx:T_ctx + L], model.empty_id)
    m = point_metrics(text, row["span_text"], tok)
    drift = int(sum(1 for a, b in zip(final_amax, committed) if a != b))
    return dict(
        idx=row.get("idx"), span_len=row["span_len"],
        bucket=bucket_of(row["span_len"]), kind=row.get("kind"),
        package=row.get("package"), steps_nominal=steps, n_fwd=n_fwd, L=L,
        r_all=r_all, r_open=r_open, conf_traj=conf_traj,
        r_final=r_all[-1] if r_all else None,
        k_star=k_star,
        early_stoppable=bool(k_star is not None and k_star <= n - 2),
        early_stoppable_nominal=bool(k_star is not None and k_star <= steps - 2),
        # strict: stopping at k* and committing the all-position argmax
        # reproduces the run's output TEXT token-for-token after decode
        strict_early_stop=bool(
            k_star is not None and k_star <= n - 2
            and decode_span(tok, argmaxes[k_star - 1], model.empty_id) == text),
        fixed_by_8=bool(k_star is not None and k_star <= 8),
        committed_ids=committed, pred_text=text,
        exact=m["exact"], edit_sim=m["edit_sim"],
        post_freeze_drift=drift,
    )


def run(args):
    assert not torch.cuda.is_available(), "X5-S0 is CPU-only by protocol"
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")

    ckpt = args.md_ckpt
    if ckpt.startswith("/mnt/"):                            # drvfs is slow
        local = Path(TMP) / Path(ckpt).name
        if not local.exists():
            print(f"copying ckpt to tmpfs: {ckpt}", flush=True)
            import shutil
            shutil.copy(ckpt, local)
        ckpt = str(local)
    print(f"loading {ckpt} (threads={args.threads})", flush=True)
    t0 = time.time()
    model = load_md(ckpt, device)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in
            open(f"{TMP}/eval_triples.jsonl", encoding="utf-8")]
    for i, r in enumerate(rows):
        r["idx"] = i
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows: {len(rows)}", flush=True)

    for steps in args.steps:
        out_path = outdir / f"residuals_steps{steps}.jsonl"
        done = 0
        if out_path.exists() and args.resume:
            done = sum(1 for _ in open(out_path))
            print(f"resume: {done} rows already in {out_path}", flush=True)
        with open(out_path, "a", encoding="utf-8") as g:
            t0 = time.time()
            for row in rows[done:]:
                rec = replay_row(model, tok, row, steps)
                g.write(json.dumps(rec) + "\n")
                g.flush()
                done += 1
                if done % 10 == 0 or done == len(rows):
                    rate = (time.time() - t0) / max(1, done - (0 if not args.resume else 0))
                    print(f"steps={steps}: {done}/{len(rows)} rows "
                          f"({(time.time()-t0)/max(1,done):.1f}s/row, "
                          f"elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
                    t0 = time.time() if done == len(rows) else t0
        print(f"steps={steps} done -> {out_path}", flush=True)


def auroc(scores, labels):
    """AUROC with `positive` label==True as the positive class.
    Ties counted half. Rank-based (Mann-Whitney)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sv = s[order]
    i = 0
    while i < len(sv):                                      # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def bootstrap_auroc(scores, labels, n=2000, seed=1273, alpha=0.05):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        a = auroc(s[idx], y[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def analyze(args):
    outdir = Path(args.out)
    for steps in args.steps:
        path = outdir / f"residuals_steps{steps}.jsonl"
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        n = len(recs)
        exact = [r["exact"] for r in recs]
        R = [r["r_final"] for r in recs if r["r_final"] is not None]
        # pre-registered convention: positive class = INCORRECT (higher
        # residual should predict incorrect; paper: AUROC 1.00 FPF / 0.50
        # vanilla)
        labels = [not bool(r["exact"]) for r in recs if r["r_final"] is not None]
        a = auroc(R, labels)
        ci = bootstrap_auroc(R, labels)
        out = dict(
            steps=steps, n_rows=n,
            exact_rate=float(np.mean(exact)), exact_count=int(np.sum(exact)),
            auroc_incorrect=a, auroc_ci95=ci,
            early_stop_frac=float(np.mean([r["early_stoppable"] for r in recs])),
            early_stop_frac_nominal=float(
                np.mean([r["early_stoppable_nominal"] for r in recs])),
            strict_early_stop_frac=float(
                np.mean([r["strict_early_stop"] for r in recs])),
            fixed_by_8_frac=float(np.mean([r["fixed_by_8"] for r in recs])),
            median_k_star=float(np.median(
                [r["k_star"] for r in recs if r["k_star"] is not None])),
            median_n_fwd=float(np.median([r["n_fwd"] for r in recs])),
            mean_saved_fwd_frac=float(np.mean([
                max(0, r["n_fwd"] - (r["k_star"] or r["n_fwd"]))
                / max(1, r["n_fwd"]) for r in recs])),
            post_freeze_drift_frac=float(np.mean(
                [r["post_freeze_drift"] / max(1, r["L"]) for r in recs])),
            buckets={},
            residual_curve={},
        )
        for b in BUCKET_NAMES:
            sel = [r for r in recs if r["bucket"] == b]
            if not sel:
                continue
            Rs = [r["r_final"] for r in sel if r["r_final"] is not None]
            lb = [not bool(r["exact"]) for r in sel
                  if r["r_final"] is not None]   # positive = incorrect
            out["buckets"][b] = dict(
                n=len(sel), exact=int(np.sum([r["exact"] for r in sel])),
                r_final_median=float(np.median(Rs)) if Rs else None,
                r_final_p25=float(np.percentile(Rs, 25)) if Rs else None,
                r_final_p75=float(np.percentile(Rs, 75)) if Rs else None,
                early_stop_frac=float(np.mean([r["early_stoppable"] for r in sel])),
                fixed_by_8_frac=float(np.mean([r["fixed_by_8"] for r in sel])),
                median_k_star=float(np.median(
                    [r["k_star"] for r in sel if r["k_star"] is not None]))
                if any(r["k_star"] is not None for r in sel) else None,
                auroc=auroc(Rs, lb),
                post_freeze_drift_frac=float(np.mean(
                    [r["post_freeze_drift"] / max(1, r["L"]) for r in sel])),
            )
        # median residual-vs-k curve per bucket (rows with n_fwd >= k)
        ks = [2, 4, 8, 16, 32, 48, 64]
        for b in BUCKET_NAMES:
            sel = [r for r in recs if r["bucket"] == b]
            curve = {}
            for k in ks:
                vals = [r["r_all"][k - 2] for r in sel
                        if r["n_fwd"] >= k and len(r["r_all"]) >= k - 1]
                if len(vals) >= 5:
                    curve[k] = round(float(np.median(vals)), 6)
            if curve:
                out["residual_curve"][b] = curve
        with open(outdir / f"analysis_steps{steps}.json", "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--steps", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results_x5_s0"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "analyze"])
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
