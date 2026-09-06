"""X5-S1 eval: the 216-row harness (verbatim metrics) on the S1 head +
recurrent depth k in {2,4,8} + post-FPF residual instrument + verdict.

Queue X5 S1 row (pre-registered, verbatim): "eval = the 216-row harness
verbatim + recurrent depth k in {2,4,8}. Verdict: WINNER-SC iff exact >
0.0741 (banked MD@64 anchor) at <=2x anchor NFE AND the 51-256 bucket
lifts off 0.000; win only at >=4x NFE -> latency-negative, write-the-
negative; 11-50 regression vs 0.110 -> KILL."

Legs (nominal NFE = steps x depth; as-run n_fwd also recorded — the
sampler early-exits when all positions freeze):
  carry-active (the S1 system):
    k=1 : steps 8/16/32/64 — the harness-verbatim grid at 1x anchor NFE
          (greedy quality + latency per row, best-of-8 temperature-1.0
          multimodality at every grid point, eval_spans conventions)
    k>1 : (8,2) (8,4) (8,8) (16,2) (16,4) (16,8) (32,2) (32,4) (32,8)
          — greedy quality + latency; the equal-NFE depth ladder:
          NFE 64 = (64,1)/(32,2)/(16,4)/(8,8); NFE 128 = (64,2)/(32,4)/
          (16,8); NFE 256 = (32,8). Paper claim under test: recurrent
          carry converts DEPTH into accuracy where schedule passes
          cannot (the 51-256 bucket).
  carry-inactive diagnostic (sampler carry=None): steps 32/64 — did the
          continuation degrade the banked zero-carry path? (house
          expectation: near-anchor; 50% of training was null-carry)
  residual replay (S0 instrument, x5_s0_residual semantics, carry-active):
          steps 64 and 32 at k=1 (S0-comparable adjacent-pass residual,
          AUROC, abstain sizing) + steps 32 at k=8 (the paper-faithful
          recurrent residual: successive recurrent distributions at a
          fixed state).

Banked anchors (eval_results.json, reproduced bit-exact by X5-S0):
  MD@64 exact 0.0741 (16/216), MD@32 0.0694 (15/216); buckets 11-50
  0.1176 @64 / 0.1103 @32 (the row's 0.110), 51-256 0.000 everywhere.

Usage (GPU):
  uv run python -m experiments.training.poc_diff.x5_s1_eval \
      --ckpt /mnt/h/sepalith/runs/x5_s1/x5_s1_b_final.pt --out results_x5_s1
Subcommands: run (legs) / resid (residual replay) / verdict (analysis).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.training.poc_diff import TMP
from experiments.training.poc_diff.eval_spans import (
    BUCKETS, K_BESTOF, norm_lines, point_metrics,
)
from experiments.training.poc_diff.sample import _chunked_probs, decode_span, \
    sample_spans
from experiments.training.poc_diff.x5_s1_sample import load_md_x5, \
    sample_spans_rc
from experiments.training.poc_diff.x5_s0_residual import auroc, \
    bootstrap_auroc, skl_rows

DEFAULT_CKPT = "/mnt/h/sepalith/runs/poc_diff/md_final.pt"
ANCHOR_EXACT = {64: 16 / 216, 32: 15 / 216}
ANCHOR_1150 = {64: 0.11764705882352941, 32: 0.11029411764705882}
BUCKET_NAMES = [b[0] for b in BUCKETS]
GRID_K1 = (8, 16, 32, 64)
GRID_DEPTH = [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8),
              (32, 2), (32, 4), (32, 8)]


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def load_rows(limit=0):
    rows = [json.loads(l) for l in
            open(f"{TMP}/eval_triples.jsonl", encoding="utf-8")]
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows[:limit] if limit else rows


def leg_key(steps, depth, carry):
    return f"s{steps}k{depth}" + ("" if carry else "null")


def run_leg(model, tok, rows, steps, depth, carry=True, bo8=False,
            device=None):
    """One (steps, depth) leg: greedy quality + latency per row (batch-1,
    eval_spans conventions); optional best-of-8 multimodality."""
    per_row, bo8_rows = [], []
    for r in rows:
        L = max(1, len(r["span_ids"]))
        if carry:
            out = sample_spans_rc(model, [r["prompt_ids"]], [L], steps,
                                  depth=depth, temperature=0.0)
        else:
            out = sample_spans(model, [r["prompt_ids"]], [L], steps,
                               temperature=0.0)
        text = decode_span(tok, out["pred_ids"][0, :L], model.empty_id)
        per_row.append(dict(
            metrics=point_metrics(text, r["span_text"], tok),
            lat_ms=out["latency_ms"], n_fwd=out["n_fwd"],
            span_len=r["span_len"], bucket=bucket_of(r["span_len"])))
    if bo8:
        for r in rows:
            L = max(1, len(r["span_ids"]))
            g = torch.Generator(device=device).manual_seed(1273 + steps)
            if carry:
                out = sample_spans_rc(model, [r["prompt_ids"]] * K_BESTOF,
                                      [L] * K_BESTOF, steps, depth=depth,
                                      temperature=1.0, generator=g)
            else:
                out = sample_spans(model, [r["prompt_ids"]] * K_BESTOF,
                                   [L] * K_BESTOF, steps, temperature=1.0,
                                   generator=g)
            texts = {norm_lines(decode_span(tok, out["pred_ids"][k, :L],
                                            model.empty_id))
                     for k in range(K_BESTOF)}
            bo8_rows.append(dict(
                bo8=int(norm_lines(r["span_text"]) in texts),
                distinct=len(texts) / K_BESTOF))
    summary = dict(
        steps=steps, depth=depth, carry=carry, nfe_nominal=steps * depth,
        nfe_asrun_mean=float(np.mean([p["n_fwd"] for p in per_row])),
        exact=float(np.mean([p["metrics"]["exact"] for p in per_row])),
        exact_count=int(np.sum([p["metrics"]["exact"] for p in per_row])),
        edit_sim=float(np.mean([p["metrics"]["edit_sim"] for p in per_row])),
        prefix8=float(np.mean([p["metrics"]["prefix8"] for p in per_row])),
        position=float(np.mean([p["metrics"]["position"] for p in per_row])),
        lat_p50=pct([p["lat_ms"] for p in per_row], 50),
        lat_p95=pct([p["lat_ms"] for p in per_row], 95),
        buckets={})
    if bo8_rows:
        summary["best_of_8"] = float(np.mean([b["bo8"] for b in bo8_rows]))
        summary["distinct"] = float(np.mean([b["distinct"] for b in bo8_rows]))
    for name in BUCKET_NAMES:
        sel = [p for p in per_row if p["bucket"] == name]
        if sel:
            summary["buckets"][name] = dict(
                n=len(sel),
                exact=float(np.mean([p["metrics"]["exact"] for p in sel])))
    return summary, per_row, bo8_rows


# ---------------- residual replay (S0 instrument, carry-active) ---------

@torch.no_grad()
def replay_row_x5(model, tok, row, steps, depth=1, chunk=512):
    """x5_s0_residual.replay_row semantics with the carry active: the
    span distributions are probed (read-only) at ALL span positions every
    forward; r_step = SKL between adjacent schedule-step distributions;
    r_rec (depth>1) = SKL between adjacent recurrent distributions within
    a schedule step (fixed input — the paper's Fig-6 instrument)."""
    device = next(model.parameters()).device
    prompt_ids, L = row["prompt_ids"], max(1, len(row["span_ids"]))
    T_ctx, T = len(prompt_ids), len(prompt_ids) + L
    x = torch.full((1, T), 1, dtype=torch.long, device=device)
    x[0, :T_ctx] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    span_pos = torch.zeros(1, T, dtype=torch.bool, device=device)
    span_pos[0, T_ctx:T_ctx + L] = True
    x[0, T_ctx:T_ctx + L] = model.mask_id
    valid = torch.ones(1, T, dtype=torch.bool, device=device)
    attn_mask = valid[:, None, None, :]

    per_step = max(1, -(-L // steps))
    frozen = torch.zeros(1, T, dtype=torch.bool, device=device)
    pred = torch.zeros(T, dtype=torch.long, device=device)
    prev_step, prev_rec = None, None
    r_all, r_rec_last, conf_traj, argmaxes = [], None, [], []
    n_fwd = 0
    for _step in range(steps):
        open_pos = span_pos & ~frozen
        if not open_pos.any():
            break
        carry = None
        probs = None
        for _j in range(depth):
            h = model.trunk(x, probe=False, attn_mask=attn_mask, carry=carry)
            n_fwd += 1
            h_span = h[0, T_ctx:T_ctx + L]
            probs = _chunked_probs(h_span, model.embed.weight, chunk=chunk,
                                   temperature=0.0)
            if prev_rec is not None:
                r_rec_last = float(skl_rows(probs, prev_rec).mean())
            prev_rec = probs
            # the carry IS the probed distribution (softmax @ E), exact
            # logits_to_carry math — reuses the probe instead of a second
            # (L,V) pass
            c = torch.zeros(1, T, h.shape[-1], device=device,
                            dtype=torch.float32)
            c[0, T_ctx:T_ctx + L] = probs @ model.embed.weight.float()
            carry = c
        argmaxes.append(probs.argmax(-1).to(torch.int32))
        conf_traj.append(float((probs.max(-1).values).mean()))
        if prev_step is not None:
            r_all.append(float(skl_rows(probs, prev_step).mean()))
        prev_step = probs
        prev_rec = None  # recurrent residuals don't cross schedule steps
        # freeze the top-`per_step` confident OPEN positions (greedy pick)
        open_local = (~frozen[0, T_ctx:T_ctx + L]).nonzero(as_tuple=True)[0]
        pconf, picks = probs.max(-1)
        k = min(per_step, int(open_local.numel()))
        order = pconf[open_local].argsort(descending=True)[:k]
        sel = open_local[order]
        frozen[0, T_ctx + sel] = True
        pred[T_ctx + sel] = picks[sel]
        x[0, T_ctx + sel] = picks[sel]

    committed = pred[T_ctx:T_ctx + L].tolist()
    final_amax = argmaxes[-1].tolist() if argmaxes else []
    k_star = None
    if argmaxes:
        tail = argmaxes[-1]
        for j in range(len(argmaxes) - 1, -1, -1):
            if torch.equal(argmaxes[j], tail):
                k_star = j + 1
            else:
                break
    n = len(argmaxes)
    text = decode_span(tok, pred[T_ctx:T_ctx + L], model.empty_id)
    m = point_metrics(text, row["span_text"], tok)
    drift = int(sum(1 for a, b in zip(final_amax, committed) if a != b))
    return dict(
        idx=row.get("idx"), span_len=row["span_len"],
        bucket=bucket_of(row["span_len"]), steps_nominal=steps, depth=depth,
        n_fwd=n_fwd, L=L, r_all=r_all, r_rec_final=r_rec_last,
        conf_traj=conf_traj, r_final=r_all[-1] if r_all else None,
        k_star=k_star,
        early_stoppable=bool(k_star is not None and k_star <= n - 2),
        fixed_by_8=bool(k_star is not None and k_star <= 8),
        committed_ids=committed, pred_text=text, exact=m["exact"],
        edit_sim=m["edit_sim"], post_freeze_drift=drift,
    )


def resid_analyze(recs, label):
    R = [r["r_final"] for r in recs if r["r_final"] is not None]
    labels = [not bool(r["exact"]) for r in recs if r["r_final"] is not None]
    out = dict(label=label, n=len(recs),
               exact_count=int(np.sum([r["exact"] for r in recs])),
               auroc_incorrect=auroc(R, labels),
               auroc_ci95=bootstrap_auroc(R, labels),
               fixed_by_8_frac=float(np.mean([r["fixed_by_8"] for r in recs])),
               median_k_star=float(np.median(
                   [r["k_star"] for r in recs if r["k_star"] is not None])),
               median_n_fwd=float(np.median([r["n_fwd"] for r in recs])),
               post_freeze_drift_frac=float(np.mean(
                   [r["post_freeze_drift"] / max(1, r["L"]) for r in recs])),
               buckets={}, abstain={})
    for b in BUCKET_NAMES:
        sel = [r for r in recs if r["bucket"] == b]
        if not sel:
            continue
        Rs = [r["r_final"] for r in sel if r["r_final"] is not None]
        out["buckets"][b] = dict(
            n=len(sel), exact=int(np.sum([r["exact"] for r in sel])),
            r_final_median=float(np.median(Rs)) if Rs else None,
            early_stop_frac=float(np.mean([r["early_stoppable"] for r in sel])),
            fixed_by_8_frac=float(np.mean([r["fixed_by_8"] for r in sel])))
    # abstain sizing (S0 convention): thresholds at the max/2nd-max exact
    # row residual -> incorrect suppressed at 0/1 exact lost
    ex = sorted(r["r_final"] for r in recs
                if r["exact"] and r["r_final"] is not None)
    inr = [r["r_final"] for r in recs
           if not r["exact"] and r["r_final"] is not None]
    if ex:
        for tag, thr in [("thr0", ex[-1]), ("thr1", ex[-2] if len(ex) > 1
                                            else ex[-1])]:
            lost = sum(1 for v in ex if v > thr)
            supp = sum(1 for v in inr if v > thr)
            out["abstain"][tag] = dict(
                threshold=thr, exact_lost=lost,
                incorrect_suppressed_frac=(supp / len(inr)) if inr else None)
    # recurrent-residual readout (paper-comparable, depth>1 legs)
    rr = [(r["r_rec_final"], r["exact"]) for r in recs
          if r.get("r_rec_final") is not None]
    if rr:
        out["auroc_rec"] = auroc([a for a, _ in rr], [not e for _, e in rr])
    return out


def do_run(args):
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    model = load_md_x5(args.ckpt, device)
    rows = load_rows(args.limit)
    print(f"rows: {len(rows)}; ckpt {args.ckpt}", flush=True)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    legs, all_rows = {}, {}
    t0 = time.time()
    plan = [(s, 1, True) for s in GRID_K1] + \
        [(s, k, True) for s, k in GRID_DEPTH] + \
        [(32, 1, False), (64, 1, False)]
    for steps, depth, carry in plan:
        key = leg_key(steps, depth, carry)
        bo8 = carry and depth == 1 and steps in GRID_K1
        summary, per_row, _ = run_leg(model, tok, rows, steps, depth,
                                      carry=carry, bo8=bo8, device=device)
        legs[key] = summary
        all_rows[key] = per_row
        with open(outdir / f"rows_{key}.jsonl", "w") as f:
            for p in per_row:
                f.write(json.dumps(p) + "\n")
        print(f"[{time.time()-t0:7.0f}s] {key}: exact={summary['exact']:.4f} "
              f"(x{summary['exact_count']}) 51-256="
              f"{summary['buckets'].get('51-256', {}).get('exact', float('nan')):.3f} "
              f"nfe={summary['nfe_nominal']} asrun={summary['nfe_asrun_mean']:.1f}",
              flush=True)
    with open(outdir / "eval_x5_s1.json", "w") as f:
        json.dump(legs, f, indent=1)
    print(f"wrote {outdir}/eval_x5_s1.json", flush=True)


def do_resid(args):
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    model = load_md_x5(args.ckpt, device)
    rows = load_rows(args.limit)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    analyses = {}
    for steps, depth in [(64, 1), (32, 1), (32, 8)]:
        key = f"steps{steps}d{depth}"
        path = outdir / f"residuals_{key}.jsonl"
        done = 0
        if path.exists() and args.resume:
            done = sum(1 for _ in open(path))
            print(f"resume: {done} rows in {path}", flush=True)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as g:
            for row in rows[done:]:
                rec = replay_row_x5(model, tok, row, steps, depth=depth)
                g.write(json.dumps(rec) + "\n")
                g.flush()
                done += 1
                if done % 20 == 0 or done == len(rows):
                    print(f"{key}: {done}/{len(rows)} "
                          f"({(time.time()-t0)/max(1,done):.2f}s/row)",
                          flush=True)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        analyses[key] = resid_analyze(recs, key)
        print(json.dumps(analyses[key], indent=1), flush=True)
    with open(outdir / "analysis_x5_s1.json", "w") as f:
        json.dump(analyses, f, indent=1)
    print(f"wrote {outdir}/analysis_x5_s1.json", flush=True)


def do_verdict(args):
    outdir = Path(args.out)
    legs = json.load(open(outdir / "eval_x5_s1.json"))
    lines = ["# X5-S1 verdict computation (pre-registered bars)", ""]
    cands128 = {k: v for k, v in legs.items()
                if v["carry"] and v["nfe_nominal"] <= 128}
    best128 = max(cands128.items(), key=lambda kv: kv[1]["exact"])
    cands64 = {k: v for k, v in legs.items()
               if v["carry"] and v["nfe_nominal"] <= 64}
    best64 = max(cands64.items(), key=lambda kv: kv[1]["exact"])
    hi = {k: v for k, v in legs.items()
          if v["carry"] and v["nfe_nominal"] >= 256}
    besthi = max(hi.items(), key=lambda kv: kv[1]["exact"]) if hi else None

    def lift(v):
        return v["buckets"].get("51-256", {}).get("exact", 0.0)

    lines.append(f"best NFE<=128 : {best128[0]} exact={best128[1]['exact']:.4f} "
                 f"vs anchor 0.0741 -> "
                 f"{'PASS' if best128[1]['exact'] > ANCHOR_EXACT[64] else 'fail'}")
    lines.append(f"  51-256 lift-off: {lift(best128[1]):.4f} -> "
                 f"{'LIFTS' if lift(best128[1]) > 0 else 'zero'}")
    b1150 = best128[1]["buckets"].get("11-50", {}).get("exact", float("nan"))
    lines.append(f"  11-50: {b1150:.4f} vs 0.110 -> "
                 f"{'regression (KILL)' if b1150 < 0.110 else 'no regression'}")
    lines.append(f"best NFE<=64  : {best64[0]} exact={best64[1]['exact']:.4f} "
                 f"vs 32-anchor 0.0694")
    if besthi:
        lines.append(f"best NFE>=256 : {besthi[0]} "
                     f"exact={besthi[1]['exact']:.4f} lift={lift(besthi[1]):.4f}")
    # WINNER-SC: any <=128 leg with exact>anchor AND 51-256 lifted
    winners = [(k, v) for k, v in cands128.items()
               if v["exact"] > ANCHOR_EXACT[64] and lift(v) > 0
               and v["buckets"].get("11-50", {}).get("exact", 1.0) >= 0.110]
    late = [(k, v) for k, v in hi.items()
            if v["exact"] > ANCHOR_EXACT[64] and lift(v) > 0]
    if winners:
        verdict = "WINNER-SC"
    elif late:
        verdict = "LATENCY-NEGATIVE (win only at >=4x NFE — write-the-negative)"
    else:
        verdict = "NO-WIN"
    lines.append(f"VERDICT: {verdict}")
    if winners:
        k, v = max(winners, key=lambda kv: kv[1]["exact"])
        lines.append(f"  winning leg {k}: exact {v['exact']:.4f}, 51-256 "
                     f"{lift(v):.4f}, 11-50 "
                     f"{v['buckets'].get('11-50', {}).get('exact', float('nan')):.4f}")
    # equal-NFE depth ladder
    lines.append("")
    lines.append("equal-NFE ladder (exact): NFE 64: " + " ".join(
        f"{leg_key(s, k, True)}={legs[leg_key(s, k, True)]['exact']:.4f}"
        for s, k in [(64, 1), (32, 2), (16, 4), (8, 8)]
        if leg_key(s, k, True) in legs))
    lines.append("NFE 128: " + " ".join(
        f"{leg_key(s, k, True)}={legs[leg_key(s, k, True)]['exact']:.4f}"
        for s, k in [(64, 2), (32, 4), (16, 8)]
        if leg_key(s, k, True) in legs))
    lines.append("NFE 256: " + " ".join(
        f"{leg_key(s, k, True)}={legs[leg_key(s, k, True)]['exact']:.4f}"
        for s, k in [(32, 8)] if leg_key(s, k, True) in legs))
    txt = "\n".join(lines)
    print(txt)
    with open(outdir / "verdict_x5_s1.txt", "w") as f:
        f.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent /
                                         "results_x5_s1"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "resid", "verdict"])
    args = ap.parse_args()
    if args.cmd == "run":
        do_run(args)
    elif args.cmd == "resid":
        do_resid(args)
    else:
        do_verdict(args)


if __name__ == "__main__":
    main()
