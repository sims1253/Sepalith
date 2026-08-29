#!/usr/bin/env python3
"""Task 5 step 3: three-way eval runner — POC-DDOT.

Scores the DDOT pre-registered comparison
(docs/research/poc-ddot-ot-coupling-plan-2026-08-26.md):
  (a) POC-DIFF's best arm (AR or MD, whichever won their Task 6);
  (b) POC-DIFF diffusion + CAL-style length search (cal_length.py — the
      honest non-OT control; no retrain);
  (c) diffusion + OT (the Task-4 OT twin sampled with sample_ot_spans).
Metric layer ADDED on top of poc_diff's eval_spans harness (reused via
import for the point metrics): length-MAE, position-MSE, [EMPTY]
precision/recall, per-bucket breakdown.

Position-MSE operationalization (documented, load-bearing): in the astfim
PSM anatomy the GT insertion point is ALWAYS the cursor, so insertion-
point error degenerates across arms unless read as the OT arm's SPURIOUS
re-anchoring: mean squared (first-snapped-slot / window). Fixed-window
arms anchor exactly and score 0; the OT arm matches 0 iff its positions
stay anchored. LOWER is better for everyone — under the frozen kill test
the OT arm cannot "beat" the baselines on this metric, only fail it;
that is the correct reading, not a loophole.

[EMPTY] precision/recall: astfim has zero GT empty spans (pocdiff's data
reality note) — recall is reported degenerate with n_gt_empty logged;
precision is live (of rows predicted empty, how many should be).

Run (checkpoints pending: md_final ~18:00, OT twin after the queued
Task-4 run):
  python3 -m experiments.training.poc_ddot.eval_ot \
      --md-ckpt /mnt/h/sepalith/runs/poc_diff/md_final.pt \
      --ot-ckpt /mnt/h/sepalith/runs/poc_ddot/ot_final.pt \
      [--base-results /tmp/poc_diff/eval_results.json] [--limit N]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import cal_length as CAL
    from . import sample_ot
    from .sample_ot import sample_ot_spans
except ImportError:
    import cal_length as CAL
    import sample_ot
    from sample_ot import sample_ot_spans

from experiments.training.poc_diff import EMPTY_ID, TMP
from experiments.training.poc_diff.eval_spans import (
    BUCKETS, GEN_TOK_CAP, norm_lines, point_metrics)
from experiments.training.poc_diff.sample import decode_span, sample_spans

# Convenience re-export so tests/plumbing can reach the CAL pick from here.
pick_length = CAL.pick_length


# ---------------- added metrics ----------------

def length_mae(rows):
    """Mean |pred_len - gt_len| over rows (token lengths, text-level)."""
    if not rows:
        return 0.0
    return float(np.mean([abs(r["pred_len"] - r["gt_len"]) for r in rows]))


def position_mse(rows, window):
    """Mean squared spurious re-anchoring: (first_slot / window)^2.

    Rows without a `first_slot` (fixed-window arms) anchor exactly -> 0.
    """
    vals = [(r.get("first_slot", 0) / window) ** 2 for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def is_pred_empty(region_ids):
    """All-[EMPTY] (or zero-length) predictions are the empty span."""
    return len(region_ids) == 0 or all(i == EMPTY_ID for i in region_ids)


def empty_precision_recall(rows):
    """([EMPTY] class P/R, n_gt_empty). Recall is degenerate when the GT
    has no empty spans — reported with the count, not dropped."""
    tp = sum(r["pred_empty"] and r["gt_empty"] for r in rows)
    fp = sum(r["pred_empty"] and not r["gt_empty"] for r in rows)
    fn = sum(not r["pred_empty"] and r["gt_empty"] for r in rows)
    n_gt = sum(r["gt_empty"] for r in rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(p), float(r), int(n_gt)


# ---------------- OT-arm model wrapper ----------------

class PosModel(torch.nn.Module):
    """poc_diff's MDGQA + the Task-4 position head, as sample_ot_spans
    expects. `predict_positions` clamps the head output to [0,1] (the
    trainer regresses slot coordinates with plain MSE; eval clamps)."""

    def __init__(self, md, pos_head):
        super().__init__()
        self.md = md
        self.pos_head = pos_head

    @property
    def mask_id(self):
        return self.md.mask_id

    @property
    def empty_id(self):
        return self.md.empty_id

    def trunk(self, x, probe=False, attn_mask=None):
        return self.md.trunk(x, probe=probe, attn_mask=attn_mask)

    @property
    def embed(self):
        return self.md.embed

    def predict_positions(self, h):
        return self.pos_head(h).squeeze(-1).clamp(0.0, 1.0)


def load_ot(path, device):
    """OT twin checkpoint: md weights + pos_head state."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from experiments.training.poc_diff.model_md import MDGQA
    md = MDGQA(ck["cfg"]).to(device).eval()
    md.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    pos_head = torch.nn.Linear(ck["cfg"]["d_model"], 1).to(device).eval()
    pos_head.load_state_dict(
        {k: v.float() for k, v in ck["pos_head"].items()})
    return PosModel(md, pos_head)


# ---------------- arm runners ----------------

def cal_arm_rows(md, rows, tok, steps, window, device, batch=16):
    """Arm (b): CAL length pick + fixed-window sampling at the picked L.
    Batched: the unbatched 216-row call materialized ~29GB of concatenated
    probs (216x256x130k fp32) and spilled to system RAM via WSL sysmem
    fallback — the machine OOM the user felt."""
    out_rows = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        lens = CAL.pick_length(md, [r["prompt_ids"] for r in chunk],
                               window=window)
        for r, L in zip(chunk, lens):
            if L == 0:
                out_rows.append(dict(
                    metrics=point_metrics("", r["span_text"], tok),
                    pred_len=0, gt_len=r["span_len"], first_slot=0,
                    pred_empty=True, gt_empty=r["span_len"] == 0,
                    lat_ms=0.0, picked_L=0))
                continue
            s = sample_spans(md, [r["prompt_ids"]], [L], steps,
                             temperature=0.0)
            ids = s["pred_ids"][0, :L]
            text = decode_span(tok, ids, md.empty_id)
            out_rows.append(dict(
                metrics=point_metrics(text, r["span_text"], tok),
                pred_len=len(tok(text,
                                 add_special_tokens=False)["input_ids"]),
                gt_len=r["span_len"], first_slot=0,
                pred_empty=is_pred_empty(ids.tolist()),
                gt_empty=r["span_len"] == 0,
                lat_ms=0.0, picked_L=L))
    return out_rows


def ot_arm_rows(ot_model, rows, tok, steps, window, batch=16):
    """Arm (c): joint value+position sampling; snap decides length/anchor.
    Batched in chunks (memory); per-row results collected directly —
    per-batch tensors pad to their own max length, so never cross-cat."""
    out_rows = []
    import sample_ot as _so
    try:
        from . import sample_ot as _so
    except ImportError:
        pass
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        o = sample_ot_spans(ot_model, [r["prompt_ids"] for r in chunk],
                            window=window, steps=steps, temperature=0.0)
        for j, r in enumerate(chunk):
            n = int(o["lengths"][j])
            ids = o["pred_ids"][j, :n].tolist()
            first = int(o["slots"][j, 0].item()) if n else 0
            text = _so.decode_ot_span(ids, tok=tok)
            out_rows.append(dict(
                metrics=point_metrics(text, r["span_text"], tok),
                pred_len=len(tok(text,
                                 add_special_tokens=False)["input_ids"]),
                gt_len=r["span_len"], first_slot=first,
                pred_empty=is_pred_empty(ids), gt_empty=r["span_len"] == 0,
                lat_ms=o["latency_ms"]))
    return out_rows


def summarize(rows, window):
    p, r, n_gt = empty_precision_recall(rows)
    return dict(
        exact=float(np.mean([x["metrics"]["exact"] for x in rows])),
        edit_sim=float(np.mean([x["metrics"]["edit_sim"] for x in rows])),
        prefix8=float(np.mean([x["metrics"]["prefix8"] for x in rows])),
        position=float(np.mean([x["metrics"]["position"] for x in rows])),
        length_mae=length_mae(rows), position_mse=position_mse(rows, window),
        empty_precision=p, empty_recall=r, n_gt_empty=n_gt,
        lat_p50=float(np.percentile([x["lat_ms"] for x in rows], 50))
        if rows else float("nan"))


def bucketize(rows_gt, arm_rows):
    def bucket_of(n):
        for name, lo, hi in BUCKETS:
            if lo <= n <= hi:
                return name
        return None
    out = {}
    for name, lo, hi in BUCKETS:
        sel = [i for i, r in enumerate(rows_gt)
               if lo <= r["span_len"] <= hi]
        if sel:
            out[name] = dict(
                n=len(sel),
                exact=float(np.mean([arm_rows[i]["metrics"]["exact"]
                                     for i in sel])),
                length_mae=length_mae([arm_rows[i] for i in sel]))
    return out


def write_results_md(results, path):
    arms = [a for a in ("base", "cal", "ot") if a in results["arms"]]
    lines = [
        "# POC-DDOT results — OT position coupling vs CAL vs best arm", "",
        "Three-way eval per the DDOT plan (pre-registered baselines and "
        "kill test). Position-MSE = spurious re-anchoring, "
        "(first-snapped-slot/window)^2 — fixed-window arms are 0 by "
        "construction; LOWER is better for every arm.", "",
        f"Eval rows: {results['n_rows']}, window {results['window']}, "
        f"steps {results['steps']}.", "",
        "## Metric table", "",
        "| metric | " + " | ".join(arms) + " |",
        "|" + "---|" * (len(arms) + 1),
    ]
    keys = [("exact", "span exact-match"), ("prefix8", "prefix-match"),
            ("position", "position-match"), ("edit_sim", "edit similarity"),
            ("length_mae", "length-MAE"), ("position_mse", "position-MSE"),
            ("empty_precision", "[EMPTY] precision"),
            ("empty_recall", "[EMPTY] recall (n_gt=%d)"),
            ("lat_p50", "p50 latency ms")]
    for key, label in keys:
        row = [label] + [f"{results['arms'][a].get(key, float('nan')):.4f}"
                         for a in arms]
        lines.append("| " + " | ".join(row) + " |")
    for a in arms:
        lines += ["", f"## per-bucket — {a}", "",
                  "| bucket | n | exact | length-MAE |",
                  "|---|---|---|---|"]
        for name, b in results["arms"][a].get("buckets", {}).items():
            lines.append(f"| {name} | {b['n']} | {b['exact']:.3f} | "
                         f"{b['length_mae']:.3f} |")
    lines += [
        "", "## KILL TEST (DDOT plan, verbatim)", "",
        "If OT-coupled diffusion fails to beat BOTH baselines on "
        "length-MAE AND position-MSE AND exact-match simultaneously, the "
        "OT mechanism adds nothing here — close family E, write the "
        "negative result. No rescue arm.", "",
        "- [ ] VERDICT: ______", "- [ ] numbers: ______", "",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-ckpt", default="/mnt/h/sepalith/runs/poc_diff/md_final.pt")
    ap.add_argument("--ot-ckpt", default="/mnt/h/sepalith/runs/poc_ddot/ot_final.pt")
    ap.add_argument("--base-results", default=f"{TMP}/eval_results.json",
                    help="poc_diff's own eval json (their best arm)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="/tmp/poc_ddot/eval_ot_results.json")
    ap.add_argument("--out", default="experiments/training/poc_ddot/RESULTS.md")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(__import__("os").environ.get("POC_MEM_FRACTION", "0.6")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    rows = [json.loads(l) for l in
            open(f"{TMP}/eval_triples.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"eval rows: {len(rows)}", flush=True)

    from experiments.training.poc_diff.eval_spans import load_md
    md = load_md(args.md_ckpt, device)
    ot = load_ot(args.ot_ckpt, device)

    arms = {}
    if Path(args.base_results).exists():
        base = json.load(open(args.base_results))
        best = "ar" if base["ar"]["exact"] >= max(
            v["exact"] for v in base["md"].values()) else "md_best"
        arms["base"] = dict(base[best if best == "ar" else "md"][
            max(base["md"], key=lambda k: base["md"][k]["exact"])]
            if best == "md_best" else base["ar"],
            source=best)
    cal_rows = cal_arm_rows(md, rows, tok, args.steps, args.window, device)
    arms["cal"] = summarize(cal_rows, args.window)
    arms["cal"]["buckets"] = bucketize(rows, cal_rows)
    ot_rows = ot_arm_rows(ot, rows, tok, args.steps, args.window)
    arms["ot"] = summarize(ot_rows, args.window)
    arms["ot"]["buckets"] = bucketize(rows, ot_rows)

    results = dict(arms=arms, n_rows=len(rows), steps=args.steps,
                   window=args.window)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.json}", flush=True)
    write_results_md(results, args.out)
    return results


if __name__ == "__main__":
    main()
