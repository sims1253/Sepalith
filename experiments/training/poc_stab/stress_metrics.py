"""
Offline stability scorer for training logs.

Metrics from the Qwen3.8-Flash-Next tech report §3.3 (stress-test criterion)
and Wortsman et al., "Small-scale proxies for large-scale Transformer
training instabilities" (arXiv:2309.14322):

  - spike count/rate: steps where loss > local rolling median + 0.1
    (Qwen: "no run exhibits a single step where the loss exceeds its local
    median by more than 0.1");
  - p99.9 / max pre-clip grad norm as a fraction of the clip threshold.

Consumes the ladder/twin log format: jsonl rows keyed by `step`; train rows
carry `loss` (and optionally pre-clip `grad_norm`), while eval/yield rows
(event:eval, yields) are skipped for stability scoring and counted in a
note. `loss`/`grad_norm` are all-or-nothing among train rows — a partial
column errors loudly rather than scoring silently wrong.

compare(A, B): "B at least as stable as A" iff spike count <= A's AND
p99.9 grad-frac <= A's (both axes must hold; margins reported either way).

CPU-only; no GPU claim needed. Line plan:
docs/research/2026-08-29-papers-recon-poc-plan.md (zcode-stabtok, P0).

Usage:
  python stress_metrics.py runA.jsonl [runB.jsonl] [--clip 1.0] \
      [--window 101] [--skip-first N]
With one path: prints the run's score. With two: run_a is the REPLACED
recipe, run_b the CANDIDATE; emits the verdict JSON "run_b at least as
stable as run_a" and exits 0 iff that holds.
"""
import argparse
import json
import sys

import numpy as np

REQUIRED_KEYS = ("step",)
DEFAULT_CLIP = 1.0  # pinned recipe grad-clip threshold


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise ValueError(
                    f"{path}:{i}: missing required key(s) {missing}")
            if "loss" in row and (row["loss"] is None
                                  or not np.isfinite(row["loss"])):
                raise ValueError(
                    f"{path}:{i}: non-finite loss at step {row['step']} — "
                    "clean the log or pass --skip-first past the blowup")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    train = [r for r in rows if "loss" in r]
    if not train:
        raise ValueError(f"{path}: no train rows (no row carries 'loss')")
    return rows


def rolling_median(xs, window):
    """Centered rolling median, edge-truncated. Falls back to the global
    median when window >= len(xs) (flagged by the caller via window_used)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = xs.size
    if window >= n:
        return np.full(n, np.median(xs)), n
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(xs[lo:hi])
    return out, window


def spike_stats(steps, losses, window=101, threshold=0.1, skip_first=0):
    med, window_used = rolling_median(losses, window)
    steps = np.asarray(steps)
    losses = np.asarray(losses, dtype=np.float64)
    keep = steps >= steps[0] + skip_first
    over = losses > med + threshold
    spikes = keep & over
    idx = np.nonzero(spikes)[0]
    return {
        "n_steps": int(keep.sum()),
        "window_used": int(window_used),
        "threshold": threshold,
        "count": int(spikes.sum()),
        "rate": float(spikes.sum() / max(keep.sum(), 1)),
        "first_spike_step": int(steps[idx[0]]) if idx.size else None,
        "last_spike_step": int(steps[idx[-1]]) if idx.size else None,
    }


def grad_stats(grad_norms, clip=DEFAULT_CLIP):
    if clip <= 0:
        raise ValueError(f"clip threshold must be > 0, got {clip}")
    g = np.asarray(grad_norms, dtype=np.float64)
    if g.size == 0:
        raise ValueError("no grad_norm rows")
    if not np.all(np.isfinite(g)):
        raise ValueError("non-finite grad_norm present — clean the log")
    p999 = float(np.percentile(g, 99.9))
    mx = float(g.max())
    return {"clip": clip, "p999": p999, "max": mx,
            "p999_frac": p999 / clip, "max_frac": mx / clip, "n": int(g.size)}


def score_run(path, clip=DEFAULT_CLIP, window=101, skip_first=0):
    rows = load_jsonl(path)
    train = [r for r in rows if "loss" in r]
    if skip_first:
        base = train[0]["step"]
        kept = [r for r in train if r["step"] >= base + skip_first]
        if not kept:
            raise ValueError(f"{path}: --skip-first {skip_first} excludes "
                             "every train row")
        train = kept
    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    out = {"path": path, "n_rows": len(rows), "n_train_rows": len(train),
           "n_skipped_rows": len(rows) - len(train),
           "spike": spike_stats(steps, losses, window=window,
                                skip_first=skip_first)}
    g = [r.get("grad_norm") for r in train if r.get("grad_norm") is not None]
    if len(g) == len(train):
        out["grad"] = grad_stats(g, clip=clip)
    elif g:
        raise ValueError(
            f"{path}: grad_norm present on {len(g)}/{len(rows)} rows — "
            "all-or-nothing; fix the logger or drop the column")
    else:
        out["grad"] = None
        out["grad_note"] = "no grad_norm column — grad axis skipped"
    return out


def compare(score_a, score_b):
    """Verdict: is B (candidate) at least as stable as A (replaced)?"""
    sa, sb = score_a["spike"], score_b["spike"]
    ga, gb = score_a.get("grad"), score_b.get("grad")
    spike_ok = sb["count"] <= sa["count"]
    if ga is not None and gb is not None:
        grad_ok = gb["p999_frac"] <= ga["p999_frac"]
        grad_margin = ga["p999_frac"] - gb["p999_frac"]
    else:
        grad_ok, grad_margin = None, None
    as_stable = spike_ok and (grad_ok is not False)
    return {
        "a": score_a["path"], "b": score_b["path"],
        "spike_count_a": sa["count"], "spike_count_b": sb["count"],
        "spike_margin": sa["count"] - sb["count"],
        "p999_frac_a": None if ga is None else ga["p999_frac"],
        "p999_frac_b": None if gb is None else gb["p999_frac"],
        "p999_frac_margin": grad_margin,
        "grad_axis": "present" if grad_ok is not None else "skipped",
        "b_at_least_as_stable_as_a": bool(as_stable),
        "verdict": ("B AT LEAST AS STABLE AS A"
                    if as_stable else "B LESS STABLE THAN A (veto)"),
    }


def _fmt(s):
    g = s["grad"]
    lines = [f"  path        {s['path']}  ({s['n_rows']} rows, "
             f"{s['n_train_rows']} train, {s['n_skipped_rows']} skipped)",
             f"  spikes      {s['spike']['count']} / {s['spike']['n_steps']}"
             f"  (rate {s['spike']['rate']:.5f}, window {s['spike']['window_used']})",
             f"  first/last  {s['spike']['first_spike_step']}"
             f"/{s['spike']['last_spike_step']}"]
    if g:
        lines.append(f"  grad p99.9  {g['p999']:.4f} = {g['p999_frac']:.2f}x"
                     f" clip({g['clip']})   max {g['max_frac']:.2f}x")
    else:
        lines.append(f"  grad        skipped ({s.get('grad_note', '')})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_a", help="jsonl log — the REPLACED recipe")
    ap.add_argument("run_b", nargs="?", help="jsonl log — the CANDIDATE recipe")
    ap.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    ap.add_argument("--window", type=int, default=101)
    ap.add_argument("--skip-first", type=int, default=0,
                    help="exclude the first N steps from spike counting")
    args = ap.parse_args(argv)

    a = score_run(args.run_a, clip=args.clip, window=args.window,
                  skip_first=args.skip_first)
    print(_fmt(a))
    if args.run_b is None:
        return 0
    b = score_run(args.run_b, clip=args.clip, window=args.window,
                  skip_first=args.skip_first)
    print(_fmt(b))
    verdict = compare(a, b)  # candidate (run_b) judged against replaced (run_a)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["b_at_least_as_stable_as_a"] else 1


if __name__ == "__main__":
    sys.exit(main())
