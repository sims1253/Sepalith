"""Two-anchor Modified-Power-Law fit over trainer loss logs (decay/CMA POC
Task 0; Puro-2B §method operationalized for our instrument).

Model:  L(t) = c + a * t^(-b)

Anchors (the "two-anchor" fit):
  - anchor A: mean loss over a window around 30% of max tokens seen
  - anchor B: mean loss over a window around 90% of max tokens seen
Asymptote c is taken as the last stable loss (a decayed run's final loss
is a near-floor proxy; for constant-LR logs c is underestimated, which
the decay-estimate caveat below accounts for). (a, b) then follow in
closed form from the two anchors.

Decay-ratio estimate (the D-test-point picker): the fitted curve says how
much loss drop remains above the asymptote as a function of tokens. We
define the decay point as where the remaining drop falls below eps (5%)
of the TOTAL observed drop (first stable loss -> last loss): i.e. the
last moment at which annealing still has >=5% of the achievable gap left
to harvest. decay_frac = 1 - t_eps/t_max. If that disagrees with the
frozen 0.2 default by >0.15, the plan says record it in RESULTS before
the D arm runs.

Usage:
  python mpl_fit.py logs/muon.jsonl [logs/probe_muon_0.01.jsonl ...]
  # each arg: a trainer JSONL with {"step","tokens","loss"} rows
RAM-discipline: streams line by line, keeps only (tokens, loss) pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def read_loss_series(path):
    """Stream a trainer JSONL -> list[(tokens, loss)] (step rows only)."""
    pts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in r and "tokens" in r and r["loss"] is not None:
                pts.append((float(r["tokens"]), float(r["loss"])))
    return pts


def _window_mean(pts, frac, width=0.08):
    """Mean (tokens, loss) over rows within [frac-width/2, frac+width/2]
    of the token range; falls back to the single nearest row."""
    if not pts:
        raise ValueError("empty series")
    t0, t1 = pts[0][0], pts[-1][0]
    span = max(1e-9, t1 - t0)
    lo = t0 + span * (frac - width / 2)
    hi = t0 + span * (frac + width / 2)
    sel = [(t, l) for t, l in pts if lo <= t <= hi]
    if not sel:
        sel = [min(pts, key=lambda p: abs(p[0] - (t0 + span * frac)))]
    ts = sum(t for t, _ in sel) / len(sel)
    ls = sum(l for _, l in sel) / len(sel)
    return ts, ls


def mpl_fit(pts):
    """Two-anchor MPL fit with asymptote solve. Anchors at 30%/60%/90%
    of the token range; (a, b, c) satisfy the three MPL equations
    L_i - c = a t_i^-b exactly. c is found by 1-D bisection on the
    consistency equation (b from anchors A,B == b from anchors B,C);
    fallback (no bracket / degenerate): c = mean of the final quarter."""
    if len(pts) < 4:
        raise ValueError("need >=4 loss points")
    tA, LA = _window_mean(pts, 0.30)
    tB, LB = _window_mean(pts, 0.60)
    tC, LC = _window_mean(pts, 0.90)

    def b_from(c):
        return (math.log((LA - c) / (LB - c)) / math.log(tB / tA),
                math.log((LB - c) / (LC - c)) / math.log(tC / tB))

    c_lo, c_hi = LC - 1.0, LC - 1e-9  # c must sit below all anchors
    b1, b2 = b_from(c_lo), b_from(c_hi)
    c = None
    if (b1[0] - b1[1]) * (b2[0] - b2[1]) < 0:
        for _ in range(60):
            c_mid = 0.5 * (c_lo + c_hi)
            bm = b_from(c_mid)
            if (b_from(c_lo)[0] - b_from(c_lo)[1]) * (bm[0] - bm[1]) <= 0:
                c_hi = c_mid
            else:
                c_lo = c_mid
        c = 0.5 * (c_lo + c_hi)
    if c is None or not (LA > c and LB > c and LC > c):
        tail = pts[-max(1, len(pts) // 4):]
        c = min(sum(l for _, l in tail) / len(tail), LC, LB, LA) - 1e-3
    b = math.log((LA - c) / (LB - c)) / math.log(tB / tA)
    a = (LA - c) * tA ** b
    return dict(a=a, b=b, c=c, anchor_A=(tA, LA), anchor_B=(tB, LB),
                anchor_C=(tC, LC), first=pts[0], last=pts[-1])


def decay_ratio_estimate(fit, eps=0.05):
    """Fraction of run tokens at which the remaining-above-asymptote drop
    still equals eps of the total observed drop. 1 - that fraction is the
    suggested decay_frac."""
    (tA, LA), (tB, LB), c, a, b = (fit["anchor_A"], fit["anchor_B"],
                                   fit["c"], fit["a"], fit["b"])
    t_last, L_last = fit["last"]
    total_drop = max(1e-12, fit["first"][1] - L_last)
    # remaining gap g(t) = a t^-b; target gap = eps * total_drop
    target = eps * total_drop
    if a <= 0 or b <= 0:
        return 0.2, dict(note="degenerate fit; keeping frozen default")
    t_eps = (a / target) ** (1.0 / b)
    frac = 1.0 - min(1.0, t_eps / t_last)
    return frac, dict(t_eps=t_eps, t_last=t_last, target_gap=target,
                      total_drop=total_drop)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="trainer JSONL loss logs")
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args(argv)
    out = {}
    for path in args.logs:
        try:
            pts = read_loss_series(path)
            fit = mpl_fit(pts)
            frac, diag = decay_ratio_estimate(fit, eps=args.eps)
            out[path] = dict(n_points=len(pts), a=round(fit["a"], 6),
                             b=round(fit["b"], 4), c=round(fit["c"], 5),
                             decay_frac=round(frac, 4), **diag)
        except Exception as e:  # noqa: BLE001 — report per-file, keep going
            out[path] = dict(error=str(e))
        print(json.dumps({path: out[path]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
