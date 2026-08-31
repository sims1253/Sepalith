#!/usr/bin/env python3
"""E3-v2 orders — so_r_qa DOSE-RESPONSE at production-style constant shares.

E3's ramp was an extreme-dose probe (so_r_qa block-share reached ~80% at
peak: -23.7% target probe for +2.0% R cost). The RESULTS.md v2
recommendation: constant-share continuations at 2x/4x the production
draw share (0.024 / 0.048), where the R cost should scale down with
dose — mapping the dose-response curve for the §3.2 re-cut.

PAIRED BASE (important): e3_control (the paired control, already trained
+ evaled from the same scorer ckpt/schedule) used the PRE-E3-drop
production shares incl. bioc 0.012 + curated_py 0.004. The v2 arms
therefore keep that same base map and override ONLY so_r_qa — one
variable changes vs control, exactly as the ramp arms did. (The live
manifest has since zeroed bioc/curated_py; that's a separate decision,
not part of this pair.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import BLOCK_TOKENS, load_strata  # noqa: E402
from e3_orders import _avail  # noqa: E402  same holdout convention

# e3-era production shares (the base e3_control was trained on)
E3_SHARES = {
    "r_causal": 0.44, "r_fim_mix": 0.24, "r_noop": 0.016,
    "so_r_qa": 0.012, "bioc": 0.012,
    "english": 0.074, "python": 0.112, "c_cpp": 0.024, "js_ts": 0.020,
    "sql": 0.016, "julia": 0.012, "matlab": 0.010, "curated_py": 0.004,
}
CAND = "so_r_qa"
ARMS = {"e3v2_sorqa_2x": 2 * E3_SHARES[CAND],
        "e3v2_sorqa_4x": 4 * E3_SHARES[CAND]}


def make_order(strata, const_share, n_blocks, seed, holdout=256):
    """Constant candidate share; others renormalized over the rest."""
    rng = np.random.RandomState(seed)
    names = [s["name"] for s in strata]
    cand = names.index(CAND)
    base = np.array([E3_SHARES.get(n, 0.0) for n in names])
    p = base.copy()
    p[cand] = 0.0
    p = p / p.sum() * (1.0 - const_share)
    p[cand] = const_share
    avail = np.array([_avail(s, holdout) for s in strata])
    slots = rng.choice(len(strata), size=n_blocks, p=p)
    order = np.empty((n_blocks, 2), dtype=np.int32)
    order[:, 0] = slots
    for si in np.unique(slots):
        mm = slots == si
        order[mm, 1] = rng.randint(0, avail[si], size=int(mm.sum()))
    return order, names, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default="/mnt/h/sepalith/a2_mixture_manifest.json")
    ap.add_argument("--out", default="/tmp/poc_cma")
    ap.add_argument("--steps", type=int, default=480)
    args = ap.parse_args(argv)
    strata = load_strata(args.manifest)
    n_blocks = args.steps * (524288 // BLOCK_TOKENS)
    for name, share in ARMS.items():
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        order, names, p = make_order(strata, share, n_blocks,
                                     seed=1273 + int(round(share * 1e4)))
        np.save(d / "order.npy", order)
        (d / "draw_manifest.json").write_text(json.dumps(dict(
            draw_id=name, seed=None, block_tokens=BLOCK_TOKENS,
            note=(f"E3-v2 CONSTANT-share continuation order: {CAND} "
                  f"pinned at {share:.4f} (e3-era base shares for the "
                  f"rest — paired vs e3_control)"),
            strata=[dict(name=n, path=s["path"], blocks=s["blocks"],
                         draw_share=float(E3_SHARES.get(n, 0.0)),
                         normalized_share=round(float(pi), 6),
                         holdout_start=_avail(s, 256),
                         holdout_blocks=s["blocks"] - _avail(s, 256))
                    for n, s, pi in zip(names, strata, p)]), indent=1))
        print(f"[{name}] {n_blocks} blocks, {CAND}={share:.4f} "
              f"(empirical {float((order[:, 0] == names.index(CAND)).mean()):.4f})",
              flush=True)


if __name__ == "__main__":
    main()
