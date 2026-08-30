"""Task 7 (E3) prep — ramped-mixing continue-train order files for the
three smallest manifest strata (so_r_qa, bioc, curated_py; all sub-1.5%
draw share), per the plan's proxy-strata protocol (Puro's proxy-benchmark
recipe at our scale).

For each candidate stratum, emits an order-file dir consumable by
train.OrderFileData (explicit .npy path + sibling draw_manifest.json):
  e3_<name>_ramp/   0.25BT continuation (480 steps x 512 blocks) where
                    the candidate's share ramps linearly 0 -> 0.8 over
                    10 equal windows; remaining mass = production shares
                    renormalized over the other strata.
  e3_control/       one plain production-share draw of the same length
                    (the no-bump control continuation).
All draws exclude each stratum's position-disjoint holdout tail
(data_prep convention: last min(256, blocks//2) blocks).

Run later on GPU: train.py --resume <scorer final.pt> --steps 960
--order-file <order .npy> ... (schedule spans scorer 480 + continuation
480), then eval_arms.py per run for the capability vector (held-out
R-BPB + per-stratum holdout loss = the stratum-target probe).

CPU-only, RAM-disciplined: strata .npy files are never opened here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import BLOCK_TOKENS, load_strata  # noqa: E402

CANDIDATES = ["so_r_qa", "bioc", "curated_py"]
RAMP_WINDOWS = 10
RAMP_TOP = 0.8


def _avail(s, holdout_blocks):
    return max(1, s["blocks"] - min(holdout_blocks, s["blocks"] // 2))


def _manifest_out(strata, holdout_blocks):
    return dict(
        draw_id="e3", seed=None, block_tokens=BLOCK_TOKENS,
        note=("E3 ramped-mixing continuation order (see e3_orders.py); "
              "holdouts position-disjoint per data_prep convention"),
        strata=[dict(name=s["name"], path=s["path"], blocks=s["blocks"],
                     draw_share=s["draw_share"],
                     normalized_share=round(s["normalized_share"], 6),
                     eval_slice=None,
                     holdout_start=_avail(s, holdout_blocks),
                     holdout_blocks=s["blocks"] - _avail(s, holdout_blocks))
                for s in strata])


def make_order(strata, n_blocks, seed, cand_slot=None, holdout_blocks=256):
    """Production-share draw; optionally ramp cand_slot share 0->RAMP_TOP.

    Slot choice per block: with prob share[cand] -> candidate; else
    multinomial over the other strata with renormalized shares. Block
    index: uniform with replacement within [0, avail)."""
    rng = np.random.RandomState(seed)
    names = [s["name"] for s in strata]
    shares = np.array([s["normalized_share"] for s in strata])
    order = np.empty((n_blocks, 2), dtype=np.int32)
    win = n_blocks // RAMP_WINDOWS or 1
    avail = np.array([_avail(s, holdout_blocks) for s in strata])
    for w in range(RAMP_WINDOWS):
        lo, hi = w * win, min((w + 1) * win, n_blocks) \
            if w < RAMP_WINDOWS - 1 else n_blocks
        m = hi - lo
        if cand_slot is None:
            slots = rng.choice(len(strata), size=m, p=shares)
        else:
            top = RAMP_TOP * (w + 1) / RAMP_WINDOWS
            p = shares.copy()
            p[cand_slot] = 0.0
            p = p / p.sum() * (1.0 - top)
            p[cand_slot] = top
            slots = rng.choice(len(strata), size=m, p=p)
        order[lo:hi, 0] = slots
        for si in np.unique(slots):
            mm = slots == si
            order[lo:hi, 1][mm] = rng.randint(0, avail[si], size=int(mm.sum()))
    return order, names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="/mnt/h/sepalith/a2_mixture_manifest.json")
    ap.add_argument("--out", default="/tmp/poc_cma")
    ap.add_argument("--steps", type=int, default=480,
                    help="continuation steps (x 512 blocks x 1024 tok)")
    ap.add_argument("--seed", type=int, default=12731)
    ap.add_argument("--holdout-blocks", type=int, default=256)
    args = ap.parse_args(argv)

    strata = load_strata(args.manifest)
    for s in strata:
        if not os.path.exists(s["path"]):
            sys.exit(f"FATAL: stratum file missing: {s['path']}")
    n_blocks = args.steps * (524288 // BLOCK_TOKENS)
    man_out = _manifest_out(strata, args.holdout_blocks)

    jobs = [("e3_control", None)] + [
        (f"e3_{n}_ramp", i) for i, n in enumerate(s["name"] for s in strata)
        if n in CANDIDATES]
    for k, (dirname, cand_slot) in enumerate(jobs):
        d = os.path.join(args.out, dirname)
        os.makedirs(d, exist_ok=True)
        if cand_slot is not None:
            names = [s["name"] for s in strata]
            assert names[cand_slot] in CANDIDATES
        order, _ = make_order(strata, n_blocks, args.seed + k, cand_slot,
                              args.holdout_blocks)
        np.save(os.path.join(d, "order.npy"), order)
        man = dict(man_out)
        man["draw_id"] = dirname
        man["seed"] = args.seed + k
        if cand_slot is not None:
            man["candidate_stratum"] = strata[cand_slot]["name"]
            seen = np.bincount(order[:, 0], minlength=len(strata))
            man["candidate_share_realized"] = round(
                float(seen[cand_slot]) / n_blocks, 4)
        with open(os.path.join(d, "draw_manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        print(f"[e3] {dirname}: {n_blocks} blocks "
              f"({n_blocks * BLOCK_TOKENS / 1e6:.0f}M tok)"
              + (f" cand share {man.get('candidate_share_realized')}"
                 if cand_slot is not None else " (control)"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
