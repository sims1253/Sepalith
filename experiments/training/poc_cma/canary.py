"""Regurgitation canary (decay/CMA POC, arm-H readout): mean CE on
SEEN training blocks (drawn, high-epoch strata especially) vs the
position-disjoint HOLDOUT blocks under the same model. A large
train<<holdout gap = memorization/regurgitation risk. Non-inferiority
of H vs C on this gap is a pre-registered condition for MuonH adoption.

Samples N blocks from the actual draw order (uniform order = what every
arm consumed), plus the holdout set, evaluates both with the same code
path (eval_arms.eval_blocks_per_stratum), reports per-stratum and
aggregate train-vs-holdout gaps.

Usage: python canary.py --ckpt <pt> --tag H [--n 256] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "poc_twin")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--draw", default="/tmp/poc_cma/draw_1bt_seed1273")
    ap.add_argument("--holdout", default="/tmp/poc_cma/eval_blocks.npy")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    import train as T
    from model import TinyGQA
    from eval_arms import eval_blocks_per_stratum

    T.LaunchGate().wait()
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("POC_MEM_FRACTION", "0.42")))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyGQA(ck["cfg"]).cuda().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})

    man = json.load(open(os.path.join(args.draw, "draw_manifest.json")))
    order = np.load(os.path.join(args.draw, "uniform_order.idx.npy"))
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(order), size=args.n, replace=False)
    rows = order[pick]
    blocks = np.empty((args.n, 1025), dtype=np.int32)
    src = []
    for si in np.unique(rows[:, 0]):
        m = rows[:, 0] == si
        st = np.load(man["strata"][si]["path"], mmap_mode="r")
        blocks[m] = st[rows[m, 1]]
        src += [(int(si), 0)] * int(m.sum())  # slot ints, like holdout src
    tmp_train = "/tmp/poc_cma/_canary_train.npy"
    np.save(tmp_train, blocks)
    with open(tmp_train + ".src.json", "w") as f:
        json.dump({"src": src}, f)

    with torch.no_grad():
        tr = eval_blocks_per_stratum(model, tmp_train, tmp_train)
        ho = eval_blocks_per_stratum(model, args.holdout, args.holdout)

    # aggregate + per-stratum gaps (train loss - holdout loss; more
    # negative = more memorization)
    agg = dict(tag=args.tag, ckpt=args.ckpt, n=args.n,
               per_stratum={}, train_all=None, holdout_all=None)
    tot_t = sum(v["loss"] * v["tokens"] for v in tr.values())
    tot_h = sum(v["loss"] * v["tokens"] for v in ho.values())
    n_t = sum(v["tokens"] for v in tr.values())
    n_h = sum(v["tokens"] for v in ho.values())
    agg["train_all"] = tot_t / n_t
    agg["holdout_all"] = tot_h / n_h
    for si, v in tr.items():
        h = ho.get(si)
        if h:
            agg["per_stratum"][str(si)] = dict(
                train=v["loss"], holdout=h["loss"], gap=v["loss"] - h["loss"])
    os.remove(tmp_train)
    os.remove(tmp_train + ".src.json")
    js = args.json or f"/tmp/poc_cma/canary_{args.tag}.json"
    with open(js, "w") as f:
        json.dump(agg, f, indent=1)
    print(json.dumps(dict(train_all=agg["train_all"],
                          holdout_all=agg["holdout_all"],
                          gap=agg["train_all"] - agg["holdout_all"]), indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
