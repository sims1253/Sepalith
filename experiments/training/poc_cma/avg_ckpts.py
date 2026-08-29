"""Task 5 averaging lab (zero training): SMA-k and weighted checkpoint
averaging over an arm's checkpoint set, per the plan (KT releases
equal-weight SMA6 over its 6 tail ckpts; C/K endpoints averaged as a
READOUT ONLY — Puro predicts neutral-to-negative without a const tail).

Usage:
  python avg_ckpts.py --ckpts a.pt b.pt c.pt [--weights 1,1,2] --out avg.pt
  python avg_ckpts.py --dir /tmp/poc_twin/ckpt_cma_KT --pattern 'tail_*.pt' --out avg.pt
Weights default uniform (SMA-k). Output ckpt keeps the LAST input's
metadata (step/cfg/args); model weights = bf16 mean. Opt states dropped.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch


def load_sd(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def average(paths, weights=None):
    if len(paths) < 2:
        raise ValueError("need >=2 checkpoints")
    if weights is None:
        weights = [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("weights count mismatch")
    tot = sum(weights)
    weights = [w / tot for w in weights]
    acc = None
    last = None
    for p, w in zip(paths, weights):
        ck = load_sd(p)
        last = ck
        sd = ck["model"]
        if acc is None:
            acc = {k: v.float() * w for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += v.float() * w
        del ck, sd
    out = dict(step=last["step"], cfg=last["cfg"], args=last.get("args"),
               avg_of=[os.path.basename(p) for p in paths],
               avg_weights=[round(w, 6) for w in weights],
               model={k: v.bfloat16() for k, v in acc.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--dir", default=None)
    ap.add_argument("--pattern", default="tail_*.pt",
                    help="glob under --dir; files sorted by trailing int")
    ap.add_argument("--weights", default=None,
                    help="comma floats, same count as ckpts (default uniform)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = list(args.ckpts)
    if args.dir:
        g = sorted(glob.glob(os.path.join(args.dir, args.pattern)),
                   key=lambda p: int(re.findall(r"(\d+)",
                                                os.path.basename(p))[-1]))
        if not g:
            sys.exit(f"no files match {args.dir}/{args.pattern}")
        paths += g
    w = ([float(x) for x in args.weights.split(",")]
         if args.weights else None)
    out = average(paths, w)
    torch.save(out, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    print(f"[avg] {len(paths)} ckpts -> {args.out} "
          f"(weights {out['avg_weights']}, step {out['step']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
