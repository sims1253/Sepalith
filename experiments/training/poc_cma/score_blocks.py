"""Task 2 GPU items for the decay/CMA POC: (a) merged quick-eval blocks
from the draw's position-disjoint holdouts, (b) the per-block CE scoring
pass under the frozen scorer checkpoint, (c) curriculum order emission.

Subcommands:
  eval-blocks   round-robin merged eval_blocks.npy over strata holdouts
                (bounded rows per stratum; memmap reads only)
  score         batched per-block mean CE over the whole draw order under
                a frozen ckpt -> block_ce.npy (float32, draw-order aligned)
  curriculum    40 token-mass buckets; per stratum, drawn multiset sorted
                ascending by block CE; bucket b takes each stratum's b-th
                group (production shares preserved per bucket); within
                bucket, seeded interleave; buckets concatenated ascending
                -> curriculum_order.idx.npy (hard/noisy-last polarity)
GPU discipline: LaunchGate before any CUDA context; POC_MEM_FRACTION cap
(default 0.42); memmap streaming, one batch in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
POC_TWIN = os.path.abspath(os.path.join(HERE, "..", "poc_twin"))
sys.path.insert(0, POC_TWIN)

import train as T  # noqa: E402
from model import TinyGQA  # noqa: E402


def load_draw(draw_dir):
    man = json.load(open(os.path.join(draw_dir, "draw_manifest.json")))
    order = np.load(os.path.join(draw_dir, "uniform_order.idx.npy"))
    return man, order


def cmd_eval_blocks(draw_dir, per_stratum, out):
    import data_prep as D  # noqa: F401  (BLOCK_TOKENS)
    man, _ = load_draw(draw_dir)
    rows, src = [], []
    for si, s in enumerate(man["strata"]):
        blocks = np.load(s["path"], mmap_mode="r")
        hold = np.arange(s["holdout_start"], s["blocks"])
        pick = hold[:per_stratum]
        rows.append(np.asarray(blocks[pick]))
        src += [(si, int(b)) for b in pick]
    # round-robin interleave so any prefix is stratified
    L = max(r.shape[0] for r in rows)
    merged = []
    for i in range(L):
        for r in rows:
            if i < len(r):
                merged.append(r[i])
    merged = np.stack(merged)
    np.save(out, merged)
    with open(out + ".src.json", "w") as f:
        json.dump(dict(src=src, per_stratum=per_stratum), f)
    print(f"[eval-blocks] {merged.shape} -> {out}", flush=True)


@torch.no_grad()
def cmd_score(draw_dir, ckpt, out, bs, gate_min_free_mib):
    T.LaunchGate(min_free_mib=gate_min_free_mib).wait()
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("POC_MEM_FRACTION", "0.42")))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = TinyGQA(cfg).cuda().eval()
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    man, order = load_draw(draw_dir)
    blocks = [np.load(s["path"], mmap_mode="r") for s in man["strata"]]
    ce = np.empty(len(order), dtype=np.float32)
    done = 0
    t0 = time.time()
    while done < len(order):
        win = order[done:done + bs]
        xb = np.empty((len(win), blocks[0].shape[1]), dtype=np.int32)
        for si in np.unique(win[:, 0]):
            m = win[:, 0] == si
            xb[m] = blocks[si][win[m, 1]]
        x = torch.from_numpy(xb.astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.trunk(x[:, :-1], probe=False)
            logits = F.linear(h, model.embed.weight)
        lg = logits.view(-1, logits.size(-1)).float()
        tg = x[:, 1:].reshape(-1)
        per = F.cross_entropy(lg, tg, reduction="none")
        per = per.view(len(win), -1).mean(dim=1)
        ce[done:done + len(win)] = per.cpu().numpy()
        done += len(win)
        if (done // bs) % 500 == 0:
            r = done / len(order)
            print(f"[score] {done}/{len(order)} ({r:.1%}) "
                  f"{(time.time() - t0) / max(1, done) * len(order) / 60:.1f}min ETA",
                  flush=True)
    np.save(out, ce)
    print(f"[score] wrote {out} mean={ce.mean():.4f}", flush=True)


def cmd_curriculum(draw_dir, ce_path, buckets, out, seed):
    man, order = load_draw(draw_dir)
    ce = np.load(ce_path)
    assert len(ce) == len(order)
    groups = []  # per bucket: list of (N,2) arrays
    for b in range(buckets):
        groups.append([])
    for si in range(len(man["strata"])):
        m = order[:, 0] == si
        sub = order[m]
        sub_ce = ce[m]
        # sort the drawn multiset ascending by CE (hard/noisy-last:
        # global trend up because buckets go 0..B-1 and bucket b holds
        # each stratum's b-th ascending group)
        asc = np.argsort(sub_ce, kind="stable")
        sub, sub_ce = sub[asc], sub_ce[asc]
        parts = np.array_split(np.arange(len(sub)), buckets)
        for b, part in enumerate(parts):
            groups[b].append(sub[part])
    out_order = []
    for b, parts in enumerate(groups):
        rng = np.random.RandomState(seed + b)
        # seeded interleave: shuffle the concatenated bucket, so mixture
        # share per bucket is exact and within-bucket order is random
        cat = np.concatenate(parts)
        rng.shuffle(cat)
        out_order.append(cat)
    cur = np.concatenate(out_order).astype(np.int32)
    assert len(cur) == len(order)
    # sanity: multiset preserved
    assert np.array_equal(np.sort(cur[:, 0]), np.sort(order[:, 0]))
    np.save(out, cur)
    print(f"[curriculum] {len(cur)} blocks, {buckets} buckets -> {out}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval-blocks")
    p.add_argument("--draw", default="/tmp/poc_cma/draw_1bt_seed1273")
    p.add_argument("--per-stratum", type=int, default=128)
    p.add_argument("--out", default="/tmp/poc_cma/eval_blocks.npy")
    p = sub.add_parser("score")
    p.add_argument("--draw", default="/tmp/poc_cma/draw_1bt_seed1273")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="/tmp/poc_cma/draw_1bt_seed1273/block_ce.npy")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--gate-min-free-mib", type=int, default=16 * 1024)
    p = sub.add_parser("curriculum")
    p.add_argument("--draw", default="/tmp/poc_cma/draw_1bt_seed1273")
    p.add_argument("--ce", default="/tmp/poc_cma/draw_1bt_seed1273/block_ce.npy")
    p.add_argument("--buckets", type=int, default=40)
    p.add_argument("--seed", type=int, default=1273)
    p.add_argument("--out",
                   default="/tmp/poc_cma/draw_1bt_seed1273/curriculum_order.idx.npy")
    args = ap.parse_args(argv)
    if args.cmd == "eval-blocks":
        cmd_eval_blocks(args.draw, args.per_stratum, args.out)
    elif args.cmd == "score":
        cmd_score(args.draw, args.ckpt, args.out, args.bs,
                  args.gate_min_free_mib)
    else:
        cmd_curriculum(args.draw, args.ce, args.buckets, args.out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
