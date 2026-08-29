"""POC manifest + order files for the decay/CMA + MuonH POC (Task 1,
plan docs/research/2026-08-28-decay-cma-muonh-poc-plan.md).

Draws a 1BT block multiset from the production strata with PRODUCTION
shares (verbatim from /mnt/h/sepalith/a2_mixture_manifest.json, shares
normalized pro-rata exactly like MixtureData — the manifest's adaptive
rule; sum is 0.992 because the edit-diff stratum is deferred). Seed 1273.
The SAME multiset (and uniform order) is reused by every arm; only
order/schedule/optimizer differ.

Outputs (default /tmp/poc_cma/draw_1bt_seed1273/):
  draw_manifest.json      strata in slot order (name, path, blocks,
                          shares), holdout spec, R eval-slice references
  uniform_order.idx.npy   (N, 2) int32 [stratum_slot, block_idx] in draw
                          order — consumed by train.OrderFileData
  eval_sets/<name>.npy    held-out block indices per stratum
  blocks_meta.csv.gz      per drawn block: stratum_slot, block_idx,
                          stratum, package, tokens
  meta.json               summary + counts for the RESULTS record

RAM discipline (earlyoom scars): strata .npy files are NEVER opened here
(only manifest metadata is read); the order array is the only sizeable
allocation (N*2*4 B ~ 7.8 MB at 1BT); the metadata CSV is streamed.

Package-disjointness DEVIATION (recorded in the plan's RESULTS): the
packed strata carry no per-block package metadata, so eval holdouts are
POSITION-disjoint (a contiguous tail excluded from the draw range) and
the R strata additionally reference the existing package-disjoint eval
slices (eval_causal.npy / eval_rc.npy / so_r_qa_eval.npy). The package
column is emitted as empty.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

DEFAULT_MANIFEST = "/mnt/h/sepalith/a2_mixture_manifest.json"
BLOCK_TOKENS = 1024  # blocks are 1025-wide; 1024 target tokens each

# existing package-disjoint eval slices (r_repack_full.py convention)
R_EVAL_SLICES = {
    "r_causal": "/mnt/h/sepalith/a2/r/eval_causal.npy",
    "r_fim_mix": "/mnt/h/sepalith/a2/r/eval_rc.npy",
    "r_noop": "/mnt/h/sepalith/a2/r/eval_rc.npy",
    "so_r_qa": "/mnt/h/sepalith/a2/r/so_r_qa_eval.npy",
    "bioc": None,  # no dedicated bioc eval slice packed
}


def load_strata(manifest_path):
    """Strata in sorted-name slot order (matches MixtureData)."""
    man = json.load(open(manifest_path))
    strata = []
    for name, s in sorted(man["strata"].items()):
        if not s.get("blocks"):
            continue  # adaptive rule: not packed -> share re-cut pro-rata
        strata.append(dict(name=name, path=s["path"],
                           blocks=int(s["blocks"]),
                           draw_share=float(s["draw_share"]),
                           eval_slice=R_EVAL_SLICES.get(name)))
    tot = sum(s["draw_share"] for s in strata)
    for s in strata:
        s["normalized_share"] = s["draw_share"] / tot
    return strata


def make_draw(strata, n_blocks, seed, holdout_blocks=256):
    """Seed-fixed 1BT draw. Returns (order (N,2) int32, per-stratum dicts
    with avail/holdout info). Slot stratum ~ multinomial(normalized
    shares); block index ~ uniform with replacement within [0, n-holdout).
    """
    rng = np.random.RandomState(seed)
    p = np.array([s["normalized_share"] for s in strata])
    slots = rng.choice(len(strata), size=n_blocks, p=p)
    order = np.empty((n_blocks, 2), dtype=np.int32)
    for si, s in enumerate(strata):
        avail = max(1, s["blocks"] - min(holdout_blocks, s["blocks"] // 2))
        s["avail_blocks"] = avail
        s["holdout_start"] = avail
        s["holdout_blocks"] = s["blocks"] - avail
        m = slots == si
        order[m, 0] = si
        order[m, 1] = rng.randint(0, avail, size=int(m.sum()))
    return order, strata


def write_outputs(out_dir, order, strata, manifest_path, seed,
                  n_blocks, holdout_blocks):
    os.makedirs(os.path.join(out_dir, "eval_sets"), exist_ok=True)
    seen = np.bincount(order[:, 0], minlength=len(strata))
    man_out = dict(
        draw_id=os.path.basename(out_dir.rstrip("/")),
        seed=seed, block_tokens=BLOCK_TOKENS,
        n_blocks=int(n_blocks),
        tokens=int(n_blocks * BLOCK_TOKENS),
        holdout_blocks=holdout_blocks,
        source_manifest=manifest_path,
        share_sum_raw=round(sum(s["draw_share"] for s in strata), 6),
        note=("shares normalized pro-rata (adaptive rule; edit-diff "
              "deferred). Eval holdouts POSITION-disjoint (tail); R "
              "strata reference package-disjoint packed slices. "
              "package column empty — no per-block package metadata "
              "exists (deviation, see plan RESULTS)."),
        strata=[dict(name=s["name"], path=s["path"], blocks=s["blocks"],
                     draw_share=s["draw_share"],
                     normalized_share=round(s["normalized_share"], 6),
                     draw_slots=int(seen[i]),
                     eval_slice=s["eval_slice"],
                     holdout_start=s["holdout_start"],
                     holdout_blocks=s["holdout_blocks"])
                for i, s in enumerate(strata)],
    )
    with open(os.path.join(out_dir, "draw_manifest.json"), "w") as f:
        json.dump(man_out, f, indent=1)
    np.save(os.path.join(out_dir, "uniform_order.idx.npy"), order)
    for s in strata:
        np.save(os.path.join(out_dir, "eval_sets", f"{s['name']}.npy"),
                np.arange(s["holdout_start"], s["blocks"], dtype=np.int32))
    # stream per-block metadata csv.gz
    names = [s["name"] for s in strata]
    with gzip.open(os.path.join(out_dir, "blocks_meta.csv.gz"), "wt",
                   newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum_slot", "block_idx", "stratum", "package",
                    "tokens"])
        for si, bi in order:
            w.writerow([int(si), int(bi), names[si], "", BLOCK_TOKENS])
    meta = dict(draw_id=man_out["draw_id"], seed=seed,
                tokens=man_out["tokens"], n_blocks=int(n_blocks),
                per_stratum={s["name"]: dict(
                    slots=int(seen[i]),
                    share=round(s["normalized_share"], 6),
                    distinct_blocks=int(len(np.unique(order[order[:, 0] == i, 1]))),
                    epochs=float(seen[i] / max(1, s["avail_blocks"])),
                    holdout_blocks=s["holdout_blocks"])
                    for i, s in enumerate(strata)})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default="/tmp/poc_cma/draw_1bt_seed1273")
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--seed", type=int, default=1273)
    ap.add_argument("--holdout-blocks", type=int, default=256)
    args = ap.parse_args(argv)

    strata = load_strata(args.manifest)
    for s in strata:  # fail fast if strata drifted off the NAS
        if not os.path.exists(s["path"]):
            sys.exit(f"FATAL: stratum file missing: {s['path']}")
    n_blocks = int(args.tokens // BLOCK_TOKENS)
    print(f"[draw] {len(strata)} strata, {n_blocks} blocks "
          f"({n_blocks * BLOCK_TOKENS / 1e9:.3f} BT), seed {args.seed}",
          flush=True)
    order, strata = make_draw(strata, n_blocks, args.seed,
                              holdout_blocks=args.holdout_blocks)
    meta = write_outputs(args.out, order, strata, args.manifest,
                         args.seed, n_blocks, args.holdout_blocks)
    print(json.dumps(meta, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
