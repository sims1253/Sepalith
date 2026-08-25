#!/usr/bin/env python3
"""A2 mixture manifest builder (the §3.2 table as an executable artifact).

Inputs: the packed strata block files under /mnt/h/sepalith/a2_transfers/
plus the R-side sources. Output: a2_mixture_manifest.json — per-stratum
block-file paths, token counts, draw weights, and epochs-equivalent
checks against the A2 table — ready for train_a2 --mixture (the trainer
samples the stratum per block-slot, seeded: repetition-by-sampling, no
giant merged file, cluster-portable).

Deferred strata (documented, adaptive-rule territory):
  edit-diff (0.5B, 2%): needs CRAN-Archive acquisition (the git mirror
    is EVAL-PROTECTED and must NOT enter training)
  SO r-tag (0.3B): stack_staging R shards — pack when the R-repack runs
R-side full-scale re-pack (astfim corpus at 25B-draw depth) is a build
step on the target machine; the manifest references the twin-scale
blocks for structure validation and marks R_TOKENS_TARGET.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("/mnt/h/sepalith/a2_mixture_manifest.json")

# stratum -> (block file, draw_share per the §3.2 table)
STRATA = {
    # R side (shares of the 25B draw; the R strata tile the SAME corpus)
    "r_causal":   ("/tmp/poc_twin/ladder/train_blocks_causal.npy", 0.44),
    "r_fim_mix":  ("/tmp/poc_twin/a2/train_blocks_fim_mixed.npy",  0.24),
    "r_noop":     ("/mnt/h/sepalith/a2_transfers/r_noop_blocks.npy", 0.016),
    # transfers (single-epoch)
    "english":    ("/mnt/h/sepalith/a2_transfers/english/blocks.npy", 0.074),
    "python":     ("/mnt/h/sepalith/a2_transfers/python_v2/blocks.npy", 0.112),
    "c_cpp":      ("/mnt/h/sepalith/a2_transfers/c_cpp_v2/blocks.npy", 0.024),
    "js_ts":      ("/mnt/h/sepalith/a2_transfers/js_ts_v2/blocks.npy", 0.020),
    "sql":        ("/mnt/h/sepalith/a2_transfers/sql_v2/blocks.npy", 0.016),
    "julia":      ("/mnt/h/sepalith/a2_transfers/julia_v2/blocks.npy", 0.012),
    "matlab":     ("/mnt/h/sepalith/a2_transfers/matlab_v2/blocks.npy", 0.010),
    "so_r_qa":    ("/mnt/h/sepalith/a2_transfers/so_r_qa/blocks.npy", 0.012),
    "curated_py": ("/mnt/h/sepalith/a2_transfers/python/blocks.npy", 0.004),
}
R_STRATA = ("r_causal", "r_fim_mix", "r_noop")
SEQ = 1025


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-draw", type=float, default=25.0e9)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    manifest = {"total_draw_tokens": args.total_draw,
                "seq": SEQ, "strata": {}, "notes": []}
    tot_avail = 0.0
    for name, (path, share) in STRATA.items():
        p = Path(path)
        entry = dict(path=path, draw_share=share,
                     draw_tokens=round(share * args.total_draw, 1))
        if p.exists():
            n_blocks = np.load(p, mmap_mode="r").shape[0]
            entry["blocks"] = int(n_blocks)
            entry["avail_tokens"] = int(n_blocks * 1024)
            entry["epochs_equiv"] = round(
                entry["draw_tokens"] / max(1, entry["avail_tokens"]), 3)
            tot_avail += entry["avail_tokens"]
        else:
            entry["blocks"] = None
            entry["avail_tokens"] = 0
            entry["epochs_equiv"] = None
            manifest["notes"].append(f"{name}: not yet packed (pending fetch)")
        manifest["strata"][name] = entry

    r_share = sum(s["draw_share"] for n, s in manifest["strata"].items()
                  if n in R_STRATA)
    missing_share = sum(s["draw_share"] for s in manifest["strata"].values()
                        if s["blocks"] is None)
    manifest["r_share"] = round(r_share, 4)
    manifest["missing_share"] = round(missing_share, 4)
    manifest["notes"] += [
        f"edit-diff stratum (2%) deferred: needs CRAN-Archive; the git "
        f"mirror is EVAL-PROTECTED (never train on it)",
        f"R strata at twin-scale blocks here are structural placeholders; "
        f"the real R re-pack happens at target-machine build time",
        f"adaptive rule: strata still missing at build time re-cut their "
        f"share to r_causal (epochs cap 4.6 governs)",
    ]
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    avail_b = tot_avail / 1e9
    print(json.dumps(dict(
        strata=len(manifest["strata"]), packed=sum(
            1 for s in manifest["strata"].values() if s["blocks"]),
        r_share=manifest["r_share"], missing_share=manifest["missing_share"],
        tokens_available_now=f"{avail_b:.2f}B",
        out=args.out), indent=1))


if __name__ == "__main__":
    main()
