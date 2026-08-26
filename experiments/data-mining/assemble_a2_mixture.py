#!/usr/bin/env python3
"""A2 mixture manifest builder (the §3.2 table as an executable artifact).

Inputs: the packed strata block files under <data-root>/a2_transfers/
(transfer strata) and <data-root>/a2/r/*.npy (R strata, from
r_repack_full.py) plus the twin-scale R-side sources. Output:
a2_mixture_manifest.json — per-stratum block-file paths, token counts,
draw weights, and epochs-equivalent checks against the A2 table — ready
for train_a2 --mixture (the trainer samples the stratum per block-slot,
seeded: repetition-by-sampling, no giant merged file, cluster-portable).

--data-root defaults to /data (the rented-instance layout) when present,
else /mnt/h/sepalith (the NAS staging) — both worlds, zero flags.

Deferred strata (documented, adaptive-rule territory):
  edit-diff (0.5B, 2%): needs CRAN-Archive acquisition (the git mirror
    is EVAL-PROTECTED and must NOT enter training)
  SO r-tag (0.3B): stack_staging R shards — pack when the R-repack runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEQ = 1025

# stratum -> (block file RELATIVE to data-root, draw_share per §3.2)
STRATA = {
    # R side (shares of the 25B draw; the R strata tile the SAME corpus)
    "r_causal":   ("a2/r/r_causal.npy",   0.44),
    "r_fim_mix":  ("a2/r/r_fim_mix.npy",  0.24),
    "r_noop":     ("a2/r/r_noop.npy",     0.016),
    # R-adjacent real-world (packed by pack_r_strata.py)
    "so_r_qa":    ("a2/r/so_r_qa.npy",  0.012),
    "bioc":       ("a2/r/bioc.npy",     0.012),
    # transfers (single-epoch)
    "english":    ("a2_transfers/english/blocks.npy",    0.074),
    "python":     ("a2_transfers/python_v2/blocks.npy",  0.112),
    "c_cpp":      ("a2_transfers/c_cpp_v2/blocks.npy",   0.024),
    "js_ts":      ("a2_transfers/js_ts_v2/blocks.npy",   0.020),
    "sql":        ("a2_transfers/sql_v2/blocks.npy",     0.016),
    "julia":      ("a2_transfers/julia_v2/blocks.npy",   0.012),
    "matlab":     ("a2_transfers/matlab_v2/blocks.npy",  0.010),
    "curated_py": ("a2_transfers/python/blocks.npy",     0.004),
}
R_STRATA = ("r_causal", "r_fim_mix", "r_noop")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total-draw", type=float, default=25.0e9)
    ap.add_argument("--data-root", default=None,
                    help="default: /data if present, else /mnt/h/sepalith")
    ap.add_argument("--out", default=None,
                    help="default: <data-root>/a2_mixture_manifest.json")
    args = ap.parse_args()
    if args.data_root is None:
        args.data_root = ("/data" if Path("/data").exists()
                          else "/mnt/h/sepalith")
    root = Path(args.data_root)
    out = Path(args.out) if args.out else root / "a2_mixture_manifest.json"
    manifest = {"total_draw_tokens": args.total_draw,
                "seq": SEQ, "data_root": str(root), "strata": {},
                "notes": []}
    tot_avail = 0.0
    for name, (rel, share) in STRATA.items():
        p = root / rel
        entry = dict(path=str(p), draw_share=share,
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
            manifest["notes"].append(f"{name}: not yet packed (pending fetch/repack)")
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
        f"R strata packed by r_repack_full.py (32K tokenizer, full depth, "
        f"contamination gate: contamination.json in a2/r/)",
        f"so_r_qa = the stack v3 R keep set (answer-CODE, ODC-By-1.0; the "
        f"English×R Q&A-prose bridge slice remains unmaterialized); "
        f"bioc = Bioconductor current R/tests only (man/ excluded per the "
        f"roxygen double-count rule; vignettes/src deferred)",
        f"so_r_qa + bioc shares are PROVISIONAL (0.012 each) pending the "
        f"GO-time §3.2 re-cut alongside the full-CRAN causal and git/ "
        f"GitHub decisions (see the R-inventory map, 2026-08-26)",
        f"adaptive rule: strata still missing at build time re-cut their "
        f"share pro-rata across packed strata (train_a2 MixtureData "
        f"applies it; epochs cap 4.6 governs)",
    ]
    out.write_text(json.dumps(manifest, indent=1))
    avail_b = tot_avail / 1e9
    print(json.dumps(dict(
        data_root=str(root),
        strata=len(manifest["strata"]), packed=sum(
            1 for s in manifest["strata"].values() if s["blocks"]),
        r_share=manifest["r_share"], missing_share=manifest["missing_share"],
        tokens_available_now=f"{avail_b:.2f}B",
        out=str(out)), indent=1))


if __name__ == "__main__":
    main()
