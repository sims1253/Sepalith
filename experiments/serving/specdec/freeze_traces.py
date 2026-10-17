#!/usr/bin/env python3
"""Freeze the specdec benchmark trace set from the DAT10 TRAIN token rows.

Family-balanced, length-stratified (prompt band + even target-length spacing)
deterministic sample. Only rows from
``train-token-rows.jsonl`` (TRAIN split) are ever read — never DEV/final.

Output: ``traces.jsonl`` (one row per trace: ids + metadata) and
``manifest.json`` (row ids, seed, per-row sha256, source file sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

SOURCE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl"
)
DEFAULT_OUT = Path(__file__).resolve().parent
SEED = 20261018
PER_STRATUM = 18          # per (family, band)
LONG_BAND_MIN = 1000      # prompt tokens; brief bands: short <=1k, long >=2k
FAMILIES = (
    "finish_block",
    "format_propagation",
    "na_rm_propagation",
    "no_op",
    "pipe_rewrite",
    "rename_propagation",
    "roxygen_drafting",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_sha(row: dict) -> str:
    payload = json.dumps(
        [row["id"], row["input_ids"], row["target_start"]], separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    a = ap.parse_args()

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    n_total = 0
    with open(a.source) as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            assert r["split"] == "train", r["id"]
            assert r["renderer_id"] == "zeta2-prm03-v1", r["id"]
            assert r["id"] not in seen_ids, f"duplicate id {r['id']}"
            seen_ids.add(r["id"])
            band = "short" if r["prompt_token_count"] < LONG_BAND_MIN else "long"
            strata[(r["family"], band)].append(r)

    rng = random.Random(SEED)
    picked: list[dict] = []
    for key in sorted(strata):
        rows = strata[key]
        rng.shuffle(rows)  # break package clustering, deterministic
        rows.sort(key=lambda r: r["target_token_count"])  # even target spacing
        n = min(a.per_stratum, len(rows))
        if n < len(rows):
            step = len(rows) / n
            rows = [rows[int(i * step)] for i in range(n)]
        picked.extend(rows)

    # stable global order: family, band, target len
    picked.sort(key=lambda r: (r["family"], r["prompt_token_count"]))

    out_traces = a.out_dir / "traces.jsonl"
    with open(out_traces, "w") as f:
        for r in picked:
            f.write(
                json.dumps(
                    {
                        "trace_id": r["id"],
                        "family": r["family"],
                        "package_id": r["package_id"],
                        "band": "short" if r["prompt_token_count"] < LONG_BAND_MIN else "long",
                        "is_noop": r["target_operation"] == "no_op",
                        "prompt_ids": r["input_ids"][: r["target_start"]],
                        "prompt_token_count": r["prompt_token_count"],
                        "target_token_count": r["target_token_count"],
                        "target_ids": r["input_ids"][r["target_start"] :],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    manifest = {
        "schema": "sepalith.specdec.traces.v1",
        "seed": SEED,
        "per_stratum": a.per_stratum,
        "long_band_min": LONG_BAND_MIN,
        "source": str(a.source),
        "source_sha256": sha256_file(a.source),
        "n_source_rows": n_total,
        "n_traces": len(picked),
        "row_sha256": {r["id"]: row_sha(r) for r in picked},
        "families": sorted({r["family"] for r in picked}),
        "prompt_tokens": {
            "min": min(r["prompt_token_count"] for r in picked),
            "median": statistics.median(r["prompt_token_count"] for r in picked),
            "max": max(r["prompt_token_count"] for r in picked),
        },
        "target_tokens": {
            "min": min(r["target_token_count"] for r in picked),
            "median": statistics.median(r["target_token_count"] for r in picked),
            "max": max(r["target_token_count"] for r in picked),
        },
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    per = defaultdict(int)
    for r in picked:
        per[
            f"{r['family']}/"
            f"{'short' if r['prompt_token_count'] < LONG_BAND_MIN else 'long'}"
        ] += 1
    print(json.dumps(per, indent=1))
    print(f"wrote {len(picked)} traces -> {out_traces}")


if __name__ == "__main__":
    main()
