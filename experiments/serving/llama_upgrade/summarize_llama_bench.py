#!/usr/bin/env python3
"""Aggregate interleaved llama-bench JSON runs (r<round>-<model>-<build>.json).

Per model x test x build: median of per-round avg_ts, plus the paired
per-round ratio master/b10453 (same round = adjacent in time, so host-load
drift largely cancels). Also records the effective bench params per build
(n_batch, n_ubatch, flash_attn, type_k/v ...) so default changes are visible.

Usage: summarize_llama_bench.py <dir> [--json-out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^r(\d+)-(.+)-(b10453|master)\.json$")
PARAMS = ("n_batch", "n_ubatch", "flash_attn", "type_k", "type_v", "n_threads",
          "cpu_mask", "poll", "use_mmap", "no_kv_offload", "embeddings", "build_commit")


def test_name(row: dict) -> str:
    if row.get("n_prompt") and not row.get("n_gen"):
        return f"pp{row['n_prompt']}"
    if row.get("n_gen") and not row.get("n_prompt"):
        return f"tg{row['n_gen']}"
    return f"pp{row.get('n_prompt')}+tg{row.get('n_gen')}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    ts: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    params: dict[str, dict] = {}
    for f in sorted(a.dir.glob("r*-*.json")):
        m = NAME.match(f.name)
        if not m:
            continue
        rnd, model, build = int(m.group(1)), m.group(2), m.group(3)
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for row in rows:
            ts[(model, test_name(row), build)][rnd] = float(row["avg_ts"])
            params.setdefault(build, {k: row.get(k) for k in PARAMS if k in row})
    out: dict = {"params": params, "results": []}
    keys = sorted({(m, t) for m, t, _ in ts})
    print(f"{'model':<22}{'test':<8}{'b10453 t/s':>12}{'master t/s':>12}{'ratio(med paired)':>19}  per-round")
    for model, test in keys:
        old, new = ts.get((model, test, "b10453"), {}), ts.get((model, test, "master"), {})
        rounds = sorted(set(old) & set(new))
        ratios = [new[r] / old[r] for r in rounds if old[r] > 0]
        rec = {
            "model": model, "test": test,
            "b10453_ts_median": statistics.median(old.values()) if old else None,
            "master_ts_median": statistics.median(new.values()) if new else None,
            "paired_ratio_median": statistics.median(ratios) if ratios else None,
            "paired_ratios": [round(x, 3) for x in ratios],
            "b10453_ts": [round(old[r], 2) for r in sorted(old)],
            "master_ts": [round(new[r], 2) for r in sorted(new)],
        }
        out["results"].append(rec)
        print(f"{model:<22}{test:<8}{rec['b10453_ts_median'] or 0:>12.2f}{rec['master_ts_median'] or 0:>12.2f}"
              f"{rec['paired_ratio_median'] or 0:>19.3f}  {rec['paired_ratios']}")
    print(json.dumps(params, indent=1))
    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
