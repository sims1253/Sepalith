#!/usr/bin/env python3
"""D_harness — the 256-row harness-training carve (H-series plan §1.1).

Authority: the sft_v3 materialized eval split
(/mnt/h/sepalith/datasets/sft_v3/eval.jsonl, the same split eval_scenarios.py
trusts). D_harness = scenarios_v1 rows from TRAIN-side packages only:

  - the row's package is NOT an eval-split package of its family
    (belt: the row's zeta2-rendered prompt is not in the eval prompt set)
  - the training-time render edit_row() accepts the row (non-empty
    region_new, prompt+target within the v1 6000-char budget)

256 rows = 52 rename + 51 each pipe/format/doc_sync/na_rm, file order after a
seed-42 shuffle per family. Seed-locked: identical output on re-run.
holdout_rule.py N/A per the plan (synthetic constructed families, no CRAN
package dimension for the 2% rule; the package split above is the
contamination control) — but we still RECORD the 2%-rule verdict per package
in the manifest as an audit column.

Writes:
  experiments/harness_search/data/d_harness.jsonl      (rows + _prompt/_target)
  experiments/harness_search/data/d_harness_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "post-processing"))    # assemble_sft_v2
sys.path.insert(0, str(HERE.parent / "synthetic-data"))     # scenarios
sys.path.insert(0, str(HERE.parent / "data-mining"))        # holdout_rule
sys.path.insert(0, str(HERE))                               # harness_search

from assemble_sft_v2 import edit_row                        # noqa: E402
import scenarios                                            # noqa: E402
import holdout_rule                                         # noqa: E402

FAMILIES = ("rename_propagation", "pipe_rewrite", "format_propagation",
            "doc_sync", "na_rm_propagation")
SCEN_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
HOLDOUT_REF = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
PER_FAMILY = {"rename_propagation": 52, "pipe_rewrite": 51,
              "format_propagation": 51, "doc_sync": 51, "na_rm_propagation": 51}
SEED = 42
TOTAL = 256


def eval_split_sets():
    """({family: eval packages}, {family: eval prompts}) from the
    materialized split (package_or_repo is the field the assembly writes)."""
    pkgs = {f: set() for f in FAMILIES}
    prompts = {f: set() for f in FAMILIES}
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        fam = r.get("family")
        if fam in pkgs:
            pkgs[fam].add(r.get("package_or_repo"))
            prompts[fam].add(r["prompt"])
    return pkgs, prompts


def row_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:12]


def carve(seed: int = SEED):
    eval_pkgs, eval_prompts = eval_split_sets()
    rows, manifest = [], {"seed": seed, "per_family": {}, "total": 0,
                          "authority": str(HOLDOUT_REF),
                          "eval_packages": {f: sorted(eval_pkgs[f]) for f in FAMILIES}}
    for fam in FAMILIES:
        pool = []
        n_total = n_eval_pkg = n_prompt_hit = n_over_budget = 0
        for line in open(SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            n_total += 1
            if row["package"] in eval_pkgs[fam]:
                n_eval_pkg += 1
                continue
            rr = edit_row(dict(row), fam, row["package"])
            if rr is None:
                n_over_budget += 1
                continue
            if rr["prompt"] in eval_prompts[fam]:
                n_prompt_hit += 1
                continue
            row = dict(row, _prompt=rr["prompt"], _target=rr["target"])
            scenarios.validate_example(row)  # every carved row must validate
            pool.append(row)
        rng = random.Random(seed)
        order = list(range(len(pool)))
        rng.shuffle(order)  # file order after a seed-42 shuffle (plan §1.1)
        take = [pool[i] for i in order[:PER_FAMILY[fam]]]
        manifest["per_family"][fam] = dict(
            file_rows=n_total, eval_split_pkg_rows=n_eval_pkg,
            over_budget=n_over_budget, eval_prompt_hits=n_prompt_hit,
            train_pool=len(pool), carved=len(take),
            packages=sorted({r["package"] for r in take}),
            rule2_holdout_pkgs=sorted({r["package"] for r in take
                                       if holdout_rule.is_holdout(r["package"])}),
            rows=[dict(id=row_id(r["_prompt"]), package=r["package"],
                       path=r["path"], note=r.get("note", ""),
                       prompt_chars=len(r["_prompt"])) for r in take])
        rows.extend(take)
    manifest["total"] = len(rows)
    return rows, manifest


def main():
    out_dir = HERE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = carve()
    assert len(rows) == TOTAL, f"carve gave {len(rows)} rows, want {TOTAL}"
    with open(out_dir / "d_harness.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "d_harness_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({f: manifest["per_family"][f]["carved"] for f in FAMILIES}))
    print(f"D_harness: {len(rows)} rows -> {out_dir}/d_harness.jsonl")


if __name__ == "__main__":
    main()
