#!/usr/bin/env python3
"""Select the C03 parity TRAIN prompts: 200 rows, seed 3407, stratified by family.

Reads the pre-tokenized editing-SFT TRAIN rows (DAT10-finish-source-repair-v3,
public copy on HF under ``campaign-20260915/sft/``) and writes a prompts
JSONL for ``scripts/kaggle/infer_entry.py --prompts`` and for the PC
reference job: ``id``, ``split``, ``family``, ``package_id`` and
``prompt_token_ids`` (``input_ids[:target_start]``, BOS included, exactly the
prefix the trainer's teacher-forced target follows).

Allocation: largest-remainder proportional to the family counts of all
15,006 rows (finish_block 104, pipe_rewrite 26, format_propagation 21,
rename_propagation 21, no_op 15, roxygen_drafting 11, na_rm_propagation 2).
Within a family, ``random.Random(f"{seed}:{family}")`` samples from the
row ids sorted ascending, so the draw does not depend on file order. Only
``split == "train"`` rows are read; anything else stops the script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

DEFAULT_ROWS = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl")
ROWS_SHA256 = "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e"
COUNT = 200
SEED = 3407


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate(family_sizes, count):
    """Largest-remainder proportional allocation; ties go to the larger family, then by name."""
    total = sum(family_sizes.values())
    exact = {family: size * count / total for family, size in family_sizes.items()}
    quota = {family: int(value) for family, value in exact.items()}
    order = sorted(family_sizes, key=lambda f: (-(exact[f] - quota[f]), -family_sizes[f], f))
    for family in order[:count - sum(quota.values())]:
        quota[family] += 1
    return quota


def select(rows, count=COUNT, seed=SEED):
    by_family = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            raise SystemExit(f"row {row.get('id')} is not a TRAIN row; refused")
        by_family[row["family"]].append(row)
    quota = allocate({family: len(members) for family, members in by_family.items()}, count)
    chosen = []
    for family in sorted(by_family):
        members = sorted(by_family[family], key=lambda r: r["id"])
        picks = random.Random(f"{seed}:{family}").sample(range(len(members)), quota[family])
        chosen.extend(members[index] for index in sorted(picks))
    chosen.sort(key=lambda r: r["id"])
    return chosen, quota


def prompt_record(row):
    ids = row["input_ids"][:row["target_start"]]
    if not ids or ids[0] != 0:
        raise SystemExit(f"row {row['id']}: prompt does not start with BOS 0")
    return {"id": row["id"], "split": "train", "family": row["family"], "package_id": row["package_id"],
            "prompt_token_ids": ids, "metadata": {"source": "DAT10-finish-source-repair-v3/train-token-rows.jsonl",
                                                 "target_token_count": row.get("target_token_count"),
                                                 "renderer_id": row.get("renderer_id")}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=COUNT)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    if sha256_file(a.rows) != ROWS_SHA256:
        raise SystemExit("TRAIN rows sha256 differs from DAT10-finish-source-repair-v3")
    with a.rows.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    chosen, quota = select(rows, a.count, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(prompt_record(row), sort_keys=True) + "\n" for row in chosen))
    lengths = sorted(row["target_start"] for row in chosen)
    receipt = {"output": str(a.output), "sha256": sha256_file(a.output), "rows": len(chosen), "quota": quota,
               "seed": a.seed, "source_sha256": ROWS_SHA256,
               "prompt_tokens": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]}}
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
