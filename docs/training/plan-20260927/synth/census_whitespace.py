"""Supplementary census analysis (sources the v1 README claim, now replaced by
C12's committed figure): finish_block near misses in the arm A census whose
decoded completion equals the gold target after deleting all whitespace, split
by kind of difference. This criterion is looser than C12's
``near_miss_with_canonical_region_equal`` (309 samples, 156 prompts), which
requires the air-canonical region to equal gold.

    PYTHONPATH=packages/sepalith/src .venv-sft/bin/python docs/training/plan-20260927/synth/census_whitespace.py \\
        --out docs/training/plan-20260927/receipts/SYNTH-census-whitespace.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

CENSUS = Path.home() / ".local/state/sepalith/c12/census-arm-a"
TRAIN = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586/tokenizer.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    gold = {}
    with TRAIN.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] == "finish_block":
                gold[row["id"]] = row["target_body_text"]
    kinds: Counter = Counter()
    prompts = set()
    with (CENSUS / "samples.jsonl").open() as fs, (CENSUS / "scores.jsonl").open() as fc:
        for ls, lc in zip(fs, fc, strict=True):
            score = json.loads(lc)
            if score["family"] != "finish_block" or score["outcome"] != "near_miss":
                continue
            sample = json.loads(ls)
            text = tok.decode([t for t in sample["generated_ids"] if t not in (1, 130073)], skip_special_tokens=False)
            body = text.split("\n>>>>>>> UPDATED")[0]
            target = gold[score["row_id"]]
            if re.sub(r"\s+", "", body) != re.sub(r"\s+", "", target):
                continue
            prompts.add(score["row_id"])
            b, g = body.split("\n"), target.split("\n")
            if [x for x in b if x.strip()] == [x for x in g if x.strip()]:
                kinds["blank_lines_only"] += 1
            elif [x.strip() for x in b if x.strip()] == [x.strip() for x in g if x.strip()]:
                kinds["indentation"] += 1
            else:
                kinds["intra_line"] += 1
    receipt = {"schema": "sepalith.synth.census-whitespace.v1", "criterion": "equal after deleting all whitespace",
               "samples": sum(kinds.values()), "prompts": len(prompts), "by_kind": dict(kinds),
               "inputs": {"scores.jsonl": sha(CENSUS / "scores.jsonl"), "samples.jsonl": sha(CENSUS / "samples.jsonl"),
                          "train_rows": sha(TRAIN)},
               "c12_committed_figure": "receipts/C12-census.json observations.near_miss_with_canonical_region_equal:"
                                       " finish_block 309 samples (156 prompts across families)"}
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
