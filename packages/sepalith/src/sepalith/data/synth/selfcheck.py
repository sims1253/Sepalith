"""Independent self-check of a written SYNTH-v1 directory (reads only the files).

For every row: the sidecar context re-renders to the row's ``prompt_text``;
re-encoding the prompt reproduces the row's prompt token IDs; the row passes
``validate_training_row``; ids are unique and disjoint from the v4 TRAIN ids;
and reward v2.1 scores the gold completion ``exact`` with reward 1.0
(the same check C12 ran on the 15,006 TRAIN rows).

    PYTHONPATH=packages/sepalith/src .venv-sft/bin/python -m sepalith.data.synth.selfcheck \
        --data /mnt/e/.../SYNTH-v1 --out receipts/SYNTH-v1-selfcheck.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from sepalith.campaign_protocol import PromptContext, encode_prompt, render_prompt, validate_training_row
from sepalith.data.synth.build import TRAIN_ROWS, load_tokenizer, make_scorer, sha256_file
from sepalith.training.rl.reward_v21 import Evidence, ReferenceSpec, Sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.data / "train-token-rows.jsonl").open(encoding="utf-8")]
    sidecar = {r["row_id"]: r for r in map(json.loads, (args.data / "context-sidecar.jsonl").open(encoding="utf-8"))}
    train_ids = {json.loads(line)["id"] for line in TRAIN_ROWS.open(encoding="utf-8")}
    tokenizer = load_tokenizer()
    scorer, _formatter, decoder = make_scorer(min(args.workers, 8))
    problems: Counter = Counter()
    ids = Counter(row["id"] for row in rows)
    problems["duplicate_ids"] = sum(1 for n in ids.values() if n > 1)
    problems["ids_in_train"] = len(set(ids) & train_ids)
    specs, samples = [], []
    for row in rows:
        validate_training_row(row)
        side = sidecar.get(row["id"])
        if side is None:
            problems["missing_sidecar"] += 1
            continue
        context = PromptContext.from_dict(side["context"])
        if render_prompt(context) != row["prompt_text"]:
            problems["prompt_render_mismatch"] += 1
        if encode_prompt(context, tokenizer) != row["input_ids"][:row["target_start"]]:
            problems["prompt_ids_mismatch"] += 1
        if side["family"] != row["family"] or side["package_id"] != row["package_id"]:
            problems["sidecar_identity_mismatch"] += 1
        specs.append(ReferenceSpec(row["id"], row["family"], context, row["target_operation"],
                                   row["target_body_text"], Evidence("unverified")))
        samples.append(Sample(row["id"], completion_ids=tuple(row["input_ids"][row["target_start"]:]),
                              sample_id="gold"))
    records = scorer.score(scorer.build_references(specs), samples, decoder)
    outcome = Counter((r["family"], r["outcome"]) for r in records)
    comparison = Counter((r["family"], r["comparison"]) for r in records)
    flags = Counter(flag for r in records for flag in r["flags"])
    exact = sum(1 for r in records if r["exact"] and float(str(r["reward"])) == 1.0)
    receipt = {
        "schema": "sepalith.synth.selfcheck.v1",
        "data": {name: sha256_file(args.data / name) for name in ("train-token-rows.jsonl", "context-sidecar.jsonl")},
        "rows": len(rows),
        "gold_exact_reward_1": exact,
        "outcomes": {f"{f}:{o}": n for (f, o), n in sorted(outcome.items())},
        "comparisons": {f"{f}:{c}": n for (f, c), n in sorted(comparison.items())},
        "reference_flags": dict(flags),
        "problems": {k: v for k, v in problems.items()},
        "scorer": scorer.identity(),
        "status": "pass" if exact == len(rows) and not any(problems.values()) else "fail",
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("rows", "gold_exact_reward_1", "problems", "status")}, indent=1))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
