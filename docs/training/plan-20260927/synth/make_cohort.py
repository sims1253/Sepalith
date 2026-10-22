"""SYNTH-v1 cohort proposal: v4 TRAIN rows verbatim, then the SYNTH rows in id
order, scheduled with the unchanged editing-SFT schedule builder (schedule_v2,
13 edits + 3 no-ops per update, seed 3407). This mirrors how C10 built the
arm-B cohort. It writes a proposal only; nothing is bound or enqueued.

    PYTHONPATH=packages/sepalith/src python docs/training/plan-20260927/synth/make_cohort.py \
        --synth /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1 \
        --out /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v1/cohort-proposal
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from sepalith.campaign_protocol import validate_training_row
from sepalith.training.sft.schedule_v2 import write_schedule

TRAIN = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TRAIN_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(TRAIN) != TRAIN_SHA256:
        raise SystemExit("TRAIN rows hash mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    synth_rows = args.synth / "train-token-rows.jsonl"
    synth_lines = synth_rows.read_text(encoding="utf-8").splitlines(keepends=True)
    synth_lines.sort(key=lambda line: json.loads(line)["id"])
    families = Counter()
    for line in synth_lines:
        row = validate_training_row(json.loads(line))
        families[row["family"]] += 1
    cohort = args.out / "train-plus-synth-cohort-rows.jsonl"
    with cohort.open("w", encoding="utf-8") as out, TRAIN.open(encoding="utf-8") as train:
        for line in train:
            out.write(line)
        for line in synth_lines:
            out.write(line)
    cohort_sha = sha256(cohort)
    schedule = args.out / "draw-schedule-v2-train-plus-synth.json"
    result = write_schedule(cohort, cohort_sha, schedule, seed=3407)
    summary = {key: result[key] for key in ("max_steps", "draw_count", "coverage", "checks") if key in result}
    receipt = {
        "schema": "sepalith.synth.cohort-proposal.v1",
        "train_rows": {"path": str(TRAIN), "sha256": TRAIN_SHA256, "copied_byte_for_byte": True},
        "synth_rows": {"path": str(synth_rows), "sha256": sha256(synth_rows), "rows": len(synth_lines),
                       "families": dict(families)},
        "cohort": {"path": str(cohort), "sha256": cohort_sha},
        "schedule": {"path": str(schedule), "sha256": sha256(schedule), "builder": "sepalith.training.sft.schedule_v2",
                     "seed": 3407, "summary": summary},
        "status": "proposal; not bound to a recipe, not enqueued",
    }
    (args.out / "cohort-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"cohort_rows": 15006 + len(synth_lines), "cohort_sha256": cohort_sha}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
