"""SYNTH cohort proposal: v4 TRAIN rows verbatim, then the SYNTH rows in id
order, scheduled with the unchanged editing-SFT schedule builder (schedule_v2,
seed 3407), plus the trainer's length profile. Mirrors how C10 built the
arm-B cohort. Writes a proposal only; nothing is bound or enqueued.

Default composition is 14 edits + 2 no-ops per update (12.5% no-op), which
keeps every no-op row at <= 3 exposures (the standing rule). The builder
asserts that bound directly from the draws: schedule_v2's own
``noop_exposures_at_most_cap`` compares against its own pass count, so it is
true by construction and cannot enforce the rule. ``--compare-13-3`` also
writes the 13 + 3 schedule for comparison; it violates the rule and is marked
so in the receipt.

schedule_v2 labels every schedule ``sft-schedule-v2-13edit-3noop``. For any
other composition this script rewrites ``policy.schedule_version`` to the true
composition (e.g. ``sft-schedule-v2-14edit-2noop``); recipe_v2 accepts only
the 13 + 3 label, so binding a 14 + 2 schedule fails closed until a recipe
change is approved.

v3.1: the receipt reports token shares next to the draw shares (SYNTH share of
finish target tokens, of all supervised tokens and of finish prompt tokens, and
the medium+long share of finish target tokens). ``--exposures 2`` writes the
2-exposure variant: every SYNTH row appears twice (copy ids ``<id>~x2``), so
the unchanged builder draws each SYNTH row twice per run; still 14 + 2 and
<= 3 no-op exposures. The dose itself is not decided here.

    PYTHONPATH=packages/sepalith/src python docs/training/plan-20260927/synth/make_cohort.py \\
        --synth /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v2 \\
        --out /mnt/e/sepalith/campaign-20260915/data-work/SYNTH-v2/cohort-proposal [--compare-13-3]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from sepalith.campaign_protocol import validate_training_row
from sepalith.data.synth import profile
from sepalith.data.synth.cohort import assert_noop_cap, dose, noop_exposures, repeat_rows
from sepalith.training.sft.recipe_v2 import build_length_profile
from sepalith.training.sft.schedule_v2 import write_schedule

TRAIN = Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl")
TRAIN_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
EFFECTIVE_BATCH = 16
SYNTH_IDS: set[str] = set()
LENGTH_CLASS: dict[str, str] = {}


def length_class(n: int) -> str:
    return "short" if n <= 8 else "medium" if n <= 20 else "long"


def finish_length_classes(synth: Path) -> dict[str, str]:
    """Row id -> short/medium/long under ``profile.target_length``, TRAIN and SYNTH finish rows."""
    out: dict[str, str] = {}
    bodies = {}
    with profile.TRAIN_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] == "finish_block" and row["target_operation"] == "replace":
                bodies[row["id"]] = row["target_body_text"].split("\n")
    with profile.TRAIN_SIDECAR.open(encoding="utf-8") as stream:
        for line in stream:
            side = json.loads(line)
            if side["row_id"] in bodies:
                out[side["row_id"]] = length_class(profile.target_length(side["context"]["region_old"],
                                                                         bodies[side["row_id"]]))
    with (synth / "cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["family"] == "finish_block":
                out[record["row_id"]] = length_class(profile.target_length(record["context"]["region_old"],
                                                                           record["region_new"]))
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def build_schedule(cohort: Path, cohort_sha: str, out: Path, noop_per_batch: int) -> tuple[dict, dict]:
    edits = EFFECTIVE_BATCH - noop_per_batch
    path = out / f"draw-schedule-v2-train-plus-synth-{edits}edit-{noop_per_batch}noop.json"
    result = write_schedule(cohort, cohort_sha, path, seed=3407, noop_per_batch=noop_per_batch)
    label = f"sft-schedule-v2-{edits}edit-{noop_per_batch}noop"
    if result["policy"]["schedule_version"] != label:
        result["policy"]["schedule_version"] = label
        result["policy"]["label_rewritten_by"] = "synth/make_cohort.py (schedule_v2 always writes the 13+3 label)"
        path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        cap = assert_noop_cap(result)
        cap["passes"] = True
    except AssertionError as error:
        cap = {"passes": False, "error": str(error), "histogram": dict(sorted(Counter(noop_exposures(result).values()).items()))}
    summary = {key: result[key] for key in ("max_steps", "draw_count", "checks") if key in result}
    summary["dose"] = dose(result["draws"], SYNTH_IDS, LENGTH_CLASS)
    draws = summary["dose"]["draws"]
    summary["finish_draw_share"] = {
        "finish_draws": draws["finish"], "synth_finish_draws": draws["synth_finish"],
        "synth_share_of_finish_draws": draws["synth_share_of_finish_draws"],
        "synth_share_of_all_draws": draws["synth_share_of_all_draws"]}
    summary["noop_share"] = result["coverage"]["noop_share"]
    return {"path": str(path), "sha256": sha256(path), "schedule_version": label, "noop_per_batch": noop_per_batch,
            "summary": summary, "noop_exposure_rule": cap}, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--noop-per-batch", type=int, default=2)
    parser.add_argument("--compare-13-3", action="store_true")
    parser.add_argument("--exposures", type=int, default=1, choices=(1, 2),
                        help="SYNTH exposures per run (2 = every SYNTH row twice)")
    args = parser.parse_args()
    if sha256(TRAIN) != TRAIN_SHA256:
        raise SystemExit("TRAIN rows hash mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    synth_rows = args.synth / "train-token-rows.jsonl"
    synth_lines = synth_rows.read_text(encoding="utf-8").splitlines(keepends=True)
    synth_lines.sort(key=lambda line: json.loads(line)["id"])
    SYNTH_IDS.update(json.loads(line)["id"] for line in synth_lines)
    LENGTH_CLASS.update(finish_length_classes(args.synth))
    families = Counter()
    for line in synth_lines:
        families[validate_training_row(json.loads(line))["family"]] += 1
    cohort = args.out / "train-plus-synth-cohort-rows.jsonl"
    with cohort.open("w", encoding="utf-8") as out, TRAIN.open(encoding="utf-8") as train:
        for line in train:
            out.write(line)
        for line in repeat_rows(synth_lines, args.exposures):
            out.write(line)
    cohort_sha = sha256(cohort)
    profile = build_length_profile(cohort, cohort_sha)
    (args.out / "length-profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    primary, primary_result = build_schedule(cohort, cohort_sha, args.out, args.noop_per_batch)
    if not primary["noop_exposure_rule"]["passes"]:
        raise SystemExit(f"default schedule violates the no-op exposure rule: {primary['noop_exposure_rule']}")
    receipt = {
        "schema": "sepalith.synth.cohort-proposal.v2",
        "train_rows": {"path": str(TRAIN), "sha256": TRAIN_SHA256, "copied_byte_for_byte": True},
        "synth_rows": {"path": str(synth_rows), "sha256": sha256(synth_rows), "rows": len(synth_lines),
                       "families": dict(families)},
        "cohort": {"path": str(cohort), "sha256": cohort_sha, "rows": 15006 + args.exposures * len(synth_lines),
                   "synth_exposures": args.exposures,
                   "copy_ids": "none" if args.exposures == 1 else "<id>~x2 for the second copy of every SYNTH row"},
        "length_profile": {"path": str(args.out / "length-profile.json"),
                           "sha256": sha256(args.out / "length-profile.json"),
                           "max_sequence_tokens": profile["sequence_tokens"]["max"]},
        "schedule": primary,
        "status": "proposal; not bound to a recipe, not enqueued",
        "binding_note": "recipe_v2 accepts only the 13+3 schedule label; a 14+2 arm needs an approved recipe change",
    }
    if args.compare_13_3:
        comparison, _ = build_schedule(cohort, cohort_sha, args.out, 3)
        comparison["status"] = "comparison only; violates the <= 3 no-op exposure rule" if not \
            comparison["noop_exposure_rule"]["passes"] else "comparison only"
        receipt["schedule_comparison_13_3"] = comparison
    (args.out / "cohort-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"cohort_rows": receipt["cohort"]["rows"], "schedule": primary["schedule_version"],
                      "noop_rule": primary["noop_exposure_rule"], "dose": primary["summary"]["dose"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
