#!/usr/bin/env python3
"""Bind the card C10 arm-B recipe with the unchanged C04 machinery (recipe_v2).

Arm B differs from arm A (docs/training/plan-20260927/c04/bound/sft-v2-lr1e-5.json)
only in its TRAIN cohort and draw schedule: the 15,006 v4 TRAIN rows plus the
3,861 admitted roxygen rows (receipts/C10-roxygen-admission.json), and the
accepted v4 arm-B schedule (receipts/C10-arm-b-schedule-v4.json: 1,368 updates,
gates 342/684/1,026/1,368). Everything else comes from recipe_v2 as it is: LR
1e-5 (selected by R2 in C08), parent CPT 11,586, aurora_mix with the CPT
mechanics, 32 warmup updates and cosine decay to 10% at the arm's own horizon,
R1 gates at the quarters, DEV75 + DEV250 with the bound generation batching,
retain_all checkpoints, and the same source manifest as arm A
(c04/bound/source-manifest.json), so both arms bind byte-identical code.

recipe_v2.build_template hard-codes two descriptive fields for the 15,006-row
cohort (``cohort.scope`` and the data queue's roxygen summary). They are
corrected here for the arm-B cohort before the template is written; no package
code changes, so arm A's bound recipe keeps verifying against main.

Run from the repository root with the training interpreter (CPU only):

    env PYTHONPATH=packages/sepalith/src CUDA_VISIBLE_DEVICES= nice -n 10 \\
        .venv-sft/bin/python -B docs/training/plan-20260927/c10/bind_arm_b.py

Outputs are write-once; move old ones aside to rebind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepalith.training.sft import c04_jobs, recipe_v2

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
ARM_B = DATA / "C10-roxygen-admission-v1/arm-b-v4"
ROWS = ARM_B / "arm-b-cohort-rows.jsonl"
ROWS_SHA256 = "256a418b914014f12b7a848e4d899e210407435cbf98c243f98d50225e8b92ec"
SCHEDULE = ARM_B / "draw-schedule-v2-armb.json"
SCHEDULE_SHA256 = "1e89713e95b8ab789b56077b9c01ad78c6b469014cd803b40ea7add0e04598d2"
COHORT_RECEIPT = ARM_B / "cohort-receipt.json"
LENGTH_PROFILE = ARM_B / "length-profile.json"
TRAIN_V4_ROWS = DATA / "DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl"
TRAIN_V4_SHA256 = "ed1c7b4ea9a7f7e0bfad744dc23f4d08866f9776818044d8584697d41ece3aef"
DEV75 = Path("/mnt/e/sepalith/campaign-20260915/campaign-overlay/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl")
DEV250 = DATA / "C05-dev250-v1r3/dev250-v1.jsonl"
DEV250_MANIFEST = DATA / "C05-dev250-v1r3/manifest.json"
PARENT = Path("/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-resume-20260921/full/checkpoint-11586")
LEARNING_RATE = 1e-5
ARM = "arm-b-lr1e-5"
COHORT_ID = "C10-arm-b-v4-train15006-plus-roxygen3861"
SCOPE = "c10_arm_b_v4_train15006_plus_admitted_roxygen3861"
ROXY_QUEUE_ROWS, ROXY_ADMITTED = 10017, 3861
ROXY_REMAINING = ROXY_QUEUE_ROWS - ROXY_ADMITTED
ROXY_REMAINING_STATUS = "held_or_pending_root_admission"
STOP_AFTER_GATE = 684
PRIORITY = 40
PURPOSE = ("C10 SFT arm B: 15,006 v4 TRAIN rows plus 3,861 admitted roxygen rows, LR 1e-5 (R2, C08), "
           "gates 342/684/1,026/1,368 under R1; compared with arm A (checkpoint 536) under R3")
DECIDED_BY = ("user decisions of 2026-09-27 in docs/training/plan-20260927/README.md (parent 11,586; roxygen admission and "
              "arm B in parallel with arm A, card C10), of 2026-10-12 (C10 option a: v4 rows, 13 edit + 3 no-op, up to 4 "
              "no-op exposures; receipts/C10-arm-b-schedule-v4.json), and R2's selection of lr1e-5 on 2026-10-14 "
              "(status/C08.json)")


def data_queue() -> recipe_v2.Json:
    return {
        "schema": "sepalith.sft11.full-weight-edit-data-queue.v2",
        "active_cohort": {
            "status": "accepted_root_reviewed",
            "rows": 18867,
            "rows_path": str(ROWS),
            "rows_sha256": ROWS_SHA256,
            "composition": {
                "train_rows": {"rows": 15006, "path": str(TRAIN_V4_ROWS), "sha256": TRAIN_V4_SHA256,
                               "accepted_by": "docs/training/plan-20260927/c04/v4/data-queue.json (sha256 0024483b...27ee)"},
                "admitted_roxygen_rows": ROXY_ADMITTED,
            },
            "acceptance": ("C10 admission of 3,861 of the 3,903 recommended roxygen rows (licence, dedup, split and "
                           "semantic checks; receipts/C10-roxygen-admission.json sha256 39863075...e673, PR #50) and the "
                           "user decision of 2026-10-12 (option a, v4 rows)"),
            "cohort_receipt": {"path": str(COHORT_RECEIPT), "sha256": recipe_v2.sha256(COHORT_RECEIPT)},
        },
        "additional_roxy_rows": ROXY_REMAINING,
        "additional_roxy_status": ROXY_REMAINING_STATUS,
        "pending_cohorts": [{
            "id": "roxygen-supported-context-10017-v1",
            "rows": ROXY_QUEUE_ROWS,
            "admitted_rows": ROXY_ADMITTED,
            "not_admitted_rows": ROXY_REMAINING,
            "not_admitted": {
                "held_by_context_admission_review": 6114,
                "held_by_c10_admission": 42,
                "note": ("later DAT-10 reviews recovered 4,694 of the 6,114 (DAT10-roxy8597 union); they still need "
                         "semantic, split and licence admission and a fresh cohort (status/C10.json notes)"),
            },
            "status": "partially_admitted_rest_held_or_pending",
        }],
        "policy": "No pending row is silently included or permanently excluded. A fresh cohort/schedule/recipe is required after root admission.",
    }


def job(root: Path, recipe: Path, *, full: bool) -> recipe_v2.Json:
    args = ["arm", "--recipe", str(recipe)]
    if full:
        identifier, minutes = f"C10-{ARM}-full", 360
        note = ("C10 arm B to the end of its schedule (1,368 updates; gates 342, 684, 1,026, 1,368 under R1). Resumes from "
                "the latest durable checkpoint, so it also continues a run stopped at gate 684. Enqueue instead of, or "
                "after, C10-arm-b-lr1e-5, never together with it")
    else:
        identifier, minutes = f"C10-{ARM}", 200
        args += ["--stop-after-gate", str(STOP_AFTER_GATE)]
        note = ("C10 arm B to gate 684 (50% of its schedule, the progress of arm A's chosen checkpoint 536), gates 342 "
                "and 684 under R1, then stop. With --stop-after-gate, an R1 stop-for-user at gate 684 exits 0 with "
                "the summary 'did not pass the R1 first-gate floor'; read decision-step-684-*.json in the morning")
    return {"id": identifier, "cwd": str(root), "env": [], "priority": PRIORITY, "resumable": True,
            "estimated_minutes": minutes, "requires": ["C07-baseline"], "command": c04_jobs.command(root, *args), "note": note}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy-root", type=Path, default=Path("/home/m0hawk/sandbox/Sepalith"))
    parser.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    paths = recipe_v2.PathMap(args.deploy_root, args.local_root.resolve())
    c04 = args.local_root / "docs/training/plan-20260927/c04"
    out = args.local_root / "docs/training/plan-20260927/c10"
    recipe_v2.require(recipe_v2.sha256(SCHEDULE) == SCHEDULE_SHA256, "arm-B schedule differs from the accepted v4 schedule")
    arm_a = json.loads((c04 / "bound/sft-v2-lr1e-5.json").read_text())

    queue_path = out / "arm-b/data-queue.json"
    recipe_v2.write_new(queue_path, data_queue())
    template = recipe_v2.build_template(
        rows=ROWS, rows_sha256=ROWS_SHA256, schedule=SCHEDULE, materialization=COHORT_RECEIPT, length_profile=LENGTH_PROFILE,
        data_queue=queue_path, optimizer_dispatch=paths.to_local(arm_a["optimizer_dispatch"]["path"]), dev75=DEV75,
        dev250=DEV250, dev250_manifest=DEV250_MANIFEST, paths=paths, cohort_id=COHORT_ID)
    cohort, queue = template["cohort"], template["data_queue"]
    assert isinstance(cohort, dict) and isinstance(queue, dict)
    cohort["scope"] = SCOPE
    queue.update(additional_roxy_rows=ROXY_REMAINING, additional_roxy_status=ROXY_REMAINING_STATUS)
    template_path = out / "arm-b/sft-v2-arm-b-recipe.template.json"
    recipe_v2.write_new(template_path, template)

    admission_path = out / f"arm-b/admission-{ARM}.json"
    recipe_v2.write_new(admission_path, recipe_v2.parent_admission(
        PARENT, step=11586, learning_rate=LEARNING_RATE, arm=ARM, purpose=PURPOSE,
        template_sha256=recipe_v2.sha256(template_path), decided_by=DECIDED_BY))
    bound_path = out / f"bound/sft-v2-{ARM}.json"
    recipe_v2.bind(template_path, admission_path, bound_path, source_manifest=c04 / "bound/source-manifest.json", paths=paths)

    deploy_recipe = paths.to_deploy(bound_path)
    for full in (False, True):
        value = job(args.deploy_root, Path(deploy_recipe), full=full)
        recipe_v2.write_new(out / f"jobs/{value['id']}.json", value)
    print(json.dumps({"bound": str(bound_path), "sha256": recipe_v2.sha256(bound_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
