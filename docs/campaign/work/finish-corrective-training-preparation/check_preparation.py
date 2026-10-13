"""Framework-free checks for the SFT-06 feasibility skeleton.

This module checks the preparation contract only.  It deliberately does not
open model, train-row, overlay, or DEV payload files and cannot authorize a
launch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SKELETON = HERE / "corrective-sft-recipe-skeleton.json"
OVERLAY_RECEIPT = HERE.parents[1] / "receipts" / "DAT-04-finish-target-overlay-v1-preparation.json"
EXPECTED_OVERLAY_RECEIPT_SHA256 = (
    "c88e79f55d0841da56a56d115a5e024756666ab9bd171ee73b2897b099d8c45f"
)


class PreparationError(ValueError):
    """Raised when the non-admitted feasibility contract is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_skeleton(path: Path = SKELETON) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise PreparationError("skeleton root must be an object")
    return value


def validate_skeleton(recipe: dict[str, Any], *, receipt_path: Path = OVERLAY_RECEIPT) -> dict[str, Any]:
    """Validate the bounded template and return compact evidence.

    Placeholders are required because corrected token rows and the draw
    schedule have not been materialized.  This function treats the template
    as invalid if it could be mistaken for an admitted recipe.
    """

    errors: list[str] = []

    if recipe.get("schema_version") != 1:
        errors.append("schema_version")
    if recipe.get("status") != "template_not_admitted":
        errors.append("status")
    if recipe.get("launch_authorized") is not False:
        errors.append("launch_authorized")

    overlay = recipe.get("overlay_preparation")
    if not isinstance(overlay, dict):
        errors.append("overlay_preparation")
    else:
        expected_overlay = {
            "packet_rows": 5000,
            "lineage_matched_rows": 4289,
            "eligible_strict_v2_rows": 4051,
            "rejected_rows": 949,
            "nonfinish_rows_untouched": 7475,
            "semantic_noop_rows_preserved": 1140,
            "tokenization_performed": False,
            "original_inputs_mutated": False,
        }
        for key, expected in expected_overlay.items():
            if overlay.get(key) != expected:
                errors.append(f"overlay_preparation.{key}")
        if overlay.get("status") != "shadow_only_tokenization_pending":
            errors.append("overlay_preparation.status")
        if overlay.get("receipt_sha256") != EXPECTED_OVERLAY_RECEIPT_SHA256:
            errors.append("overlay_preparation.receipt_sha256")
        if overlay.get("overlay_sha256") != (
            "03cff340de0c2b0e641a0399ddc8de727c39c9aa5e51aeed984c6420117054cf"
        ):
            errors.append("overlay_preparation.overlay_sha256")
        reasons = overlay.get("rejected_reason_counts")
        if reasons != {
            "registry_or_sft_lineage_missing": 711,
            "target_not_lf_terminated": 238,
        }:
            errors.append("overlay_preparation.rejected_reason_counts")

    materialization = recipe.get("required_input_materialization")
    if not isinstance(materialization, dict):
        errors.append("required_input_materialization")
    else:
        expected_materialization = {
            "base_train_rows": 11764,
            "corrected_train_rows": 11764,
            "changed_rows_pending_tokenization": 4051,
            "rejected_finish_ids_excluded_from_schedule": 238,
            "absent_finish_ids_not_reconstructed": 711,
            "required_schedule_draws": 3200,
            "required_effective_batch": 16,
        }
        for key, expected in expected_materialization.items():
            if materialization.get(key) != expected:
                errors.append(f"required_input_materialization.{key}")
        dev = materialization.get("corrected_dev_candidate")
        if not isinstance(dev, dict) or dev.get("records") != 75 or dev.get("changed_targets") != 6:
            errors.append("required_input_materialization.corrected_dev_candidate")
        elif dev.get("status") != "candidate_independent_review_pending":
            errors.append("required_input_materialization.corrected_dev_candidate.status")

    parameters = recipe.get("parameters", {})
    schedule = recipe.get("identity", {}).get("schedule", {})
    for key in (
        "max_steps",
        "per_device_batch",
        "gradient_accumulation",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "max_sequence_tokens",
    ):
        if parameters.get(key) != schedule.get(key):
            errors.append(f"identity.schedule_matches_parameters.{key}")
    if parameters.get("per_device_batch", 0) * parameters.get("gradient_accumulation", 0) != 16:
        errors.append("effective_batch")
    if parameters.get("max_steps") != 200:
        errors.append("max_steps")

    checkpoints = recipe.get("checkpoint", {})
    if checkpoints.get("light_every") != 50 or checkpoints.get("full_every") != 50:
        errors.append("checkpoint_cadence")
    if checkpoints.get("evaluation_steps") != [50, 100, 200]:
        errors.append("evaluation_steps")
    if recipe.get("decision_steps") != [50]:
        errors.append("decision_steps")
    milestone = recipe.get("milestone_plan", {})
    if milestone.get("stage_identity_max_steps") != 200:
        errors.append("milestone_plan.stage_identity_max_steps")
    if milestone.get("draws") != 3200 or milestone.get("effective_batch") != 16:
        errors.append("milestone_plan.geometry")
    attempts = milestone.get("attempts")
    expected_attempts = [
        {"label": "fresh_to_50", "decision_steps": [50], "resume_from": None},
        {"label": "resume_50_to_100", "decision_steps": [100], "resume_from": "FULL_CHECKPOINT_50"},
        {"label": "resume_100_to_200", "decision_steps": [200], "resume_from": "FULL_CHECKPOINT_100"},
    ]
    if attempts != expected_attempts:
        errors.append("milestone_plan.attempts")

    data = recipe.get("identity", {}).get("data", {})
    for key in ("train_rows_sha256", "draw_schedule_sha256", "development_panel_sha256", "correction_manifest_sha256"):
        if not str(data.get(key, "")).startswith("REQUIRED_"):
            errors.append(f"placeholder_required.{key}")
    token_rows = recipe.get("token_rows", {})
    draw_schedule = recipe.get("draw_schedule", {})
    dev_panel = recipe.get("development_panel", {})
    for field, value in (
        ("token_rows.path", token_rows.get("path")),
        ("token_rows.sha256", token_rows.get("sha256")),
        ("draw_schedule.path", draw_schedule.get("path")),
        ("draw_schedule.sha256", draw_schedule.get("sha256")),
        ("development_panel.path", dev_panel.get("path")),
        ("development_panel.sha256", dev_panel.get("sha256")),
    ):
        if not str(value).startswith("REQUIRED_"):
            errors.append(f"placeholder_required.{field}")

    if receipt_path.exists() and _sha256(receipt_path) != EXPECTED_OVERLAY_RECEIPT_SHA256:
        errors.append("overlay_receipt_file_sha256")

    if errors:
        raise PreparationError("; ".join(errors))
    return {
        "status": recipe["status"],
        "launch_authorized": recipe["launch_authorized"],
        "max_steps": parameters["max_steps"],
        "effective_batch": parameters["per_device_batch"] * parameters["gradient_accumulation"],
        "draws": milestone["draws"],
        "overlay_eligible": overlay["eligible_strict_v2_rows"],
        "overlay_rejected": overlay["rejected_rows"],
        "overlay_tokenized": overlay["tokenization_performed"],
    }


def main() -> int:
    evidence = validate_skeleton(load_skeleton())
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
