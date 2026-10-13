#!/usr/bin/env python3
"""Independent CPU-only review of the corrected RL fresh5 packet.

This review intentionally does not import the training stack.  It checks the
immutable packet, the source-backed row/sidecar geometry, the finite sampler
schedule, protocol/no-op controls, and the root admission metadata.  It does
not read model weights, final outputs, checkpoints, or live run state.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


# Keep this process to one CPU.  This is process-local and disappears when the
# review exits.  No training/runtime module is imported by this file.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

try:
    _before_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {_before_affinity[0]})
    _after_affinity = sorted(os.sched_getaffinity(0))
except (AttributeError, OSError, IndexError):
    _before_affinity = []
    _after_affinity = []


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
CAMPAIGN = PLAN / "docs/campaign"
AUDIT = CAMPAIGN / "work/corrected-rl-admission-audit-v1"
LEAD = CAMPAIGN / "work/lead/corrected-rl-fresh5-a"
OUT = CAMPAIGN / "work/corrected-rl-fresh5-independent-review"
RECEIPT = CAMPAIGN / "receipts/RL-08-corrected-rl-fresh5-independent-review.json"
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACT_MANIFEST = AUDIT / "artifact-manifest.json"
SOURCE_PINS = AUDIT / "source-pins.json"
CANDIDATE_RECIPE = AUDIT / "fresh5.recipe.json"
FINAL_RECIPE = LEAD / "recipe.json"
ROOT_ADMISSION = CAMPAIGN / "receipts/RL-08-corrected-fresh5-root-admission.json"

FAILURES: list[dict[str, object]] = []
CHECKS: dict[str, dict[str, object]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def compact_hash(value: object) -> str:
    return hashlib.sha256(compact_json(value)).hexdigest()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check(name: str, ok: bool, detail: object) -> None:
    result = {"ok": bool(ok), "detail": detail}
    CHECKS[name] = result
    if not ok:
        FAILURES.append({"check": name, "detail": detail})


def short_failure(kind: str, row_id: str | None, detail: str) -> None:
    # Keep the review useful without copying row contents into the receipt.
    if len(ROW_FAILURES) < 25:
        ROW_FAILURES.append({"kind": kind, "row_id": row_id, "detail": detail})


def canonical_without(value: object, excluded: set[str]) -> object:
    if isinstance(value, dict):
        return {
            key: canonical_without(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [canonical_without(item, excluded) for item in value]
    return value


def recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value.keys())
        for item in value.values():
            keys.update(recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursive_keys(item))
    return keys


def nonempty(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def equivalent_region(selection_region: object, context_region: object) -> bool:
    """Treat [] and [''] as the same zero-width region representation."""
    if selection_region == context_region:
        return True
    return (
        isinstance(selection_region, list)
        and isinstance(context_region, list)
        and selection_region == [""]
        and context_region == []
    )


ROW_FAILURES: list[dict[str, object]] = []


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    candidate = load_json(CANDIDATE_RECIPE)
    final = load_json(FINAL_RECIPE)
    artifact_manifest = load_json(ARTIFACT_MANIFEST)
    source_pins = load_json(SOURCE_PINS)
    data_audit = load_json(AUDIT / "data-audit.json")
    candidate_preflight = load_json(AUDIT / "candidate-cpu-preflight.json")
    independent_preflight = load_json(AUDIT / "independent-candidate-cpu-preflight.json")
    production_tests = load_json(AUDIT / "production-cpu-tests.json")
    timing = load_json(AUDIT / "timing-and-budget.json")
    allocation = load_json(AUDIT / "candidate-data/allocation-policy.json")
    sequence = load_json(AUDIT / "candidate-data/source-row-draw-sequence.json")
    selected_doc = load_json(AUDIT / "candidate-data/selected-train-ids.json")
    root_admission = load_json(ROOT_ADMISSION)
    root_verification = load_json(LEAD / "root-verification.json")
    root_preflight = load_json(LEAD / "root-full-preflight.json")
    supervisor = load_json(LEAD / "supervisor-launch.json")
    guard_process = load_json(LEAD / "guard-process.json")
    command = load_json(LEAD / "command.json")

    expected_candidate_recipe_sha = (
        "6661af89eb83fd95070f939290a3e74ff2c1e0b09c561d2d16a1704f32bbabee"
    )
    expected_final_recipe_sha = (
        "9042c8e1ad3f5c9580551360678a65fbe66d8f9fc980fa2c68eb94825bd7e281"
    )
    expected_final_identity_sha = (
        "ae851d72f158518f979f61b70f0a292151aed2a017c0cd71b282527ddc604bed"
    )
    expected_full_preflight_sha = (
        "9f4edaa5e8046af0cb768e1481ec56b142812959cffecc69ffde18d9f877f674"
    )
    expected_guard_sha = (
        "9518f5d10ea92c424df10f4d49be32f5dd1bf8a42a0deae5d13ea2998f18a845"
    )
    expected_data_admission_sha = (
        "cd088bdab29d873eca71ea71bdaa5c7e8f21048371a5b4084bf1fbc87b393770"
    )

    candidate_recipe_sha = sha256_file(CANDIDATE_RECIPE)
    final_recipe_sha = sha256_file(FINAL_RECIPE)
    root_admission_sha = sha256_file(ROOT_ADMISSION)
    full_preflight_sha = sha256_file(LEAD / "root-full-preflight.json")
    root_verification_sha = sha256_file(LEAD / "root-verification.json")
    supervisor_sha = sha256_file(LEAD / "supervisor-launch.json")
    guard_process_sha = sha256_file(LEAD / "guard-process.json")
    command_sha = sha256_file(LEAD / "command.json")

    # Verify every immutable candidate artifact and every pinned source file.
    # The source files are hashed/read as bytes only; no source module is
    # imported, and the parent model/weights are never opened.
    artifact_results: list[dict[str, object]] = []
    artifact_ok = True
    for entry in artifact_manifest["files"]:
        path = CAMPAIGN / entry["path"]
        actual_exists = path.is_file()
        actual_bytes = path.stat().st_size if actual_exists else None
        actual_sha = sha256_file(path) if actual_exists else None
        item = {
            "path": str(path),
            "expected_bytes": entry["bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_sha,
            "match": actual_exists
            and actual_bytes == entry["bytes"]
            and actual_sha == entry["sha256"],
        }
        artifact_results.append(item)
        artifact_ok = artifact_ok and bool(item["match"])
    check(
        "candidate_artifact_manifest",
        artifact_ok,
        {"count": len(artifact_results), "mismatches": [x for x in artifact_results if not x["match"]]},
    )

    source_results: list[dict[str, object]] = []
    source_ok = True
    for entry in source_pins["files"]:
        path = Path(entry["path"])
        actual_exists = path.is_file()
        actual_bytes = path.stat().st_size if actual_exists else None
        actual_sha = sha256_file(path) if actual_exists else None
        item = {
            "path": str(path),
            "expected_bytes": entry["bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_sha,
            "match": actual_exists
            and actual_bytes == entry["bytes"]
            and actual_sha == entry["sha256"],
        }
        source_results.append(item)
        source_ok = source_ok and bool(item["match"])
    check(
        "frozen_source_pins",
        source_ok,
        {"count": len(source_results), "mismatches": [x for x in source_results if not x["match"]]},
    )

    lead_results: list[dict[str, object]] = []
    for path in [
        FINAL_RECIPE,
        ROOT_ADMISSION,
        LEAD / "root-full-preflight.json",
        LEAD / "root-verification.json",
        LEAD / "supervisor-launch.json",
        LEAD / "guard-process.json",
        LEAD / "command.json",
    ]:
        lead_results.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    check(
        "recipe_hashes",
        candidate_recipe_sha == expected_candidate_recipe_sha
        and final_recipe_sha == expected_final_recipe_sha,
        {
            "candidate": candidate_recipe_sha,
            "candidate_expected": expected_candidate_recipe_sha,
            "final": final_recipe_sha,
            "final_expected": expected_final_recipe_sha,
        },
    )

    # The final recipe is deliberately the candidate runtime identity with
    # root admission metadata and the larger checkpoint reserve bound changed.
    identity_mismatches: list[str] = []
    for section in ("parent", "policy", "renderer", "schedule", "tokenizer"):
        if candidate["identity"][section] != final["identity"][section]:
            identity_mismatches.append("identity." + section)
    candidate_data_identity = canonical_without(
        candidate["identity"]["data"], {"row_identities", "selected_ids"}
    )
    final_data_identity = canonical_without(final["identity"]["data"], {"row_identities", "selected_ids"})
    if candidate_data_identity != final_data_identity:
        identity_mismatches.append("identity.data")
    candidate_source_identity = canonical_without(
        candidate["identity"]["source"],
        {"rl02_admission_receipt_sha256", "rl02_admission_status"},
    )
    final_source_identity = canonical_without(
        final["identity"]["source"],
        {"rl02_admission_receipt_sha256", "rl02_admission_status"},
    )
    if candidate_source_identity != final_source_identity:
        identity_mismatches.append("identity.source.runtime")
    runtime_excluded = {
        "checkpoint_reserve_seconds",
        "preparation_status",
        "rl02_admission",
        "identity",
    }
    candidate_runtime = {k: v for k, v in candidate.items() if k not in runtime_excluded}
    final_runtime = {k: v for k, v in final.items() if k not in runtime_excluded}
    if candidate_runtime != final_runtime:
        identity_mismatches.append("top_level.runtime")
    check(
        "final_preserves_candidate_runtime_identity",
        not identity_mismatches,
        {"mismatches": identity_mismatches, "final_identity_sha256": expected_final_identity_sha},
    )

    expected_runtime = {
        "candidate_count": 4,
        "prompt_max_tokens": 2048,
        "completion_max_tokens": 192,
        "context_max_tokens": 2240,
        "model_load_max_seq_length": 4096,
        "candidate_buffer_reuse": 8,
        "source_draws": 24000,
        "source_draws_per_update": 8,
        "generation_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "steps_per_generation": 8,
        "max_steps": 5,
        "full_save_steps": 5,
        "evaluation_steps": [5],
        "decision_steps": [5],
        "resume_from": None,
        "learning_rate": 2e-5,
        "warmup_steps": 0,
        "renderer_id": "zeta2-prm03-v1",
        "sampler_id": "campaign-repeat-manifest-order-v1",
        "seed": 3407,
        "loss_type": "bnpo",
        "scale_rewards": "group",
        "beta": 0,
    }
    actual_runtime = {
        "candidate_count": final["identity"]["policy"]["candidate_count"],
        "prompt_max_tokens": final["identity"]["policy"]["prompt_max_tokens"],
        "completion_max_tokens": final["identity"]["policy"]["completion_max_tokens"],
        "context_max_tokens": final["identity"]["policy"]["context_max_tokens"],
        "model_load_max_seq_length": final["identity"]["policy"]["model_load_max_seq_length"],
        "candidate_buffer_reuse": final["data"]["buffer_reuse"],
        "source_draws": final["identity"]["schedule"]["source_draws"],
        "source_draws_per_update": final["identity"]["schedule"]["source_draws_per_update"],
        "generation_batch_size": final["identity"]["schedule"]["generation_batch_size"],
        "gradient_accumulation_steps": final["identity"]["policy"]["gradient_accumulation_steps"],
        "steps_per_generation": final["identity"]["schedule"]["steps_per_generation"],
        "max_steps": final["max_steps"],
        "full_save_steps": final["full_save_steps"],
        "evaluation_steps": final["evaluation_steps"],
        "decision_steps": final["decision_steps"],
        "resume_from": final["resume_from"],
        "learning_rate": final["learning_rate"],
        "warmup_steps": final["warmup_steps"],
        "renderer_id": final["renderer_id"],
        "sampler_id": final["identity"]["schedule"]["sampler_id"],
        "seed": final["identity"]["schedule"]["seed"],
        "loss_type": final["identity"]["policy"]["loss_type"],
        "scale_rewards": final["identity"]["policy"]["scale_rewards"],
        "beta": final["identity"]["policy"]["beta"],
    }
    check("runtime_geometry_and_policy", actual_runtime == expected_runtime, {"actual": actual_runtime, "expected": expected_runtime})

    # Stream the rows and sidecar.  Keep only compact metadata after checking
    # each full JSON object so that this remains a CPU data review, not a model
    # or framework operation.
    selected_ids = selected_doc["row_ids"]
    selected_set = set(selected_ids)
    selected_path = AUDIT / "candidate-data/selected-train-ids.json"
    selected_hash = sha256_file(selected_path)
    check(
        "selected_ids_manifest",
        selected_doc.get("schema_version") == "sepalith.prm07.selected-train-ids.v1"
        and selected_doc.get("split") == "train"
        and len(selected_ids) == 8246
        and len(selected_set) == len(selected_ids)
        and selected_hash == final["data"]["selected_ids_sha256"],
        {"count": len(selected_ids), "unique": len(selected_set), "sha256": selected_hash},
    )

    rows: dict[str, dict[str, object]] = {}
    row_family_counts: collections.Counter[str] = collections.Counter()
    row_operation_counts: collections.Counter[str] = collections.Counter()
    row_geometry_failures = 0
    row_eos_failures = 0
    row_format_failures = 0
    rows_path = AUDIT / "candidate-data/eligible-train-rows.jsonl"
    with rows_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row_geometry_failures += 1
                short_failure("row_json", None, f"line {line_number}: {exc}")
                continue
            row_id = row.get("id")
            if not isinstance(row_id, str):
                row_geometry_failures += 1
                short_failure("row_id", None, f"line {line_number}")
                continue
            if row_id in rows:
                row_geometry_failures += 1
                short_failure("duplicate_row", row_id, f"line {line_number}")
            rows[row_id] = {
                "family": row.get("family"),
                "package_id": row.get("package_id"),
                "operation": row.get("target_operation"),
                "prompt_sha256": hashlib.sha256(row.get("prompt_text", "").encode("utf-8")).hexdigest(),
            }
            row_family_counts[row.get("family")] += 1
            row_operation_counts[row.get("target_operation")] += 1
            body_count = row.get("target_body_token_count")
            terminal_count = row.get("target_terminal_token_count")
            target_count = row.get("target_token_count")
            prompt_count = row.get("prompt_token_count")
            target_start = row.get("target_start")
            input_ids = row.get("input_ids")
            body_tokens = row.get("target_body_tokens")
            terminal_tokens = row.get("target_terminal_tokens")
            geometry_ok = (
                row.get("split") == "train"
                and row.get("bos_token_id") == 0
                and row.get("eos_token_id") == 1
                and row.get("renderer_id") == "zeta2-prm03-v1"
                and row.get("tokenizer_revision") == final["identity"]["tokenizer"]["revision"]
                and row.get("tokenizer_json_sha256") == final["identity"]["tokenizer"]["json_sha256"]
                and row.get("tokenization_policy") == final["identity"]["renderer"]["tokenization_policy"]
                and isinstance(input_ids, list)
                and isinstance(body_tokens, list)
                and isinstance(terminal_tokens, list)
                and isinstance(prompt_count, int)
                and isinstance(target_start, int)
                and isinstance(body_count, int)
                and isinstance(terminal_count, int)
                and isinstance(target_count, int)
                and target_start == prompt_count + 1
                and target_start <= 2048
                and target_count == body_count + terminal_count
                and len(body_tokens) == body_count
                and len(terminal_tokens) == terminal_count
                and len(input_ids) == target_start + target_count + 1
                and len(input_ids) >= target_start + target_count + 1
                and input_ids[0] == 0
                and input_ids[target_start : target_start + body_count] == body_tokens
                and input_ids[target_start + body_count : target_start + target_count] == terminal_tokens
                and input_ids[-1] == 1
                and target_count + 1 <= 192
                and target_start + target_count + 1 <= 2240
            )
            if not geometry_ok:
                row_geometry_failures += 1
                short_failure("row_geometry", row_id, "recorded token/renderer geometry mismatch")
            if isinstance(input_ids, list) and 1 in input_ids[: max(0, target_start or 0)]:
                row_eos_failures += 1
                short_failure("prompt_eos", row_id, "EOS appears before target start")
            expected_target = f"{row.get('target_body_text', '')}\n>>>>>>> UPDATED"
            format_ok = (
                row.get("target_text") == expected_target
                and isinstance(row.get("target_body_text"), str)
                and row.get("target_text", "").endswith("\n>>>>>>> UPDATED")
            )
            if not format_ok:
                row_format_failures += 1
                short_failure("target_format", row_id, "target text is not body plus terminal")

    row_set = set(rows)
    check(
        "row_identity_and_geometry",
        len(rows) == 8246
        and row_set == selected_set
        and row_geometry_failures == 0
        and row_eos_failures == 0
        and row_format_failures == 0,
        {
            "rows": len(rows),
            "selected_minus_rows": len(selected_set - row_set),
            "rows_minus_selected": len(row_set - selected_set),
            "families": dict(sorted(row_family_counts.items())),
            "operations": dict(sorted(row_operation_counts.items())),
            "geometry_failures": row_geometry_failures,
            "prompt_eos_failures": row_eos_failures,
            "target_format_failures": row_format_failures,
        },
    )

    side_meta: dict[str, dict[str, object]] = {}
    row_identities: dict[str, dict[str, object]] = {}
    side_geometry_failures = 0
    side_context_failures = 0
    side_source_failures = 0
    side_prompt_hash_failures = 0
    required_context_keys = {
        "cursor",
        "diagnostics",
        "document_eol",
        "history",
        "path",
        "prefix",
        "region_old",
        "replacement_range",
        "retrieval",
        "schema_version",
        "scope_lines",
        "scope_mode",
        "selected_references",
        "suffix_lines",
    }
    forbidden_context_keys = {
        "target",
        "target_text",
        "target_body_text",
        "target_tokens",
        "reward",
        "reward_score",
        "generated",
        "completion",
        "expected",
        "label",
    }
    side_path = AUDIT / "candidate-data/context-sidecar.jsonl"
    with side_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                side = json.loads(line)
            except json.JSONDecodeError as exc:
                side_context_failures += 1
                short_failure("sidecar_json", None, f"line {line_number}: {exc}")
                continue
            row_id = side.get("row_id")
            context = side.get("context")
            geometry = side.get("selection_geometry")
            source_identity = side.get("source_identity")
            if not isinstance(row_id, str):
                side_context_failures += 1
                short_failure("sidecar_row_id", None, f"line {line_number}")
                continue
            if row_id in side_meta:
                side_context_failures += 1
                short_failure("duplicate_sidecar", row_id, f"line {line_number}")
            side_meta[row_id] = {
                "family": side.get("family"),
                "package_id": side.get("package_id"),
                "prompt_sha256": side.get("prompt_sha256"),
                "geometry": geometry,
                "context": context,
            }
            row_identities[row_id] = {
                "id": row_id,
                "selection_geometry": geometry,
                "source_identity": source_identity,
            }
            context_keys = set(context) if isinstance(context, dict) else set()
            nested_context_keys = recursive_keys(context)
            geometry_ok = (
                isinstance(geometry, dict)
                and geometry.get("availability") in {"full_snapshot", "source_builder_window"}
                and geometry.get("budget_utf16_units") == 6000
                and geometry.get("overflow") is False
                and geometry.get("required_overflow") is False
                and isinstance(geometry.get("used_utf16_units"), int)
                and isinstance(geometry.get("required_utf16_units"), int)
                and geometry["used_utf16_units"] <= geometry["budget_utf16_units"]
                and geometry["required_utf16_units"] <= geometry["used_utf16_units"]
                and equivalent_region(
                    geometry.get("region"),
                    context.get("region_old") if isinstance(context, dict) else None,
                )
                and geometry.get("context_range") == (context.get("replacement_range") if isinstance(context, dict) else None)
                and geometry.get("policy_id") == "prm05-source-balanced-v1"
                and geometry.get("policy_id_combined") == "prm05-selection-dropout-v2"
            )
            if not geometry_ok:
                side_geometry_failures += 1
                short_failure("selection_geometry", row_id, "source selection geometry mismatch")
            context_ok = (
                isinstance(context, dict)
                and required_context_keys.issubset(context_keys)
                and side.get("context_has_target_or_reward_keys") is False
                and side.get("offline_static_source") is True
                and not (forbidden_context_keys & nested_context_keys)
            )
            if not context_ok:
                side_context_failures += 1
                short_failure("context_scope", row_id, "context missing required fields or contains target/reward fields")
            source_ok_for_row = (
                isinstance(source_identity, dict)
                and source_identity.get("row_id") in (None, row_id)
                and source_identity.get("package_id") == side.get("package_id")
                and isinstance(source_identity.get("source_ref"), dict)
                and source_identity["source_ref"].get("row_id") == row_id
                and source_identity["source_ref"].get("package_id") == side.get("package_id")
                and source_identity["source_ref"].get("family") == side.get("family")
                and isinstance(source_identity["source_ref"].get("source"), str)
                and isinstance(source_identity["source_ref"].get("source_sha256"), str)
                and isinstance(source_identity.get("source_provenance"), dict)
            )
            if not source_ok_for_row:
                side_source_failures += 1
                short_failure("source_identity", row_id, "source identity/provenance mismatch")
            corresponding = rows.get(row_id)
            if corresponding and corresponding["prompt_sha256"] != side.get("prompt_sha256"):
                side_prompt_hash_failures += 1
                short_failure("prompt_hash", row_id, "sidecar prompt hash differs from row prompt")

    side_set = set(side_meta)
    check(
        "sidecar_context_source_geometry",
        len(side_meta) == 8246
        and side_set == selected_set
        and side_geometry_failures == 0
        and side_context_failures == 0
        and side_source_failures == 0
        and side_prompt_hash_failures == 0,
        {
            "sidecar_rows": len(side_meta),
            "selected_minus_sidecar": len(selected_set - side_set),
            "sidecar_minus_selected": len(side_set - selected_set),
            "selection_geometry_failures": side_geometry_failures,
            "context_scope_failures": side_context_failures,
            "source_identity_failures": side_source_failures,
            "prompt_hash_failures": side_prompt_hash_failures,
        },
    )

    # Recompute the row-identity digest from sidecar geometry/source identity
    # in selected-ID order.  This binds the admitted identity to the actual
    # sidecar objects rather than trusting a copied digest.
    canonical_row_identities = [row_identities[row_id] for row_id in selected_ids if row_id in row_identities]
    recomputed_row_identity_sha = compact_hash(canonical_row_identities)
    check(
        "row_identity_digest",
        len(canonical_row_identities) == len(selected_ids)
        and recomputed_row_identity_sha == final["identity"]["data"]["row_identity_sha256"]
        and recomputed_row_identity_sha == sequence["row_identity_sha256"],
        {
            "rows": len(canonical_row_identities),
            "recomputed_sha256": recomputed_row_identity_sha,
            "recipe_sha256": final["identity"]["data"]["row_identity_sha256"],
            "schedule_sha256": sequence["row_identity_sha256"],
        },
    )

    # Check the materialized schedule, not the diagnostic filtered proposal in
    # data-audit.schedule (which is intentionally 23,635 draws and mod-8 == 3).
    sequence_ids = sequence["row_ids"]
    sequence_counts = collections.Counter(sequence_ids)
    sequence_family_counts = collections.Counter(rows[row_id]["family"] for row_id in sequence_ids if row_id in rows)
    sequence_source_counts: collections.Counter[str] = collections.Counter()
    for row_id, exposure in sequence_counts.items():
        if row_id in side_meta:
            source_ref = row_identities[row_id]["source_identity"].get("source_ref", {})
            source_key = f"{side_meta[row_id]['package_id']}::{source_ref.get('source')}"
            sequence_source_counts[source_key] += exposure
    recomputed_sequence_sha = compact_hash(sequence_ids)
    recomputed_ordered_sha = compact_hash(selected_ids)
    policy_exposure_mismatches = [
        row_id
        for row_id, expected in allocation["exposures"].items()
        if sequence_counts.get(row_id, 0) != expected
    ]
    policy_source_mismatches = [
        key
        for key, expected in allocation["source_exposures"].items()
        if sequence_source_counts.get(key, 0) != expected
    ]
    policy_source_extra = [key for key in sequence_source_counts if key not in allocation["source_exposures"]]
    family_target_mismatches = {
        family: {"actual": sequence_family_counts.get(family, 0), "expected": expected}
        for family, expected in allocation["family_targets"].items()
        if sequence_family_counts.get(family, 0) != expected
    }
    check(
        "materialized_schedule_and_allocation",
        sequence.get("schema_version") == "sepalith.dat09.source-row-draw-sequence.v1"
        and sequence.get("status") == "candidate_corrected_pool_root_review_required"
        and sequence.get("source_draws") == 24000
        and len(sequence_ids) == 24000
        and len(sequence_ids) % sequence.get("source_draws_per_update", 1) == 0
        and sequence.get("candidate_count") == 4
        and sequence.get("buffer_reuse") == 8
        and sequence.get("source_draws_per_update") == 8
        and sequence.get("prompt_groups_per_update") == 8
        and sequence.get("completions_per_update") == 32
        and sequence.get("gradient_accumulation_steps") == 8
        and sequence.get("steps_per_generation") == 8
        and sequence.get("no_truncation") is True
        and set(sequence_ids).issubset(selected_set)
        and recomputed_sequence_sha == sequence["sequence_sha256"]
        and recomputed_ordered_sha == sequence["ordered_ids_sha256"]
        and sequence.get("selected_ids_sha256") == selected_hash
        and sequence.get("allocation_policy_sha256") == sha256_file(AUDIT / "candidate-data/allocation-policy.json")
        and not policy_exposure_mismatches
        and not policy_source_mismatches
        and not policy_source_extra
        and not family_target_mismatches,
        {
            "draws": len(sequence_ids),
            "unique_rows": len(sequence_counts),
            "max_exposure": max(sequence_counts.values()),
            "min_exposure": min(sequence_counts.values()),
            "exposure_histogram": dict(sorted(collections.Counter(sequence_counts.values()).items())),
            "families": dict(sorted(sequence_family_counts.items())),
            "source_keys": len(sequence_source_counts),
            "family_target_mismatches": family_target_mismatches,
            "row_exposure_mismatches": len(policy_exposure_mismatches),
            "source_exposure_mismatches": len(policy_source_mismatches),
            "source_extra": policy_source_extra[:10],
            "sequence_sha256": recomputed_sequence_sha,
            "ordered_ids_sha256": recomputed_ordered_sha,
            "diagnostic_filtered_schedule": {
                "draws": data_audit["schedule"]["retained_draws"],
                "mod_8": data_audit["schedule"]["retained_mod_8"],
                "interpretation": "diagnostic filtered proposal; not materialized candidate schedule",
            },
        },
    )

    expected_family_targets = {
        "finish_block": 3825,
        "format_propagation": 6000,
        "rename_propagation": 6000,
        "no_op": 4800,
        "pipe_rewrite": 2400,
        "na_rm_propagation": 375,
        "roxygen_drafting": 600,
    }
    check(
        "allocation_policy_geometry",
        allocation.get("candidate_count") == 4
        and allocation.get("buffer_reuse") == 8
        and allocation.get("ordinary_replay_cap") == 8
        and allocation.get("small_pack_cap") == 3
        and allocation.get("seed") == 3407
        and allocation.get("family_targets") == expected_family_targets
        and sum(allocation.get("exposures", {}).values()) == 24000
        and sum(allocation.get("source_exposures", {}).values()) == 24000,
        {
            "candidate_count": allocation.get("candidate_count"),
            "buffer_reuse": allocation.get("buffer_reuse"),
            "ordinary_replay_cap": allocation.get("ordinary_replay_cap"),
            "small_pack_cap": allocation.get("small_pack_cap"),
            "seed": allocation.get("seed"),
            "family_targets": allocation.get("family_targets"),
            "exposure_sum": sum(allocation.get("exposures", {}).values()),
            "source_exposure_sum": sum(allocation.get("source_exposures", {}).values()),
        },
    )

    # The development prefixes are sampler diagnostics and must retain the
    # exact candidate/buffer geometry at both decision boundaries.
    prefix_checks: dict[str, object] = {}
    expected_prefix = {
        "5": {"draws": 40, "generated_candidates": 160, "source_cursor": 40, "sampler_consumed_rows": 1280, "unique_rows": 40, "no_op": 8},
        "25": {"draws": 200, "generated_candidates": 800, "source_cursor": 200, "sampler_consumed_rows": 6400, "unique_rows": 200, "no_op": 40},
    }
    prefixes_ok = True
    for step, expected in expected_prefix.items():
        actual = candidate_preflight["prefixes"].get(step, {})
        family_noop = actual.get("families", {}).get("no_op")
        got = {
            "draws": actual.get("draws"),
            "generated_candidates": actual.get("generated_candidates"),
            "source_cursor": actual.get("source_cursor"),
            "sampler_consumed_rows": actual.get("sampler_consumed_rows"),
            "unique_rows": actual.get("unique_rows"),
            "no_op": family_noop,
        }
        prefix_checks[step] = {"actual": got, "expected": expected, "match": got == expected}
        prefixes_ok = prefixes_ok and got == expected
    check("sampler_prefix_boundaries", prefixes_ok, prefix_checks)

    # No-op controls are checked in the actual row, sidecar, and pinned
    # protocol source.  The sidecar deliberately carries context only; the
    # target/reward fields remain in the row record.
    no_op_ids = [row_id for row_id, meta in rows.items() if meta["operation"] == "no_op"]
    no_op_rows_ok = True
    no_op_region_lengths: collections.Counter[int] = collections.Counter()
    for row_id in no_op_ids:
        side = side_meta.get(row_id, {})
        context = side.get("context", {})
        # Full text is not retained in rows; these exact values are verified
        # while streaming below through a small second pass over no-op rows.
        region_old = context.get("region_old", []) if isinstance(context, dict) else []
        no_op_region_lengths[len(region_old)] += 1
        no_op_rows_ok = no_op_rows_ok and isinstance(region_old, list)
    no_op_target_counts = collections.Counter()
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("target_operation") == "no_op":
                no_op_target_counts[(row.get("target_body_text"), row.get("target_text"))] += 1
    no_op_rows_ok = no_op_rows_ok and no_op_target_counts == collections.Counter({("[NO_EDIT]", "[NO_EDIT]\n>>>>>>> UPDATED"): 1140})
    check(
        "no_op_row_acceptance",
        len(no_op_ids) == 1140
        and row_family_counts.get("no_op") == 1140
        and no_op_rows_ok
        and sequence_family_counts.get("no_op") == 4800,
        {
            "eligible_no_op_rows": len(no_op_ids),
            "no_op_row_targets": {f"{body!r} | {target!r}": count for (body, target), count in no_op_target_counts.items()},
            "region_old_line_counts": dict(sorted(no_op_region_lengths.items())),
            "materialized_no_op_draws": sequence_family_counts.get("no_op", 0),
            "protocol_source": "NO_EDIT + exact >>>>>>> UPDATED; no_op metadata remains unchanged region_old",
        },
    )

    protocol_path = Path(source_pins["files"][9]["path"])
    train_path = Path(source_pins["files"][0]["path"])
    protocol_text = protocol_path.read_text(encoding="utf-8")
    train_text = train_path.read_text(encoding="utf-8")
    protocol_markers = {
        "terminal_constant": 'TERMINAL = ">>>>>>> UPDATED"' in protocol_text,
        "no_edit_constant": 'NO_EDIT = "[NO_EDIT]"' in protocol_text,
        "no_op_target": 'expected_target = f"{NO_EDIT}\\n{TERMINAL}"' in protocol_text,
        "no_op_metadata_guard": "no_op target must equal region_old exactly" in protocol_text,
        "reward_actual_noop_region": "actual = list(context.region_old) if parsed.operation == \"no_op\"" in train_text,
        "sampler_id": 'SAMPLER_ID = "campaign-repeat-manifest-order-v1"' in train_text,
        "bnpo_group": '"loss_type": "bnpo"' in train_text and '"scale_rewards": "group"' in train_text,
    }
    check("pinned_protocol_and_sampler_markers", all(protocol_markers.values()), protocol_markers)

    probes = data_audit["reward_control_probes"]
    reward_probe_ok = len(probes) == 9 and all(probe.get("protocol_valid") is True for probe in probes)
    exact_controls = [p for p in probes if p.get("target") == p.get("generated")]
    mismatch_controls = [p for p in probes if p.get("target") == "corrected" and p.get("generated") == "old"]
    reward_probe_ok = reward_probe_ok and len(exact_controls) == 6 and all(
        p.get("exact_region") is True and p.get("score") == 1.2 for p in exact_controls
    )
    reward_probe_ok = reward_probe_ok and len(mismatch_controls) == 3 and all(
        p.get("exact_region") is False and 0.0 <= p.get("score", 9.0) < 0.2 for p in mismatch_controls
    )
    check(
        "reward_control_probes",
        reward_probe_ok,
        {
            "probe_count": len(probes),
            "exact_controls": len(exact_controls),
            "mismatch_controls": len(mismatch_controls),
            "scores": [p.get("score") for p in mismatch_controls],
            "scope": "actual reward function with explicit decoder control; no model or tokenizer proof",
        },
    )

    # Source-backed all-TRAIN overflow audit and eligible intersection.  This
    # records the rejected bands explicitly; no truncation/reconstruction is
    # accepted by this review.
    all_train = data_audit["all_corrected_TRAIN"]
    reachable = data_audit["corrected_reachable_192"]
    overflow_summary = {
        "all_corrected_train_rows": all_train["unique_rows"],
        "prompt_plus_bos_over_2048": all_train["prompt_plus_bos_over_2048"],
        "target_plus_eos_over_192": all_train["target_plus_eos_over_192"],
        "exact_target_total_over_2240": all_train["exact_target_total_over_2240"],
        "eligible_intersection_rows": reachable["unique_rows"],
        "eligible_prompt_over_2048": reachable["prompt_plus_bos_over_2048"],
        "eligible_target_over_192": reachable["target_plus_eos_over_192"],
        "eligible_context_over_2240": reachable["exact_target_total_over_2240"],
        "overflow_band_policy": "excluded before materialized schedule; no truncation or fabricated prefix",
    }
    check(
        "overflow_controls",
        all_train["unique_rows"] == 11526
        and all_train["prompt_plus_bos_over_2048"] == 1251
        and all_train["target_plus_eos_over_192"] == 2031
        and reachable["unique_rows"] == 8246
        and reachable["prompt_plus_bos_over_2048"] == 0
        and reachable["target_plus_eos_over_192"] == 0
        and reachable["exact_target_total_over_2240"] == 0,
        overflow_summary,
    )

    # Scope and launch separation.  Candidate CPU checks were framework-free;
    # root's final entry preflight is the separate parent-weight/resource gate.
    framework_flags = {
        **candidate_preflight.get("framework_imports", {}),
        **independent_preflight.get("framework_imports", {}),
        **root_preflight.get("framework_imports", {}),
    }
    scope_ok = (
        data_audit["scope"] == {"one_cpu": True, "model_or_weights": False, "GPU": False, "training": False, "final_data": False}
        and production_tests.get("GPU") is False
        and production_tests.get("weights_loaded") is False
        and production_tests.get("one_cpu") is True
        and production_tests.get("failures") == 0
        and production_tests.get("errors") == 0
        and all(value is False for value in framework_flags.values())
        and independent_preflight.get("launch_performed") is False
        and root_preflight.get("CUDA_started") is False
        and root_preflight.get("training_started") is False
    )
    check(
        "cpu_only_scope_and_framework_gate",
        scope_ok,
        {
            "audit_scope": data_audit["scope"],
            "production_tests": {k: production_tests.get(k) for k in ("tests_run", "failures", "errors", "GPU", "weights_loaded", "one_cpu")},
            "framework_imports": framework_flags,
            "candidate_launch_performed": independent_preflight.get("launch_performed"),
            "root_preflight_CUDA_started": root_preflight.get("CUDA_started"),
            "root_preflight_training_started": root_preflight.get("training_started"),
        },
    )

    # Candidate pending metadata is superseded by the final root admission.
    admission_ok = (
        root_admission.get("status") == "admitted_bounded_fresh5"
        and root_admission.get("recipe_sha256") == final_recipe_sha == expected_final_recipe_sha
        and root_admission.get("identity_sha256") == expected_final_identity_sha
        and root_admission.get("full_preflight_sha256") == full_preflight_sha == expected_full_preflight_sha
        and root_admission.get("parent_weights_verified") is True
        and root_admission.get("data_receipt_sha256") == expected_data_admission_sha
        and root_admission.get("guard_sha256") == expected_guard_sha
        and root_admission.get("max_attempt_seconds") == 1800
        and root_admission.get("max_guard_seconds") == 1860
        and root_admission.get("reserve_seconds") == 750
        and root_verification.get("status") == "pass"
        and root_verification.get("parent_weights_verified") is True
        and root_verification.get("recipe_sha256") == expected_final_recipe_sha
        and root_verification.get("full_preflight_sha256") == expected_full_preflight_sha
        and supervisor.get("pid") is not None
        and supervisor.get("start_tick") is not None
        and guard_process.get("pid") is not None
        and guard_process.get("start_tick") is not None
        and "--seconds" in supervisor.get("argv", [])
        and "1860" in supervisor.get("argv", [])
        and isinstance(command, list)
        and str(FINAL_RECIPE) in command
    )
    check(
        "final_root_admission_and_guard",
        admission_ok,
        {
            "root_admission_sha256": root_admission_sha,
            "root_status": root_admission.get("status"),
            "recipe_sha256": root_admission.get("recipe_sha256"),
            "identity_sha256": root_admission.get("identity_sha256"),
            "full_preflight_sha256": root_admission.get("full_preflight_sha256"),
            "parent_weights_verified": root_admission.get("parent_weights_verified"),
            "reserve_seconds": root_admission.get("reserve_seconds"),
            "max_attempt_seconds": root_admission.get("max_attempt_seconds"),
            "max_guard_seconds": root_admission.get("max_guard_seconds"),
            "guard_sha256": root_admission.get("guard_sha256"),
            "supervisor": {"pid": supervisor.get("pid"), "start_tick": supervisor.get("start_tick")},
            "command_contains_final_recipe": isinstance(command, list) and str(FINAL_RECIPE) in command,
            "guard": {"pid": guard_process.get("pid"), "start_tick": guard_process.get("start_tick")},
            "final_access": root_admission.get("final_access"),
        },
    )

    # The protocol acceptance target is a future live milestone.  Keeping it
    # explicit prevents a successful launch receipt from being mistaken for a
    # quality result.
    acceptance = root_admission.get("acceptance", {})
    milestone_pending = {
        "full5_checkpoint_and_all75_corrected_dev": "pending live run; no output/checkpoint read by this review",
        "edits_minimum": ">=26/43",
        "strict_noops_minimum": ">=25/32",
        "false_suggestions_maximum": "<=5",
        "corrected_finish_cases": 6,
        "continuation": acceptance.get("continuation"),
        "promotion": acceptance.get("promotion"),
    }

    data_identity = {
        "candidate_rows_sha256": sha256_file(rows_path),
        "candidate_sidecar_sha256": sha256_file(side_path),
        "selected_ids_sha256": selected_hash,
        "source_schedule_file_sha256": sha256_file(AUDIT / "candidate-data/source-row-draw-sequence.json"),
        "source_sequence_sha256": sequence["sequence_sha256"],
        "row_identity_sha256": recomputed_row_identity_sha,
        "row_count": len(rows),
        "sidecar_count": len(side_meta),
        "train_dev_overlap": data_audit["train_dev_overlap"],
        "changed_finish_targets": data_audit["rows"]["changed_targets"],
        "excluded_old_rl_rows": data_audit["rows"]["excluded_old_RL"],
    }

    review = {
        "schema": "sepalith.rl08.corrected-rl-fresh5-independent-review.v1",
        "task": "RL-08",
        "status": "pass_final_root_admission; live_milestone5_quality_pending" if not FAILURES else "blocked_review_checks_failed",
        "at": started.isoformat(),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "review_scope": {
            "cpu_only": True,
            "one_cpu_affinity_before": _before_affinity,
            "one_cpu_affinity_after": _after_affinity,
            "one_cpu_verified": len(_after_affinity) == 1,
            "framework_imports": framework_flags,
            "weights_read": False,
            "GPU_used": False,
            "native_launch_by_reviewer": False,
            "SSH_or_network_used": False,
            "final_outputs_or_checkpoints_read": False,
            "live_state_or_lease_modified": False,
        },
        "input_identities": {
            "candidate_recipe_sha256": candidate_recipe_sha,
            "candidate_recipe_expected_sha256": expected_candidate_recipe_sha,
            "final_recipe_sha256": final_recipe_sha,
            "final_identity_sha256": expected_final_identity_sha,
            "candidate_artifact_manifest_sha256": sha256_file(ARTIFACT_MANIFEST),
            "candidate_source_pins_sha256": sha256_file(SOURCE_PINS),
            "root_admission_sha256": root_admission_sha,
            "root_verification_sha256": root_verification_sha,
            "root_full_preflight_sha256": full_preflight_sha,
            "supervisor_launch_sha256": supervisor_sha,
            "guard_process_sha256": guard_process_sha,
            "command_sha256": command_sha,
            "final_root_identity_sha256_reported": expected_final_identity_sha,
        },
        "artifact_manifest": {
            "candidate_files": len(artifact_results),
            "candidate_manifest_status": artifact_manifest.get("status"),
            "all_hashes_match": artifact_ok,
            "frozen_source_files": len(source_results),
            "all_source_hashes_match": source_ok,
        },
        "identity": {
            "candidate_to_final_runtime_mismatches": identity_mismatches,
            "runtime_identity_preserved": not identity_mismatches,
            "final_recipe_changes": {
                "checkpoint_reserve_seconds": {"candidate": candidate.get("checkpoint_reserve_seconds"), "final": final.get("checkpoint_reserve_seconds")},
                "preparation_status": {"candidate": candidate.get("preparation_status"), "final": final.get("preparation_status")},
                "data_admission_receipt": {"candidate": candidate["rl02_admission"], "final": final["rl02_admission"]},
            },
        },
        "data_identity": data_identity,
        "source_geometry": {
            "eligible_rows": len(rows),
            "eligible_sidecars": len(side_meta),
            "prompt_bos_max": data_audit["corrected_reachable_192"]["prompt_plus_bos_max"],
            "target_eos_max": data_audit["corrected_reachable_192"]["target_plus_eos_max"],
            "source_budget_utf16_units": 6000,
            "selection_availability_counts": collections.Counter(
                meta["geometry"].get("availability") for meta in side_meta.values() if isinstance(meta.get("geometry"), dict)
            ),
            "source_provenance_simulated_flag_counts": {
                "false": sum(
                    meta.get("context") is not None
                    and isinstance(row_identities[row_id]["source_identity"].get("source_provenance"), dict)
                    and row_identities[row_id]["source_identity"]["source_provenance"].get("source_snapshot_is_simulated") is False
                    for row_id, meta in side_meta.items()
                ),
                "true": sum(
                    isinstance(row_identities[row_id]["source_identity"].get("source_provenance"), dict)
                    and row_identities[row_id]["source_identity"]["source_provenance"].get("source_snapshot_is_simulated") is True
                    for row_id in side_meta
                ),
                "absent": sum(
                    isinstance(row_identities[row_id]["source_identity"].get("source_provenance"), dict)
                    and "source_snapshot_is_simulated" not in row_identities[row_id]["source_identity"]["source_provenance"]
                    for row_id in side_meta
                ),
            },
            "overflow_controls": overflow_summary,
        },
        "schedule": {
            "materialized_draws": len(sequence_ids),
            "materialized_unique_rows": len(sequence_counts),
            "family_counts": dict(sorted(sequence_family_counts.items())),
            "no_op_draws": sequence_family_counts.get("no_op", 0),
            "max_row_exposure": max(sequence_counts.values()),
            "allocation_policy_sha256": sha256_file(AUDIT / "candidate-data/allocation-policy.json"),
            "sequence_sha256": recomputed_sequence_sha,
            "ordered_ids_sha256": recomputed_ordered_sha,
            "prefixes": prefix_checks,
            "diagnostic_filter_caveat": "data-audit retained 23,635/mod-8=3 is a filtered diagnostic; admitted sequence is 24,000/mod-8=0",
        },
        "protocol_and_reward": {
            "no_op_eligible_rows": len(no_op_ids),
            "no_op_target_exact": no_op_rows_ok,
            "pinned_markers": protocol_markers,
            "reward_probe_count": len(probes),
            "reward_probe_protocol_valid": all(probe.get("protocol_valid") is True for probe in probes),
            "reward_probe_exact_controls": len(exact_controls),
            "reward_probe_corrected_vs_old_controls": len(mismatch_controls),
            "reward_probe_scope_caveat": "decoder controls exercise actual reward code; they are not model output or tokenizer proof",
        },
        "root_admission": {
            "status": root_admission.get("status"),
            "parent_weights_verified": root_admission.get("parent_weights_verified"),
            "full_preflight_sha256": root_admission.get("full_preflight_sha256"),
            "guard_sha256": root_admission.get("guard_sha256"),
            "reserve_seconds": root_admission.get("reserve_seconds"),
            "max_attempt_seconds": root_admission.get("max_attempt_seconds"),
            "max_guard_seconds": root_admission.get("max_guard_seconds"),
            "final_access": root_admission.get("final_access"),
            "budget": root_admission.get("budget"),
        },
        "acceptance_pending": milestone_pending,
        "checks": CHECKS,
        "row_failures_sample": ROW_FAILURES,
        "limits": [
            "This review did not inspect model weights, framework execution, live run state, checkpoints, final outputs, or sealed evaluation.",
            "The full-TRAIN overflow counts are source audit controls; only the 8,246 reachable intersection is materialized in the admitted schedule.",
            "Reward probes are explicit decoder controls and do not establish model learning or quality.",
            "No promotion or continuation is implied. Root milestone-5 acceptance remains required: full5 plus corrected DEV75 and finish/no-op signal review.",
        ],
    }

    input_manifest = {
        "schema": "sepalith.rl08.corrected-rl-fresh5-independent-review-inputs.v1",
        "task": "RL-08",
        "review_started_at": started.isoformat(),
        "candidate_artifacts": artifact_results,
        "frozen_source_pins": source_results,
        "lead_metadata": lead_results,
        "forbidden_reads": [
            "parent model/merged weight bytes",
            "framework/model imports",
            "live process/state/lease files",
            "fresh5 checkpoints/final outputs/sealed evaluation",
        ],
    }

    input_path = OUT / "input-manifest.json"
    review_path = OUT / "review.json"
    input_path.write_text(json.dumps(input_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_sha = sha256_file(review_path)
    input_sha = sha256_file(input_path)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    receipt = {
        "schema": "sepalith.campaign.receipt.v1",
        "task": "RL-08",
        "status": review["status"],
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owner": "stress_fixture",
        "review": str(review_path),
        "review_sha256": review_sha,
        "input_manifest": str(input_path),
        "input_manifest_sha256": input_sha,
        "review_script": str(script_path),
        "review_script_sha256": script_sha,
        "candidate_recipe_sha256": candidate_recipe_sha,
        "final_recipe_sha256": final_recipe_sha,
        "final_identity_sha256": expected_final_identity_sha,
        "root_admission_sha256": root_admission_sha,
        "root_full_preflight_sha256": full_preflight_sha,
        "frozen_source_pins_sha256": sha256_file(SOURCE_PINS),
        "candidate_artifact_count": len(artifact_results),
        "frozen_source_file_count": len(source_results),
        "cpu_only": True,
        "one_cpu_verified": len(_after_affinity) == 1,
        "weights_read": False,
        "GPU_used": False,
        "native_launch": False,
        "launch_blocker": None if not FAILURES else "; ".join(str(f["check"]) for f in FAILURES),
        "result": {
            "identity_scope_data_schedule_noop_reward_checks": "pass" if not FAILURES else "fail",
            "final_root_launch_metadata": "admitted_bounded_fresh5",
            "live_milestone5_quality": "pending; root must inspect full5 + corrected DEV75 before continuation",
            "materialized_draws": len(sequence_ids),
            "materialized_unique_rows": len(sequence_counts),
            "family_counts": dict(sorted(sequence_family_counts.items())),
            "no_op_draws": sequence_family_counts.get("no_op", 0),
            "eligible_rows": len(rows),
            "overflow_excluded_rows_reported": overflow_summary,
            "train_dev_overlap": data_audit["train_dev_overlap"],
        },
        "limits": review["limits"],
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": review["status"],
        "failures": FAILURES,
        "review": str(review_path),
        "review_sha256": review_sha,
        "input_manifest": str(input_path),
        "input_manifest_sha256": input_sha,
        "receipt": str(RECEIPT),
        "receipt_sha256": sha256_file(RECEIPT),
        "script_sha256": script_sha,
        "materialized_draws": len(sequence_ids),
        "materialized_unique_rows": len(sequence_counts),
        "no_op_draws": sequence_family_counts.get("no_op", 0),
    }, indent=2))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
