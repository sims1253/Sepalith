#!/usr/bin/env python3
"""Recompute the frozen remaining-CPT admission facts without training imports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
PACKET = PLAN / "docs/campaign/work/lead/r2-cloud-cpt-admission-v1"
PRODUCER = PLAN / "docs/campaign/work/lead/r2-cpt-remaining-v1"
RECEIPT_PATH = PLAN / "docs/campaign/receipts/SFT-11-CPT-remaining-preparation.json"
OUT = PACKET / "cpt-frozen-independent-review.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    receipt = load(RECEIPT_PATH)
    require(sha256(RECEIPT_PATH) == "ce34e52357802e4e95e632a6eb7115ae70f89e89ba8a64f75ffe9c6ad0973f9f", "producer receipt changed")

    artifact_checks = {}
    for name, record in receipt["artifacts"].items():
        path = Path(record["path"])
        require(path.is_file(), f"missing receipt artifact: {name}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        require(actual == {"bytes": record["bytes"], "sha256": record["sha256"]}, f"receipt artifact mismatch: {name}")
        artifact_checks[name] = actual

    recipe_path = Path(receipt["artifacts"]["recipe"]["path"])
    recipe = load(recipe_path)
    source_manifest_path = Path(receipt["artifacts"]["source_manifest"]["path"])
    source_manifest = load(source_manifest_path)
    source_root = PRODUCER / "source"
    source_entries = []
    source_mode_mismatches = []
    for expected in source_manifest["files"]:
        path = source_root / expected["path"]
        actual = {
            "path": expected["path"],
            "sha256": sha256(path),
            "mode": os.stat(path).st_mode & 0o777,
        }
        require(actual["sha256"] == expected["sha256"], f"source byte mismatch: {expected['path']}")
        if actual["mode"] != expected["mode"]:
            source_mode_mismatches.append({"path": expected["path"], "expected": expected["mode"], "actual": actual["mode"]})
        source_entries.append(expected)
    encoded_entries = json.dumps(source_entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    source_id = hashlib.sha256(encoded_entries).hexdigest()
    require(source_id == source_manifest["id"] == recipe["identity"]["source"] == receipt["source"]["snapshot"], "source identity mismatch")

    expected_parameters = {
        "gradient_accumulation": 8,
        "learning_rate": 0.0001,
        "lora_alpha": 64,
        "lora_rank": 32,
        "max_sequence_tokens": 2048,
        "max_steps": 1902,
        "per_device_batch": 2,
    }
    require(recipe["parameters"] == expected_parameters, "training parameters changed")
    require(recipe["parameters"]["gradient_accumulation"] * recipe["parameters"]["per_device_batch"] == 16, "effective batch mismatch")
    require(recipe["checkpoint"] == {"evaluation_steps": [317, 951, 1585, 1902], "full_every": 317, "light_every": 317}, "checkpoint/evaluation cadence mismatch")
    require(recipe["resume_contract"]["full_cadence_steps"] == [317, 634, 951, 1268, 1585, 1902], "full checkpoint milestones mismatch")
    require(recipe["resume_from"] is None and recipe["resume_contract"]["fresh_lora"] is True and recipe["resume_contract"]["fresh_optimizer"] is True, "stage must initialize fresh LoRA and optimizer")
    require(recipe["identity"]["policy"]["initialization"] == "new_lora_on_merged_cpt_parent", "parent initialization policy mismatch")
    require(recipe["launch_authorized"] is False, "frozen producer recipe unexpectedly authorizes launch")
    require(recipe["decision_steps"] == [], "unexpected decision stop")

    parent_manifest_path = Path(recipe["merged_cpt_parent"]["manifest"]["path"])
    parent_manifest = load(parent_manifest_path)
    parent = recipe["identity"]["parent"]
    require(parent_manifest["kind"] == "merged_cpt", "parent is not a merged CPT model")
    require(parent_manifest["base_model_revision"] == parent["revision"], "parent base revision mismatch")
    require(parent["merged_cpt_manifest_sha256"] == sha256(parent_manifest_path), "parent manifest identity mismatch")
    parent_weight = Path(recipe["model_path"]) / "model.safetensors"
    require(parent["weights_sha256"] == sha256(parent_weight), "parent weight mismatch")
    previous_checkpoint = load(Path(recipe["merged_cpt_parent"]["previous_checkpoint_manifest"]["path"]))
    require(previous_checkpoint["step"] == parent["previous_checkpoint_step"] == 250, "parent checkpoint step mismatch")
    require(previous_checkpoint["full"] is True, "parent checkpoint is not full")
    require(previous_checkpoint["identity"] == parent_manifest["cpt_identity"], "merged parent/checkpoint identity mismatch")

    input_checks = []
    for record in recipe["inputs"]:
        path = Path(record["path"])
        require(path.is_file(), f"missing recipe input: {path}")
        actual = sha256(path)
        require(actual == record["sha256"], f"recipe input hash mismatch: {path}")
        input_checks.append({"path": str(path), "bytes": path.stat().st_size, "sha256": actual})

    train_path = Path(recipe["train_rows"]["path"])
    validation_path = Path(recipe["validation_rows"]["path"])
    train_ids, train_docs, train_groups, train_packages = [], set(), set(), set()
    loss_tokens = input_tokens = payload_tokens = overlap_tokens = 0
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            require(row["cpt_partition"] == "cpt_train", f"non-TRAIN row at {line_number}")
            require(len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"]) <= 2048, f"invalid sequence geometry at {line_number}")
            require(row["row_id"] == f"{row['document_id']}:{row['chunk_index']}", f"row identity mismatch at {line_number}")
            require(row["supervised_tokens"] == sum(label != -100 for label in row["labels"]), f"loss-token count mismatch at {line_number}")
            train_ids.append(row["row_id"])
            train_docs.add(row["document_id"])
            train_groups.add(row["group_id"])
            train_packages.add(row["package"])
            loss_tokens += row["supervised_tokens"]
            input_tokens += len(row["input_ids"])
            payload_tokens += row["source_token_end"] - row["source_token_start"]
            overlap_tokens += row["overlap_context_tokens"]
    require(len(train_ids) == len(set(train_ids)) == 30421, "TRAIN rows are not 30,421 unique identities")

    val_ids, val_docs, val_groups, val_packages = set(), set(), set(), set()
    val_loss_tokens = 0
    with validation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            require(row["cpt_partition"] == "cpt_validation", f"non-validation row at {line_number}")
            require(len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"]) <= 2048, f"invalid validation geometry at {line_number}")
            val_ids.add(row["row_id"])
            val_docs.add(row["document_id"])
            val_groups.add(row["group_id"])
            val_packages.add(row["package"])
            val_loss_tokens += row["supervised_tokens"]
    require(len(val_ids) == 499 and len(val_docs) == 294, "validation cardinality mismatch")
    require(set(train_ids).isdisjoint(val_ids), "TRAIN/validation row overlap")
    require(train_docs.isdisjoint(val_docs), "TRAIN/validation document overlap")
    require(train_groups.isdisjoint(val_groups), "TRAIN/validation group overlap")
    require(train_packages.isdisjoint(val_packages), "TRAIN/validation package overlap")

    schedule_path = Path(recipe["draw_schedule"]["path"])
    schedule = load(schedule_path)
    require(schedule["row_ids"][:30421] == train_ids, "schedule does not cover TRAIN in exact source order")
    require(schedule["row_ids"][30421:] == schedule["replay_row_ids"], "schedule tail is not the declared replay")
    require(len(schedule["row_ids"]) == 30432 and len(schedule["replay_row_ids"]) == 11, "schedule cardinality mismatch")
    require(set(schedule["replay_row_ids"]).issubset(set(train_ids)), "replay names an unknown row")
    require(schedule["effective_batch"] == 16 and schedule["max_steps"] == 1902, "schedule step geometry mismatch")
    require(schedule["token_rows_sha256"] == recipe["train_rows"]["sha256"], "schedule/TRAIN hash binding mismatch")

    data_manifest = load(Path(receipt["artifacts"]["data_manifest"]["path"]))
    selection = data_manifest["selection"]
    require(selection["remaining_distinct_rows"] == len(train_ids), "data-manifest row count mismatch")
    require(selection["documents"] == len(train_docs), "data-manifest document count mismatch")
    require(selection["groups"] == len(train_groups) == selection["packages"] == len(train_packages), "data-manifest package/group count mismatch")
    require(selection["loss_tokens"] == loss_tokens, "data-manifest loss tokens mismatch")
    require(selection["input_tokens"] == input_tokens, "data-manifest input tokens mismatch")
    require(selection["payload_tokens"] == payload_tokens, "data-manifest payload tokens mismatch")
    require(selection["overlap_context_tokens"] == overlap_tokens, "data-manifest overlap tokens mismatch")
    require(sorted(train_packages) == recipe["identity"]["data"]["train_package_ids"], "recipe package identity mismatch")
    require(selection["selected_lineage_distinct_rows_excluded"] == 16000, "selected lineage exclusion count mismatch")
    require(not source_mode_mismatches, f"source mode mismatches: {source_mode_mismatches}")

    result = {
        "schema": "sepalith.cpt.remaining-independent-review.v1",
        "status": "PASS_BOUNDED_STAGE_ROOT_ADMISSION_ELIGIBLE",
        "review_scope": "Frozen remaining-CPT recipe/source/data/parent bindings and producer contract tests; CPU metadata/bytes only.",
        "producer_receipt": {"path": str(RECEIPT_PATH), "sha256": sha256(RECEIPT_PATH)},
        "recipe": {"path": str(recipe_path), "bytes": recipe_path.stat().st_size, "sha256": sha256(recipe_path)},
        "source": {"manifest_path": str(source_manifest_path), "manifest_sha256": sha256(source_manifest_path), "snapshot_id": source_id, "files": len(source_entries)},
        "parent": {"kind": parent_manifest["kind"], "checkpoint_step": 250, "manifest_sha256": sha256(parent_manifest_path), "weights_sha256": sha256(parent_weight), "fresh_lora": True, "fresh_optimizer": True, "inherited_checkpoint_resume": False},
        "data": {"train_rows": len(train_ids), "train_documents": len(train_docs), "train_groups": len(train_groups), "train_packages": len(train_packages), "input_tokens": input_tokens, "payload_tokens": payload_tokens, "loss_tokens": loss_tokens, "overlap_context_tokens": overlap_tokens, "validation_rows": len(val_ids), "validation_documents": len(val_docs), "validation_loss_tokens": val_loss_tokens, "train_validation_rows_documents_groups_packages_disjoint": True, "selected_lineage_rows_excluded": 16000},
        "schedule": {"draws": len(schedule["row_ids"]), "unique_rows_before_replay": 30421, "declared_replay_rows": 11, "effective_batch": 16, "steps": 1902, "full_checkpoints": [317, 634, 951, 1268, 1585, 1902], "evaluations": [317, 951, 1585, 1902], "source_order_exact": True},
        "inputs": {"count": len(input_checks), "all_bytes_and_hashes_match": True, "records": input_checks},
        "producer_contract_tests": {"independent_result": "5 PASS", "log_path": str(PACKET / "cpt-producer-tests-independent.log"), "log_sha256": sha256(PACKET / "cpt-producer-tests-independent.log")},
        "admission_findings": {
            "scientific_or_identity_blockers": [],
            "required_before_launch": ["Root must create the affirmative admission record; frozen recipe correctly has launch_authorized=false."],
            "scope_limit": "This stage exhausts the 30,421 eligible rows in the two admitted/materialized shards. It does not materialize the wider 8,092-group upstream TRAIN inventory.",
            "policy_condition": "Admit this bounded incremental stage only with the broader all-eligible continuation retained; the producer's broader command remains intentionally non-executable pending an uncapped reviewed materializer.",
            "cpu_preflight_note": "The producer's 46.44-second CPU preflight reported PASS and CUDA_started=false. This review independently rehashed all inputs, including the merged parent weights, and did not repeat the 4.12 GiB-RSS trainer preflight while another local training branch was active.",
        },
        "actions_performed": {"cuda": False, "training": False, "cloud_upload": False, "paid_launch": False, "mutated_producer_artifacts": False},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUT), "sha256": sha256(OUT), "train_rows": len(train_ids), "draws": len(schedule["row_ids"]), "inputs": len(input_checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
