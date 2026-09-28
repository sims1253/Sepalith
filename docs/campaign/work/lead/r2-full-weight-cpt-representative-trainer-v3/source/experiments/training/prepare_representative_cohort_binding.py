#!/usr/bin/env python3
"""Validate and bind the received representative CPT cohort without scientific choices."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from campaign_cpt_data import validate_draw_schedule, validate_materialized_rows

EXPECTED = {
    "rows": ("/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/cpt_train_ctx16384.jsonl", "0d71f1b7b519281c6d063c64083505b1c730d30edda7cbd86dc0124e7fb1caaf"),
    "schedule": ("/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/draw-schedule.json", "5649b14a58213ddf95d25b6068999a48fbcdb51f01acaf67303682b3fdfbc212"),
    "manifest": ("/mnt/e/sepalith/campaign-20260915/data-work/CPT-representative-full-doc-pilot-v1/selection-manifest.json", "202fbc8632415b156096f19180eb3922241c5d1617d723083473653af8583e41"),
    "producer_receipt": ("docs/campaign/receipts/SFT-11-CPT-representative-full-doc-pilot-preparation.json", "4b06cfc840826ca2fcec24c2f08546b18087dfb0bfdff2f0829737405be70ee2"),
    "provenance_review": ("docs/campaign/receipts/SFT-11-CPT-representative-root-provenance.json", "f7bb958cb0586f04d5310b356dba66722e8c9d4caaea76a4813efdacd21baadd"),
    "data_admission": ("docs/campaign/receipts/SFT-11-CPT-representative-diagnostic-data-admission.json", "12617952b47c6ca41a7e8fadf84825c840c826b98e192e099cb37633fb561794"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked(name: str, supplied: Path, root: Path) -> tuple[Path, str]:
    expected_path, expected_sha = EXPECTED[name]
    expected = Path(expected_path)
    if not expected.is_absolute():
        expected = root / expected
    supplied = supplied.resolve()
    require(supplied == expected.resolve(), f"{name} path differs")
    actual = sha256(supplied)
    if expected_sha is not None:
        require(actual == expected_sha, f"{name} bytes differ")
    return supplied, actual


def write_new(path: Path, value: dict) -> None:
    require(not path.exists(), "cohort binding output must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(*, rows: Path, schedule: Path, manifest: Path, producer_receipt: Path,
            provenance_review: Path, data_admission: Path, output: Path, repository_root: Path) -> dict:
    paths = {}
    for name, supplied in (("rows", rows), ("schedule", schedule), ("manifest", manifest),
                           ("producer_receipt", producer_receipt), ("provenance_review", provenance_review),
                           ("data_admission", data_admission)):
        path, digest = checked(name, supplied, repository_root)
        paths[name] = {"path": str(path), "sha256": digest}

    producer = json.loads(Path(paths["producer_receipt"]["path"]).read_text())
    provenance = json.loads(Path(paths["provenance_review"]["path"]).read_text())
    admission = json.loads(Path(paths["data_admission"]["path"]).read_text())
    selection = json.loads(Path(paths["manifest"]["path"]).read_text())
    schedule_value = json.loads(Path(paths["schedule"]["path"]).read_text())
    require(producer.get("status") == "verified_root_review_required", "producer packet status differs")
    require(provenance.get("status") == "pilot_split_and_base_dedup_verified", "root provenance review missing")
    require(provenance.get("nontrain_or_CPT_validation_groups") == 0, "non-TRAIN groups present")
    require(provenance.get("intersection_prior_base_or_validation_document_hashes") == 0, "prior/validation overlap present")
    require(provenance.get("intersection_protected_nontrain_hashes") == 0, "protected overlap present")
    require(admission.get("schema") == "sepalith.sft11.representative-diagnostic-data-admission.v1" and admission.get("status") == "admitted", "diagnostic data admission missing")
    require(admission.get("training_launch_authorized") is False and admission.get("parent_or_optimizer_selected") is False, "data admission exceeds data-only scope")
    require(selection.get("diagnostic_cohort_only") is True and selection.get("production_cap") is False, "cohort scope differs")

    materialized = []
    with Path(paths["rows"]["path"]).open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                materialized.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid row JSON at line {line_number}") from exc
    summary = validate_materialized_rows(materialized, max_sequence_tokens=16384, require_complete_documents=True)
    schedule_audit = validate_draw_schedule(schedule_value, summary["rows_checked"], token_rows_sha256=paths["rows"]["sha256"], max_steps=66, effective_batch=16)
    expected_counts = {"rows": 1046, "documents": 1024, "packages": 814, "input_tokens": 2578086, "payload_tokens": 2575972, "loss_tokens": 2576996}
    observed_counts = {"rows": summary["rows"], "documents": summary["documents"], "packages": len(summary["packages"]), "input_tokens": summary["input_tokens"], "payload_tokens": summary["payload_tokens"], "loss_tokens": summary["loss_tokens"]}
    require(observed_counts == expected_counts, "cohort denominators differ")
    require(schedule_audit["draws"] == 1056 and schedule_audit["unique_rows"] == 1046, "schedule coverage differs")
    replay_ids = schedule_value.get("replay_row_ids")
    require(isinstance(replay_ids, list) and len(replay_ids) == 10, "named replay count differs")
    require(schedule_value["row_ids"][-10:] == replay_ids, "named replay suffix differs")
    require(len(set(schedule_value["row_ids"][:1046])) == 1046, "unique rows are not all exposed before replay")

    cohort = {
        "id": "CPT-representative-full-doc-pilot-v1",
        "scope": "representative_complete_documents_diagnostic_not_all_eligible_corpus",
        "rows": paths["rows"], "draw_schedule": paths["schedule"], "manifest": paths["manifest"],
        "unique_rows": 1046, "documents": 1024, "packages": 814,
        "input_tokens": 2578086, "payload_tokens": 2575972, "loss_tokens": 2576996,
        "max_sequence_tokens": 16384, "named_replays": 10, "updates": 66,
    }
    result = {
        "schema": "sepalith.sft11.representative-cpt-cohort-binding-preparation.v1",
        "status": "cohort_verified_scientific_choices_unbound", "launch_authorized": False,
        "cohort": cohort,
        "data_admission": paths["data_admission"],
        "evidence": {"producer_receipt": paths["producer_receipt"], "root_provenance_review": paths["provenance_review"]},
        "validation": {"complete_documents": True, "target_truncation": False, "all_unique_rows_before_named_replay": True, "denominators": expected_counts, "draws": 1056, "updates": 66},
        "scientific_choices": {"parent": None, "learning_rate": None, "milestone_gates": None},
        "corpus_scope": {"diagnostic_cohort_only": True, "production_cap": False, "unselected_documents_queued": 105716},
    }
    write_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True); parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--provenance-review", type=Path, required=True); parser.add_argument("--data-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args(); result = prepare(**vars(args))
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
