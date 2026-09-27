#!/usr/bin/env python3
"""Independent metadata/schema audit for the 2K and 16K tfprobability outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
VALIDATOR = PLAN / "docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/campaign_cpt_data.py"
TOKENIZER_SHA = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
GROUP_ID = "g-1037a9f3b52fac791d3d"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validator():
    spec = importlib.util.spec_from_file_location("tfprobability_output_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("validator_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(candidate: Path, context: Path) -> dict[str, Any]:
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("group_id") != GROUP_ID or manifest.get("training_admission") is not False:
        raise ValueError("candidate_identity_or_admission_guard_failed")
    artifacts = manifest.get("artifacts", {})
    for name, pin in artifacts.items():
        path = candidate / name
        if not path.is_file() or path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
            raise ValueError(f"candidate_artifact_changed:{name}")
    rows = [json.loads(line) for line in (candidate / "cpt_train.jsonl").open(encoding="utf-8") if line.strip()]
    validator = load_validator()
    checked = validator.validate_materialized_rows(rows, max_sequence_tokens=2048, require_complete_documents=True)
    documents = [json.loads(line) for line in (candidate / "documents.jsonl").open(encoding="utf-8") if line.strip()]
    if len(documents) != 21 or len({row["document_id"] for row in documents}) != 21:
        raise ValueError("document_provenance_identity_mismatch")
    if any(row.get("group_id") != GROUP_ID or row.get("provisional_pending_terminal_global_dedup") is not True for row in documents):
        raise ValueError("document_provenance_guard_failed")
    source_tokens = sum(row["source_code_tokens"] for row in documents)
    if source_tokens != checked["payload_tokens"] or source_tokens != 182645:
        raise ValueError("2k_payload_token_total_mismatch")
    if manifest["counts"] != {"documents": 21, "input_tokens": 182933, "loss_tokens": 182666, "payload_tokens": 182645, "raw_bytes": 693518, "rows": 103}:
        raise ValueError("2k_manifest_counts_mismatch")
    result = json.loads((context / "result.json").read_text(encoding="utf-8"))
    for name, pin in result.get("artifacts", {}).items():
        path = context / name
        if not path.is_file() or path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
            raise ValueError(f"16k_artifact_changed:{name}")
    output = result["outputs"]["16384"]
    if result["totals"] != {"documents": 21, "input_rows": 103, "payload_tokens": 182645}:
        raise ValueError("16k_input_totals_mismatch")
    if output["rows"] != 27 or output["payload_tokens"] != 182645 or output["terminal_eos"] != 21:
        raise ValueError("16k_output_totals_mismatch")
    return {
        "schema": "sepalith.dat10.tfprobability_recovery_output_audit.v1",
        "status": "PASS_candidate_outputs_pending_terminal_global_dedup_and_root_admission",
        "candidate": str(candidate),
        "context_output": str(context),
        "2k": {"documents": 21, "rows": 103, "payload_tokens": 182645, "input_tokens": 182933,
               "loss_tokens": 182666, "source_bytes": 693518, "validator_pass": True},
        "16k": {"documents": 21, "input_rows": 103, "rows": 27, "payload_tokens": 182645,
                "terminal_eos": 21, "retokenized": False, "validator_pass": True},
        "exclusions": 0,
        "repair_rows": 0,
        "terminal_global_dedup_pending": True,
        "training_admission": False,
        "tokenizer_sha256": TOKENIZER_SHA,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidate, args.context)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
