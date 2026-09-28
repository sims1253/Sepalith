#!/usr/bin/env python3
"""Bounded, read-only RUN-02 dependency and evidence audit.

The script hashes only the explicit RUN-01 source closure and the small
artifacts named by the RUN-01/RUN-04/RL-01 receipts.  It never walks a
workspace, loads a model, starts a server, or imports the extension.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
EXEC = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
EXT = EXEC / "extensions/vscode-sepalith"
RECEIPTS = PLAN / "docs/campaign/receipts"
OUT = PLAN / "docs/campaign/work/run02-acceptance-review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    record["bytes"] = path.stat().st_size
    record["sha256"] = sha256(path)
    checks: dict[str, bool] = {}
    if expected_sha is not None:
        checks["sha256"] = record["sha256"] == expected_sha
    if expected_bytes is not None:
        checks["bytes"] = record["bytes"] == expected_bytes
    if checks:
        record["matches"] = checks
    return record


def load(name: str) -> dict[str, Any]:
    return json.loads((RECEIPTS / name).read_text(encoding="utf-8"))


def main() -> None:
    snapshot = load("RUN-01-acceptance-source-snapshot.json")
    source_rows: list[dict[str, Any]] = []
    for entry in snapshot["files"]:
        current = EXT / entry["path"]
        copied = Path(snapshot["snapshot"]) / "source" / entry["path"]
        current_row = file_record(current, entry["sha256"], entry["bytes"])
        copied_row = file_record(copied, entry["sha256"], entry["bytes"])
        source_rows.append({
            "relative_path": entry["path"],
            "current": current_row,
            "snapshot_copy": copied_row,
            "current_equals_snapshot": (
                current_row.get("sha256") is not None
                and current_row.get("sha256") == copied_row.get("sha256")
            ),
        })

    closure = {
        "source_root": str(EXT),
        "source_git_head_from_receipt": snapshot["source_git_head"],
        "manifest_identity_sha256_from_receipt": snapshot["manifest_identity_sha256"],
        "manifest_json_sha256": sha256(Path(snapshot["snapshot"]) / "manifest.json"),
        "brief_source_documents": [
            file_record(PLAN / "docs/research/72h-prompt-training-contract.md"),
        ],
        "files": source_rows,
        "all_current_hashes_match_receipt": all(
            row["current"].get("matches", {}).get("sha256") is True for row in source_rows
        ),
        "all_snapshot_copies_match_receipt": all(
            row["snapshot_copy"].get("matches", {}).get("sha256") is True for row in source_rows
        ),
    }
    (OUT / "source-closure.json").write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")

    named_receipts = [
        "PRM-05-lead-v3-review.json",
        "PRM-04-live-cpu-parity.json",
        "RL-01-theta0-lead-review.json",
        "RUN-01-acceptance-source-snapshot.json",
        "RUN-01-acceptance-identity-lead-integration.json",
        "RUN-04-editor-g-full-acceptance-lead-review.json",
    ]
    receipt_rows = [file_record(RECEIPTS / name) for name in named_receipts]

    identity = load("RUN-01-acceptance-identity-lead-integration.json")
    identity_artifacts = []
    for value in identity["artifacts"].values():
        identity_artifacts.append(file_record(Path(value["path"]), value["sha256"], value["bytes"]))

    run04 = load("RUN-04-editor-g-full-acceptance-lead-review.json")
    run04_artifacts = []
    for value in run04["artifacts"]:
        path = Path(value["path"])
        if not path.is_absolute():
            path = PLAN / path
        run04_artifacts.append(file_record(path, value["sha256"]))

    prm05 = load("PRM-05-lead-v3-review.json")
    validation_log = Path(prm05["verification"]["validation_log"])
    validation_log_row = file_record(validation_log, prm05["verification"]["validation_log_sha256"])

    rl01 = load("RL-01-theta0-lead-review.json")
    gate_row = file_record(Path(rl01["gate_path"]), rl01["gate_sha256"])

    artifact_closure = {
        "receipt_files": receipt_rows,
        "run01_identity_artifacts": identity_artifacts,
        "run04_full_acceptance_artifacts": run04_artifacts,
        "prm05_validation_log": validation_log_row,
        "rl01_gate": gate_row,
        "model_weight_reads": 0,
        "server_launches": 0,
    }
    (OUT / "artifact-closure.json").write_text(json.dumps(artifact_closure, indent=2) + "\n", encoding="utf-8")

    tests = json.loads((OUT / "targeted-tests.json").read_text(encoding="utf-8"))
    test_rows = [
        {
            "command": row["command"],
            "returncode": row["returncode"],
            "seconds": row["seconds"],
            "output_tail": row["output"][-500:],
        }
        for row in tests["tests"]
    ]
    all_tests_pass = all(row["returncode"] == 0 for row in tests["tests"])

    report = {
        "task": "RUN-02",
        "status": "prepared_partial_live_gates_pending",
        "scope": "CPU-only source/evidence audit; no model, server, GPU, SSH, network, or CLI launch",
        "source_closure": {
            "path": str(OUT / "source-closure.json"),
            "files": len(source_rows),
            "all_current_hashes_match_receipt": closure["all_current_hashes_match_receipt"],
            "all_snapshot_copies_match_receipt": closure["all_snapshot_copies_match_receipt"],
        },
        "targeted_tests": {"path": str(OUT / "targeted-tests.json"), "count": len(test_rows), "all_pass": all_tests_pass, "tests": test_rows},
        "criteria": {
            "serving_training_fixture_ids_agree": {
                "status": "partial",
                "evidence": [
                    "RUN-01 source snapshot closes 17 extension files; current and copied hashes match.",
                    "campaign-protocol: 14 assertions; PRM-05: 38 assertions; context: 46 checks; history: 51 checks.",
                    "RL-01 reports 20 native prompt identities and 6 source-authoritative TRAIN rows.",
                ],
                "gap": "Named receipts do not publish one shared ID-to-prompt-SHA manifest covering the serving fixtures and the training rows. The six RL-01 TRAIN rows are source-authoritative, but do not establish full-panel ID equality.",
            },
            "stale_or_out_of_date_responses_rejected": {
                "status": "pass_mechanical_pending_live_stale_ghost",
                "evidence": [
                    "campaign_protocol.ts:478-485 checks URI, document version, and pre-edit content SHA; 497-510 gates application.",
                    "next_edit.ts:23-59 requires the current request lease after awaited work.",
                    "extension.ts:833-840 and 999-1004 recheck lease, runtime generation, restart gate, cancellation, URI/version/content SHA before publication.",
                    "history_provider.ts:478-507 rejects stale version, equal-version desync, and replay mismatch.",
                    "provider lifecycle and acceptance identity scripts pass; RUN-04 exercises one real full inline acceptance.",
                ],
                "gap": "RUN-04 has one real full inline acceptance; partial acceptance, stale ghost nonpublication, and a real out-of-date response from a live sidecar remain unobserved.",
            },
            "cache_reuse_preserves_correctness_and_checkpoint_freshness": {
                "status": "partial",
                "evidence": [
                    "campaign_client.ts:157-175 sends the primary integer prompt with cache_prompt=true and no text stop list; protocol EOS/terminal validation follows.",
                    "extension.ts:849-858 keys primary reuse by rendered prompt, replacement identity, and cursor; 710-718 clears cached items/keys and in-flight work on invalidation.",
                    "extension.ts:835-840, 999-1004 and 269-299 invalidate on document/runtime/config generations; provider lifecycle tests cover same-key reuse, stop/start invalidation, and late transport results.",
                    "runtime.ts:400-455 validates cached manifests on every use, requires explicit refresh for invalid cache, and partitions installed assets by selected-bundle identity.",
                    "targeted runtime, profile, protocol, request, provider, context and history tests pass (10 commands).",
                ],
                "gap": "No named evidence runs an old-checkpoint response against a newly loaded checkpoint and proves the server cache cannot return old state. Source-side generation and identity guards are present; a tagged old/new sidecar response test remains the smallest live closure test.",
            },
        },
        "receipt_interpretation": {
            "PRM-04": "transport/token/EOG evidence on MiniCPM midtrain with n_ctx=1024; it does not prove the current primary 4096 profile or fixture equality.",
            "PRM-05": "mocked provider integration and protocol evidence; its historical pre-RUN-01 source hashes are retained as evidence, not treated as the current closure.",
            "RL-01": "theta0 gate and source-bound prompt identity evidence; it is not a full serving/training fixture manifest.",
            "RUN-01/RUN-04": "current 17-file source closure and one real full inline acceptance; partial/stale editor paths remain open.",
        },
        "smallest_remaining_test": "Using the accepted source snapshot, run one controlled response tagged with old runtime/checkpoint generation, invalidate/restart the managed sidecar, then assert the late old response is rejected and cannot populate lastItems. Publish the exact serving/training fixture ID+prompt-SHA manifest alongside the same run so full fixture-ID equality can be checked.",
    }
    (OUT / "audit-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_files": len(source_rows), "tests": len(test_rows), "all_tests_pass": all_tests_pass}, indent=2))


if __name__ == "__main__":
    main()
