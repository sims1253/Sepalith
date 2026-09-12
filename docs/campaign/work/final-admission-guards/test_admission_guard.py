#!/usr/bin/env python3
"""Synthetic-only tests for the DAT-08 raw-source admission boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admission_guard as guard


OBSERVED = "2026-09-14T10:02:00Z"
WEIGHTS_SHA = hashlib.sha256(b"pinned-weights").hexdigest()
HARNESS_SHA = hashlib.sha256(b"pinned-harness").hexdigest()
SOURCE_LOCK_SHA = hashlib.sha256(b"pinned-source-lock").hexdigest()
CONSTRUCTOR_SHA = hashlib.sha256(b"pinned-constructor").hexdigest()


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def frozen_receipt() -> dict[str, object]:
    return {
        "status": "frozen",
        "weights_frozen": True,
        "harness_frozen": True,
        "final_access_unlocked": True,
        "weights_sha256": WEIGHTS_SHA,
        "harness_sha256": HARNESS_SHA,
        "weights_frozen_at": "2026-09-14T10:01:00Z",
        "harness_frozen_at": "2026-09-14T10:01:00Z",
    }


def case_mapping(root: Path, *, row_id: str = "final-row-1", content: bytes = b"synthetic R\n") -> dict[str, object]:
    source = root / "fixture.R"
    source.write_bytes(content)
    return {
        "row_id": row_id,
        "split_identity": f"final-identity-{row_id}",
        "package_id": f"package-{row_id}",
        "repository_id": "synthetic-repository",
        "group_id": f"group-{row_id}",
        "family": "rename_propagation",
        "source_path": str(source),
        "source_sha256": digest(content),
        "context_sha256": digest(f"context-{row_id}"),
        "target_sha256": digest(f"target-{row_id}"),
    }


def semantic_receipt(cases: list[dict[str, object]]) -> dict[str, object]:
    bindings = {
        str(case["row_id"]): {
            "split_identity": case["split_identity"],
            "source_sha256": case["source_sha256"],
            "context_sha256": case["context_sha256"],
            "target_sha256": case["target_sha256"],
        }
        for case in cases
    }
    return {
        "schema": "dat08.semantic-source-case-receipt.v1",
        "status": "source_cases_verified",
        "split": guard.SEMANTIC_RECEIPT_SPLIT,
        "source_lock_sha256": SOURCE_LOCK_SHA,
        "content_access": "post_freeze_source_bytes",
        "verified_case_ids": sorted(bindings),
        "constructor_sha256": CONSTRUCTOR_SHA,
        "case_bindings": bindings,
    }


def make_guard(root: Path, cases: list[dict[str, object]], **changes: object) -> guard.AdmissionGuard:
    values: dict[str, object] = {
        "allowed_source_roots": [root],
        "train_identities": {
            "row_ids": ["train-row"],
            "package_ids": ["train-package"],
            "group_ids": ["train-group"],
            "split_identities": ["train-identity"],
        },
        "dev_identities": {
            "row_ids": ["dev-row"],
            "package_ids": ["dev-package"],
            "group_ids": ["dev-group"],
            "split_identities": ["dev-identity"],
        },
        "freeze_receipt": frozen_receipt(),
        "semantic_receipt": semantic_receipt(cases),
        "expected_weights_sha256": WEIGHTS_SHA,
        "expected_harness_sha256": HARNESS_SHA,
        "expected_source_lock_sha256": SOURCE_LOCK_SHA,
        "observed_at": OBSERVED,
    }
    values.update(changes)
    return guard.AdmissionGuard(**values)  # type: ignore[arg-type]


class AdmissionGuardTests(unittest.TestCase):
    def test_valid_admission_handoffs_bytes_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            output = root / "admission.json"
            reads: list[str] = []
            result = make_guard(root, cases).admit(
                cases,
                output_path=output,
                before_source_read=lambda case: reads.append(case.row_id),
            )
            self.assertEqual(reads, ["final-row-1"])
            self.assertEqual(result.source_bytes["final-row-1"], b"synthetic R\n")
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "source_admission_verified")
            self.assertTrue(payload["source_read_policy"]["preflight_before_bytes"])
            self.assertFalse(payload["final_content_opened"])
            self.assertEqual(payload["source_records"][0]["context_sha256"], cases[0]["context_sha256"])
            self.assertEqual(
                payload["manifest_body_sha256"],
                guard.sha256_json({key: value for key, value in payload.items() if key != "manifest_body_sha256"}),
            )

    def test_freeze_time_and_explicit_authorization_precede_source_reads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            reads: list[str] = []
            pending = make_guard(root, cases, observed_at="2026-09-14T09:59:59Z")
            with self.assertRaisesRegex(guard.AdmissionGuardError, "freeze_time_not_reached"):
                pending.admit(
                    cases,
                    output_path=root / "pending.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])
            bad_receipt = frozen_receipt()
            bad_receipt["final_access_unlocked"] = False
            unauthorized = make_guard(root, cases, freeze_receipt=bad_receipt)
            with self.assertRaisesRegex(guard.AdmissionGuardError, "freeze_authorization_missing"):
                unauthorized.admit(
                    cases,
                    output_path=root / "unauthorized.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

    def test_positive_allowlist_rejects_outside_and_symlink_before_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary) / "allowed"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            outside_source = outside / "outside.R"
            outside_source.write_bytes(b"outside\n")
            escaped = case_mapping(root)
            escaped["source_path"] = str(outside_source)
            escaped["source_sha256"] = digest(b"outside\n")
            reads: list[str] = []
            with self.assertRaisesRegex(guard.AdmissionGuardError, "source_outside_positive_allowlist"):
                make_guard(root, [escaped]).admit(
                    [escaped],
                    output_path=root / "outside.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

            link = root / "link.R"
            link.symlink_to(outside_source)
            linked = case_mapping(root, row_id="linked")
            linked["source_path"] = str(link)
            linked["source_sha256"] = digest(b"outside\n")
            linked_semantic = semantic_receipt([linked])
            with self.assertRaisesRegex(guard.AdmissionGuardError, "source_symlink_component"):
                make_guard(root, [linked], semantic_receipt=linked_semantic).admit(
                    [linked],
                    output_path=root / "link.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

    def test_split_overlap_and_duplicate_final_identities_reject_before_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            overlapping = case_mapping(root)
            overlapping["package_id"] = "train-package"
            overlap_semantic = semantic_receipt([overlapping])
            reads: list[str] = []
            with self.assertRaisesRegex(guard.AdmissionGuardError, "case_split_identity_overlap"):
                make_guard(root, [overlapping], semantic_receipt=overlap_semantic).admit(
                    [overlapping],
                    output_path=root / "overlap.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

            first = case_mapping(root, row_id="same-1")
            second = case_mapping(root, row_id="same-2")
            second["split_identity"] = first["split_identity"]
            duplicate_semantic = semantic_receipt([first, second])
            with self.assertRaisesRegex(guard.AdmissionGuardError, "case_split_identity_duplicate"):
                make_guard(root, [first, second], semantic_receipt=duplicate_semantic).admit(
                    [first, second],
                    output_path=root / "duplicate.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

    def test_semantic_binding_is_exact_and_stale_receipt_reads_zero_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            stale = semantic_receipt(cases)
            stale["case_bindings"] = copy.deepcopy(stale["case_bindings"])
            stale["case_bindings"]["final-row-1"]["target_sha256"] = digest("forged-target")
            reads: list[str] = []
            with self.assertRaisesRegex(guard.AdmissionGuardError, "semantic_binding_mismatch"):
                make_guard(root, cases, semantic_receipt=stale).admit(
                    cases,
                    output_path=root / "stale.json",
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])

    def test_replaced_source_after_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            source = Path(str(cases[0]["source_path"]))
            replacement = root / "replacement.R"
            replacement.write_bytes(b"replacement\n")
            reads: list[str] = []

            def replace_after_gate(case: guard.SourceCaseSpec) -> None:
                reads.append(case.row_id)
                replacement.replace(source)

            with self.assertRaisesRegex(guard.AdmissionGuardError, "source_replaced_after_gate"):
                make_guard(root, cases).admit(
                    cases,
                    output_path=root / "replaced.json",
                    before_source_read=replace_after_gate,
                )
            self.assertEqual(reads, ["final-row-1"])
            self.assertFalse((root / "replaced.json").exists())

    def test_existing_output_is_preserved_and_blocks_source_reads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            output = root / "immutable.json"
            output.write_text("sentinel\n", encoding="utf-8")
            reads: list[str] = []
            with self.assertRaisesRegex(guard.AdmissionGuardError, "output_exists"):
                make_guard(root, cases).admit(
                    cases,
                    output_path=output,
                    before_source_read=lambda case: reads.append(case.row_id),
                )
            self.assertEqual(reads, [])
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_cli_uses_same_synthetic_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dat08-guard-") as temporary:
            root = Path(temporary)
            cases = [case_mapping(root)]
            metadata = root / "metadata"
            metadata.mkdir()
            cases_path = metadata / "cases.json"
            freeze_path = metadata / "freeze.json"
            semantic_path = metadata / "semantic.json"
            train_path = metadata / "train.json"
            dev_path = metadata / "dev.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            freeze_path.write_text(json.dumps(frozen_receipt()), encoding="utf-8")
            semantic_path.write_text(json.dumps(semantic_receipt(cases)), encoding="utf-8")
            train_path.write_text(json.dumps({"row_ids": ["train-row"]}), encoding="utf-8")
            dev_path.write_text(json.dumps({"row_ids": ["dev-row"]}), encoding="utf-8")
            output = root / "cli-output.json"
            status = guard.main(
                [
                    "--cases",
                    str(cases_path),
                    "--freeze-receipt",
                    str(freeze_path),
                    "--semantic-receipt",
                    str(semantic_path),
                    "--train-identities",
                    str(train_path),
                    "--dev-identities",
                    str(dev_path),
                    "--allowed-root",
                    str(root),
                    "--weights-sha256",
                    WEIGHTS_SHA,
                    "--harness-sha256",
                    HARNESS_SHA,
                    "--source-lock-sha256",
                    SOURCE_LOCK_SHA,
                    "--observed-at",
                    OBSERVED,
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "source_admission_verified")


if __name__ == "__main__":
    unittest.main()
