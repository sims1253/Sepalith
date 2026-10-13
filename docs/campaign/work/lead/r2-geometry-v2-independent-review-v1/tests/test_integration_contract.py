#!/usr/bin/env python3
"""Bounded independent checks for geometry producer/audit consumer integration."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT.parents[4]
GEOMETRY_PACKET = PLAN / "docs/campaign/work/lead/r2-expanded-union-geometry-recovery-preparation-v2"
AUDIT_PACKET = PLAN / "docs/campaign/work/lead/r2-expanded-union-audit-preparation-v2"
ORIGINAL = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "RL11-15006-context-v1/context-sidecar.jsonl"
)
SEMANTIC_CONTEXT = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic10948-provider-materialization-v2/selected-01/selected-contexts.jsonl"
)
SEMANTIC_PROVENANCE = Path(
    "/mnt/e/sepalith/campaign-20260915/data-work/"
    "Semantic10948-provider-materialization-v2/final-01/candidate-provenance.jsonl"
)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


GEOMETRY = module("geometry_v2_review", GEOMETRY_PACKET / "recover_geometry_v2.py")
AUDIT_V1 = module("audit_v1_review", AUDIT_PACKET / "audit_expanded_union_v1.py")


def first(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.loads(next(stream))


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def consumer_probe(source_path: str, cursor: dict, start: dict, end: dict):
    geometry = {
        "schema": "sepalith.source-cursor-geometry.v1",
        "source_path": source_path,
        "source_sha256": "0" * 64,
        "preedit_sha256": "1" * 64,
        "cursor": cursor,
        "replacement_range": {"start": start, "end": end},
        "window_sha256": "2" * 64,
    }
    return AUDIT_V1.geometry_from_provenance(
        {
            "source_cursor_geometry": geometry,
            "source_cursor_geometry_sha256": digest(geometry),
            "source_identity": {},
        }
    )


class IntegrationContractTest(unittest.TestCase):
    def test_actual_original_identity_survives_producer_and_consumer(self) -> None:
        record = first(ORIGINAL)
        recovered = GEOMETRY.recover(
            record, record, "context", "nested_original15006", record["row_id"]
        )
        identity = record["source_identity"]["source_provenance"]
        geometry = recovered["source_cursor_geometry"]
        self.assertEqual(geometry["source_path"], identity["source_snapshot_path"])
        self.assertEqual(geometry["source_sha256"], identity["source_snapshot_sha256"])
        self.assertEqual(geometry["preedit_sha256"], identity["after_snapshot_sha256"])
        self.assertEqual(
            AUDIT_V1.geometry_from_provenance(recovered),
            (recovered["source_cursor_geometry_sha256"], None),
        )

    def test_actual_semantic_identity_survives_provider_join(self) -> None:
        record = first(SEMANTIC_CONTEXT)
        provenance = first(SEMANTIC_PROVENANCE)
        self.assertEqual(record["row_id"], provenance["row_id"])
        recovered = GEOMETRY.recover(
            record, provenance, "selected_context", "direct", record["row_id"]
        )
        identity = provenance["source_identity"]
        geometry = recovered["source_cursor_geometry"]
        self.assertEqual(geometry["source_path"], identity["source_path"])
        self.assertEqual(geometry["source_sha256"], identity["source_sha256"])
        self.assertEqual(geometry["preedit_sha256"], provenance["preedit_sha256"])
        self.assertEqual(
            AUDIT_V1.geometry_from_provenance(recovered),
            (recovered["source_cursor_geometry_sha256"], None),
        )

    def test_producer_rejects_path_and_range_tampering(self) -> None:
        record = first(SEMANTIC_CONTEXT)
        provenance = first(SEMANTIC_PROVENANCE)
        tampered = copy.deepcopy(provenance)
        tampered["source_identity"]["source_path"] = "relative.R"
        with self.assertRaisesRegex(GEOMETRY.RecoveryError, "direct_source_path"):
            GEOMETRY.recover(record, tampered, "selected_context", "direct", record["row_id"])
        tampered_record = copy.deepcopy(record)
        tampered_record["selected_context"]["replacement_range"]["end"]["character"] = 1
        with self.assertRaisesRegex(GEOMETRY.RecoveryError, "partial_nonempty_range_unverified"):
            GEOMETRY.recover(
                tampered_record,
                provenance,
                "selected_context",
                "direct",
                record["row_id"],
            )

    def test_audit_consumer_gap_relative_path_is_reproducible(self) -> None:
        # The producer rejects this. The current audit consumer accepts it when
        # source_identity is absent, so the integration is not independently
        # strict at the audit boundary.
        observed, reason = consumer_probe(
            "relative/file.R",
            {"line": 0, "character": 1},
            {"line": 0, "character": 0},
            {"line": 0, "character": 0},
        )
        self.assertIsNotNone(observed)
        self.assertIsNone(reason)

    def test_audit_consumer_gap_cursor_range_mismatch_is_reproducible(self) -> None:
        observed, reason = consumer_probe(
            "/absolute/file.R",
            {"line": 0, "character": 9},
            {"line": 0, "character": 1},
            {"line": 0, "character": 1},
        )
        self.assertIsNotNone(observed)
        self.assertIsNone(reason)

    def test_semantic_input_registry_filters_sidecar_to_candidates(self) -> None:
        registry = json.loads((GEOMETRY_PACKET / "input-registry.json").read_text())
        cohort = registry["cohorts"][1]
        self.assertEqual(cohort["candidate_ids"]["rows"], 10682)
        self.assertEqual(cohort["provenance"]["rows"], 10682)
        self.assertEqual(cohort["sidecar"]["rows"], 10948)
        self.assertEqual(cohort["context_key"], "selected_context")
        audit_spec = json.loads((AUDIT_PACKET / "union-spec.template.json").read_text())
        self.assertIsNone(audit_spec["cohorts"][2]["terminal_manifest"])


if __name__ == "__main__":
    unittest.main()
