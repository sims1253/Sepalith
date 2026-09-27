#!/usr/bin/env python3
"""Synthetic fail-closed tests for the final CPT union builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("final_union", HERE / "build_final_union.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
VALIDATOR = MODULE.load_validator()


def row(source: str = "a" * 64, chunk: int = 0, start: int = 0, end: int = 2,
        final: bool = True, document_count: int | None = None) -> dict:
    ids = [0, 22, 23, 1]
    labels = [-100, 22, 23, 1 if final else -100]
    return {
        "schema": 1,
        "row_id": f"{source}:{chunk}",
        "document_id": source,
        "package": "fixture",
        "group_id": "g-fixture",
        "cpt_partition": "cpt_train",
        "source_path": "/fixture/R/example.R",
        "source_sha256": source,
        "chunk_index": chunk,
        "input_ids": ids,
        "labels": labels,
        "attention_mask": [1] * len(ids),
        "source_token_start": start,
        "source_token_end": end,
        "token_start": start,
        "token_end": end,
        "document_token_count": document_count if document_count is not None else (2 if final else 4),
        "overlap_context_tokens": 0,
        "is_document_end": final,
        "supervised_tokens": sum(value != -100 for value in labels),
    }


class FinalUnionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = {"g-fixture": "train_group"}
        self.partition = {"g-fixture": "cpt_train"}

    def test_valid_row_and_native_protocol(self) -> None:
        value = MODULE.validate_row(row(), VALIDATOR, self.split, self.partition, set(), set())
        self.assertEqual(value, ("a" * 64, "/fixture/R/example.R"))

    def test_rejects_nontrain_and_bad_token(self) -> None:
        with self.assertRaises(MODULE.UnionError):
            MODULE.validate_row({**row(), "group_id": "g-dev"}, VALIDATOR,
                                {"g-dev": "dev_group"}, {"g-dev": "cpt_train"}, set(), set())
        bad = {**row(), "input_ids": [0, 130560, 23, 1], "labels": [-100, 130560, 23, 1]}
        with self.assertRaises(MODULE.UnionError):
            MODULE.validate_row(bad, VALIDATOR, self.split, self.partition, set(), set())

    def test_builder_keeps_chunks_and_drops_duplicate_document_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = MODULE.UnionBuilder(Path(directory), "synthetic", self.split, self.partition,
                                          set(), set(), {}, VALIDATOR)
            first = row(final=False, end=2)
            second = row(chunk=1, start=2, end=4, final=True, document_count=4)
            builder.stream(MODULE.Source("fixture:first", Path("fixture.jsonl")), [first, second])
            builder.stream(MODULE.Source("fixture:duplicate", Path("duplicate.jsonl")),
                           [row(chunk=2, start=4, end=6, final=True, document_count=6)])
            artifacts = builder.close()
            self.assertEqual(builder.emitted_rows, 2)
            self.assertEqual(builder.emitted_documents, 1)
            self.assertEqual(builder.exclusions["duplicate_source_document"], 1)
            self.assertEqual(artifacts["cpt_train.jsonl"]["bytes"] > 0, True)

    def test_duplicate_row_id_requires_canonical_content_equality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = MODULE.UnionBuilder(Path(directory), "synthetic", self.split, self.partition,
                                          set(), set(), {}, VALIDATOR)
            original = row()
            builder.stream(MODULE.Source("fixture", Path("fixture.jsonl")), [original, dict(original)])
            self.assertEqual(builder.exclusions["duplicate_row_id_exact"], 1)
            changed = {**original, "package": "contradictory-fixture"}
            with self.assertRaisesRegex(MODULE.UnionError, "duplicate_row_id_content_conflict"):
                builder.stream(MODULE.Source("fixture", Path("fixture.jsonl")), [changed])
            builder.close()

    def test_main_group_directory_is_indexed_once_and_missing_or_duplicate_rejected(self) -> None:
        old = (MODULE.MAIN_START, MODULE.MAIN_END, MODULE.EXPECTED_GROUPS)
        MODULE.MAIN_START, MODULE.MAIN_END, MODULE.EXPECTED_GROUPS = 1, 4, 3
        try:
            with tempfile.TemporaryDirectory() as directory:
                groups = Path(directory)
                for index in range(1, 4): (groups / f"{index:06d}-g{index}").mkdir()
                with mock.patch.object(MODULE.os, "scandir", wraps=MODULE.os.scandir) as scan:
                    indexed = MODULE.index_main_group_directories(groups)
                self.assertEqual(set(indexed), {1, 2, 3}); self.assertEqual(scan.call_count, 1)
                (groups / "000002-duplicate").mkdir()
                with self.assertRaisesRegex(MODULE.UnionError, "ambiguous"):
                    MODULE.index_main_group_directories(groups)
            with tempfile.TemporaryDirectory() as directory:
                groups = Path(directory); (groups / "000001-a").mkdir(); (groups / "000003-c").mkdir()
                with self.assertRaisesRegex(MODULE.UnionError, "missing:2"):
                    MODULE.index_main_group_directories(groups)
        finally:
            MODULE.MAIN_START, MODULE.MAIN_END, MODULE.EXPECTED_GROUPS = old

    def test_base_source_requires_reviewed_payload_hash_size_and_row_pins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);payload=d/"cpt_train.jsonl";payload.write_text("{}\n")
            digest="a"*64;manifest=d/"manifest.json";manifest.write_text(json.dumps({"artifacts":{"cpt_train.jsonl":{"sha256":digest,"bytes":3}},"counts":{"cpt_train":{"rows":1}}}))
            source=MODULE.reviewed_base_source("base",payload,manifest,digest,3,1);self.assertEqual(source.expected_sha256,digest)
            manifest.write_text(json.dumps({"artifacts":{"cpt_train.jsonl":{"bytes":3}},"counts":{"cpt_train":{"rows":1}}}))
            with self.assertRaisesRegex(MODULE.UnionError,"base_manifest_binding_invalid"):
                MODULE.reviewed_base_source("base",payload,manifest,digest,3,1)

    def test_profile_validation_requires_manifest_pin_counts_and_complete_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);docs=d/"documents.jsonl";digest="a"*64;record={"cpt_partition":"cpt_validation","sha256":digest,"document_id":digest,"package":"p","group_id":"g","path":"/R/a.R","chunks":1,"source_code_tokens":3};docs.write_text(json.dumps(record)+"\n")
            manifest=d/"manifest.json"
            def write_manifest(rows=370,sha=MODULE.PINS["profile_validation_documents"][1]):manifest.write_text(json.dumps({"schema":1,"status":"CPU_materialized_candidate_not_training_admission","tokenizer_sha256":MODULE.PINS["tokenizer"][1],"partition_sha256":MODULE.PINS["cpt_partition"][1],"parent_hash_guard_sha256":MODULE.PINS["protected_parent_hashes"][1],"artifacts":{"documents.jsonl":{"sha256":sha,"bytes":docs.stat().st_size}},"counts":{"cpt_validation":{"documents":1,"rows":rows}}}))
            write_manifest();reserved,_,count=MODULE.profile_validation_reservations(docs,manifest,1);self.assertEqual((reserved,count),({digest},1))
            write_manifest(rows=369)
            with self.assertRaisesRegex(MODULE.UnionError,"manifest_count_invalid"):MODULE.profile_validation_reservations(docs,manifest,1)
            bad={**record};bad.pop("package");docs.write_text(json.dumps(bad)+"\n");write_manifest()
            with self.assertRaisesRegex(MODULE.UnionError,"identity_metadata_invalid"):MODULE.profile_validation_reservations(docs,manifest,1)

    def test_verify_pins_rejects_changed_profile_hash_without_bulk_source_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"profile.jsonl";p.write_text("metadata\n")
            with mock.patch.object(MODULE,"PINS",{"profile_validation_documents":(p,"0"*64)}):
                with self.assertRaisesRegex(MODULE.UnionError,"pinned_file_changed:profile_validation_documents"):MODULE.verify_pins()

    def test_terminal_gate_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory)
            (main / "progress.json").write_text(json.dumps({
                "groups_committed": MODULE.EXPECTED_GROUPS - 1,
                "groups_remaining": 1,
                "first_uncommitted_seeded_index": MODULE.MAIN_END - 1,
                "status": "in_progress",
            }))
            with self.assertRaises(MODULE.UnionError):
                MODULE.main_terminal_gate(main)


if __name__ == "__main__":
    unittest.main()
