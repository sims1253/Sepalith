#!/usr/bin/env python3
"""Synthetic fail-closed tests for the final CPT union builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


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
