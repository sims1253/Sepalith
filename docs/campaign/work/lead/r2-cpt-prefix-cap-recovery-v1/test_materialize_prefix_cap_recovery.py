#!/usr/bin/env python3
"""Synthetic contract tests for the pre-cap recovery worker.

These tests never open campaign source or terminal payloads.  They exercise
the fail-closed split/licence path and prove that the old 4 MiB policy is not
present in the recovery worker.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import materialize_prefix_cap_recovery as recovery


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(ids=[2, 3, 4])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.text


class FakeContract:
    @staticmethod
    def chunks(ids: list[int], size: int):
        yield {
            "input_ids": [0, *ids, 1],
            "labels": [-100, *ids, 1],
            "attention_mask": [1] * (len(ids) + 2),
            "source_token_start": 0,
            "source_token_end": len(ids),
            "overlap_context_tokens": 0,
            "is_document_end": True,
            "supervised_tokens": len(ids) + 1,
        }


def row(path: Path, description: Path, group: str = "g-synthetic") -> dict:
    raw = path.read_bytes()
    description_raw = description.read_bytes()
    return {
        "bytes": len(raw),
        "cpt_partition": "cpt_train",
        "description_path": str(description),
        "description_sha256": hashlib.sha256(description_raw).hexdigest(),
        "group_id": group,
        "license": "MIT",
        "package": "synthetic",
        "path": str(path),
        "reason": "whole_document_exceeds_remaining_group_cap",
        "seeded_index": 1,
        "split": "train_group",
        "version": "0.0.0",
    }


def guards() -> dict:
    return {
        "protected_parent_hashes": set(),
        "terminal_paths": set(),
        "terminal_source_sha256": set(),
        "reserved_paths": set(),
        "reserved_source_sha256": set(),
        "reserved_source_sha1": set(),
        "reserved_git_blob_sha1": set(),
    }


def run_one(source_size: int) -> tuple[dict, Path]:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "R" / "example.R"
        source.parent.mkdir()
        source.write_bytes(b"x" * source_size)
        desc = root / "DESCRIPTION"
        desc.write_text("Package: synthetic\nVersion: 0.0.0\nLicense: MIT\n", encoding="utf-8")
        args = SimpleNamespace(output=root / "out", frontier=Path("synthetic-frontier"))
        fake = FakeTokenizer()
        fake.text = "x" * source_size
        runner = recovery.RecoveryRun(args, [row(source, desc)], guards(), FakeContract)
        runner._tokenizer = fake
        result = runner.process_group("g-synthetic", [row(source, desc)])
        docs = list(recovery.read_jsonl(args.output / "groups" / "000001-g-synthetic" / "documents.jsonl"))
        assert result["documents"] == 1
        assert docs[0]["source_bytes"] == source_size
        assert docs[0]["oversized_source"] is (source_size > 4 * 1024 * 1024)
        # Copy the small result out before TemporaryDirectory cleanup.
        copied = root / "result.json"
        copied.write_text(json.dumps(docs[0]), encoding="utf-8")
        return docs[0], copied


def test_large_source_is_retained() -> None:
    # The old materializer rejected this size.  Recovery retains it and marks
    # the measurement for downstream context/length review.
    document, _ = run_one(4 * 1024 * 1024 + 1)
    assert document["oversized_source"] is True
    assert document["chunks"] == 1


def test_empty_source_is_named_exclusion() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "example.R"
        source.write_bytes(b"")
        desc = root / "DESCRIPTION"
        desc.write_text("Package: synthetic\nVersion: 0.0.0\nLicense: MIT\n", encoding="utf-8")
        args = SimpleNamespace(output=root / "out", frontier=Path("synthetic-frontier"))
        runner = recovery.RecoveryRun(args, [row(source, desc)], guards(), FakeContract)
        result = runner.process_group("g-synthetic", [row(source, desc)])
        assert result.get("documents", 0) == 0
        exclusion = root / "out" / "groups" / "000001-g-synthetic" / "exclusions.jsonl"
        assert json.loads(exclusion.read_text(encoding="utf-8"))["reason"] == "empty_source_no_payload"


def test_license_restriction_is_named_exclusion() -> None:
    row_value = {"license": "MIT"}
    description = {"fields": {"License": "MIT", "License_restricts_use": "yes"}}
    assert recovery.license_reason(row_value, description) == "license_requires_review_restricts_use"


if __name__ == "__main__":
    test_large_source_is_retained()
    test_empty_source_is_named_exclusion()
    test_license_restriction_is_named_exclusion()
    print("prefix-cap recovery synthetic tests: PASS")
