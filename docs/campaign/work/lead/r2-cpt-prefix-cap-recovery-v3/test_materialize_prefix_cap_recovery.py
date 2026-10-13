#!/usr/bin/env python3
"""Synthetic tests for the v3 no-cap recovery and resume contract."""

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


def make_row(path: Path, description: Path, *, ordinal: int = 1, group: str = "g-synthetic") -> dict:
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
        "scope": "prefix_global_cap",
        "frontier_ordinal": ordinal,
        "seeded_index": ordinal,
        "split": "train_group",
        "version": "0.0.0",
        "source_sha256": None,
    }


def fake_guards() -> dict:
    input_pins = {
        name: {"path": f"/synthetic/{name}", "bytes": 1, "sha256": "a" * 64}
        for name in ("frontier", "global_split", "cpt_partition", "protected_parent_hashes", "profile_documents", "terminal_document_provenance")
    }
    return {
        "input_pins": input_pins,
        "frontier_scope_counts": {"prefix_global_cap": 1},
        "protected_parent_hashes": set(),
        "terminal_paths": set(),
        "terminal_source_sha256": set(),
        "reserved_paths": set(),
        "reserved_source_sha256": set(),
        "reserved_source_sha1": set(),
        "reserved_git_blob_sha1": set(),
    }


def prepare_runner(root: Path, source_size: int = 3) -> tuple[recovery.RecoveryRun, dict, Path]:
    source = root / "R" / "example.R"
    source.parent.mkdir()
    source.write_bytes(b"x" * source_size)
    description = root / "DESCRIPTION"
    description.write_text("Package: synthetic\nVersion: 0.0.0\nLicense: MIT\n", encoding="utf-8")
    source_row = make_row(source, description)
    args = SimpleNamespace(output=root / "out", frontier=root / "frontier.jsonl")
    runner = recovery.RecoveryRun(args, [source_row], fake_guards(), FakeContract)
    fake = FakeTokenizer()
    fake.text = "x" * source_size
    runner._tokenizer = fake
    return runner, source_row, args.output


def test_large_source_is_retained() -> None:
    with tempfile.TemporaryDirectory() as temp:
        runner, source_row, output = prepare_runner(Path(temp), 4 * 1024 * 1024 + 1)
        result = runner.process_group("g-synthetic", [source_row])
        document = recovery.read_jsonl(output / "groups" / "000001-g-synthetic" / "documents.jsonl")[0]
        assert result["documents"] == 1
        assert document["source_bytes"] == 4 * 1024 * 1024 + 1
        assert document["oversized_source"] is True
        assert document["omission_scope"] == "prefix_global_cap"


def test_empty_source_is_named_exclusion() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runner, source_row, output = prepare_runner(root, 0)
        result = runner.process_group("g-synthetic", [source_row])
        assert result.get("documents", 0) == 0
        exclusion = output / "groups" / "000001-g-synthetic" / "exclusions.jsonl"
        assert json.loads(exclusion.read_text(encoding="utf-8"))["reason"] == "empty_source_no_payload"
        resumed = recovery.RecoveryRun(SimpleNamespace(output=output, frontier=root / "frontier.jsonl"), [source_row], fake_guards(), FakeContract)
        assert "g-synthetic" in resumed.completed_groups


def test_positive_resume_duplicate_only_group() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runner, source_row, output = prepare_runner(root)
        source_sha = hashlib.sha256(Path(source_row["path"]).read_bytes()).hexdigest()
        runner.guards["terminal_source_sha256"] = {source_sha}
        runner.process_group("g-synthetic", [source_row])
        resumed = recovery.RecoveryRun(SimpleNamespace(output=output, frontier=root / "frontier.jsonl"), [source_row], runner.guards, FakeContract)
        assert "g-synthetic" in resumed.completed_groups


def test_positive_resume_repair_only_group() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runner, source_row, output = prepare_runner(root)
        source_row["description_path"] = str(root / "missing-DESCRIPTION")
        runner.process_group("g-synthetic", [source_row])
        resumed = recovery.RecoveryRun(SimpleNamespace(output=output, frontier=root / "frontier.jsonl"), [source_row], runner.guards, FakeContract)
        assert "g-synthetic" in resumed.completed_groups


def test_positive_resume_emitted_group() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runner, source_row, output = prepare_runner(root)
        runner.process_group("g-synthetic", [source_row])
        resumed = recovery.RecoveryRun(SimpleNamespace(output=output, frontier=root / "frontier.jsonl"), [source_row], runner.guards, FakeContract)
        assert "g-synthetic" in resumed.completed_groups


def test_license_restriction_is_named_exclusion() -> None:
    assert recovery.license_reason(
        {"license": "MIT"},
        {"fields": {"License": "MIT", "License_restricts_use": "yes"}},
    ) == "license_requires_review_restricts_use"


def test_resume_rejects_tampered_payload() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runner, source_row, output = prepare_runner(root)
        runner.process_group("g-synthetic", [source_row])
        payload = output / "groups" / "000001-g-synthetic" / "cpt_train.jsonl"
        payload.write_text(payload.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        try:
            recovery.RecoveryRun(SimpleNamespace(output=output, frontier=root / "frontier.jsonl"), [source_row], fake_guards(), FakeContract)
        except recovery.RecoveryError as exc:
            assert "resume_artifact_" in str(exc)
        else:
            raise AssertionError("tampered completed group was accepted")


def test_alternate_frontier_identity_is_rejected() -> None:
    original = Path("docs/campaign/work/lead/r2-cpt-prefix-cap-recovery-v3/combined-frontier.jsonl")
    with tempfile.TemporaryDirectory() as temp:
        alternate = Path(temp) / "alternate.jsonl"
        alternate.write_bytes(original.read_bytes() + b"\n")
        rows = recovery.read_jsonl(alternate)
        try:
            recovery.load_guards(rows, alternate)
        except recovery.RecoveryError as exc:
            assert "frontier" in str(exc)
        else:
            raise AssertionError("alternate frontier identity was accepted")


if __name__ == "__main__":
    test_large_source_is_retained()
    test_empty_source_is_named_exclusion()
    test_positive_resume_duplicate_only_group()
    test_positive_resume_repair_only_group()
    test_positive_resume_emitted_group()
    test_license_restriction_is_named_exclusion()
    test_resume_rejects_tampered_payload()
    test_alternate_frontier_identity_is_rejected()
    print("prefix-cap recovery v3 synthetic tests: PASS")
