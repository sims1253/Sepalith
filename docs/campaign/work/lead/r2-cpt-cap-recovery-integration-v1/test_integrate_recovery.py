#!/usr/bin/env python3
"""Synthetic tests for append-only recovery integration gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import integrate_recovery as integration


def token_hash(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def one_document() -> tuple[dict, dict]:
    source = [2, 3, 4]
    source_sha = "a" * 64
    document = {
        "document_id": source_sha, "source_sha256": source_sha, "source_sha1": "b" * 40,
        "git_blob_sha1": "c" * 40, "source_path": "/synthetic/example.R", "group_id": "g-test",
        "split": "train_group", "cpt_partition": "cpt_train", "chunks": 1,
        "document_token_count": len(source), "token_stream_sha256": token_hash(source),
    }
    row = {
        "schema": 1, "row_id": f"{source_sha}:0", "document_id": source_sha,
        "package": "synthetic", "group_id": "g-test", "cpt_partition": "cpt_train",
        "source_path": "/synthetic/example.R", "source_sha256": source_sha, "chunk_index": 0,
        "input_ids": [0, 2, 3, 4, 1], "labels": [-100, 2, 3, 4, 1],
        "attention_mask": [1, 1, 1, 1, 1], "source_token_start": 0, "source_token_end": 3,
        "overlap_context_tokens": 0, "is_document_end": True, "supervised_tokens": 4,
        "document_token_count": 3, "token_stream_sha256": token_hash(source),
        "tokenizer_revision": integration.TOKENIZER_REVISION,
    }
    return document, row


def test_eos_and_token_conservation() -> None:
    document, row = one_document()
    owned = integration.validate_chunk(row, document, True)
    assert owned == [2, 3, 4]


def test_alignment_replay_is_separate_and_minimal() -> None:
    document, row = one_document()
    result = {
        "unique_documents": [document], "unique_rows": [row],
        "frontier_by_path": {document["source_path"]: {"frontier_ordinal": 1}},
        "accounting": {"frontier_outcomes": [], "outcomes": {}},
        "integration_exclusions": [],
    }
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "checkpoint.json"
        checkpoint.write_text(json.dumps({"checkpoint_step": 250, "source_cursor": 177190, "checkpoint_id": "synthetic-250"}), encoding="utf-8")
        output = Path(temp) / "augmentation"
        built = integration.build_augmentation(result, output, checkpoint, 16)
        schedule = built["schedule"]
        assert schedule["unique_candidate_rows"] == 1
        assert schedule["alignment_replay_count"] == 15
        assert schedule["draw_count"] == 16
        assert schedule["existing_schedule_policy"].startswith("preserve unconsumed")
        assert len(integration.read_jsonl(output / "augmentation-cpt-train.jsonl")) == 1


def test_status_does_not_treat_missing_progress_as_complete() -> None:
    with tempfile.TemporaryDirectory() as temp:
        value = integration.live_status(Path(temp) / "recovery", Path(temp) / "root.terminal.json")
        assert value["status"] == "pending"


if __name__ == "__main__":
    test_eos_and_token_conservation()
    test_alignment_replay_is_separate_and_minimal()
    test_status_does_not_treat_missing_progress_as_complete()
    print("recovery integration synthetic tests: PASS")
