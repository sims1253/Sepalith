#!/usr/bin/env python3
"""Small deterministic and fail-closed tests for the pilot preparation helper."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pilot", HERE / "prepare_representative_pilot.py")
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_rank_is_stable_and_ascii_bound() -> None:
    document_id = "12fc2cd0d94fed6ee2e1a86108c3daadf44dda850afd675cbd8c9844517f9a21"
    expected = hashlib.sha256((pilot.SEED + "\0" + document_id).encode("ascii")).hexdigest()
    assert pilot.selection_rank(document_id) == expected
    assert len(expected) == 64 and expected == expected.lower()


def test_length_buckets_have_no_gaps_at_boundaries() -> None:
    expected = [
        (1, "1-512"), (512, "1-512"), (513, "513-1024"), (1024, "513-1024"),
        (1025, "1025-2048"), (2048, "1025-2048"), (2049, "2049-4096"),
        (4096, "2049-4096"), (4097, "4097-8192"), (8192, "4097-8192"),
        (8193, "8193-16384"), (16384, "8193-16384"), (16385, "16385-32768"),
        (32768, "16385-32768"), (32769, "32769-65536"), (65536, "32769-65536"),
        (65537, "65537-131072"), (131072, "65537-131072"), (131073, "131073+"),
    ]
    assert [(value, pilot.length_bucket(value)) for value, _ in expected] == expected


def test_schedule_replay_is_bounded_and_covers_every_row() -> None:
    rows = [f"row-{i}" for i in range(37)]
    order = list(rows)
    random.Random(pilot.SCHEDULE_SEED).shuffle(order)
    replay_count = (-len(order)) % pilot.EFFECTIVE_BATCH
    assert 0 <= replay_count <= 15
    draws = order + order[:replay_count]
    assert set(draws) == set(rows)
    assert len(draws) % pilot.EFFECTIVE_BATCH == 0


def test_complete_document_rejects_gap_and_accepts_terminal() -> None:
    document_id = "a" * 64
    base = {
        "document_id": document_id, "source_sha256": document_id,
        "package": "pkg", "group_id": "g", "cpt_partition": "cpt_train",
        "document_token_count": 3, "original_chunk_count": 2,
    }
    rows = [
        {**base, "chunk_index": 0, "row_id": f"{document_id}:ctx16384:0", "input_ids": [0, 10, 11, 1],
         "token_start": 0, "source_token_start": 0, "token_end": 2, "source_token_end": 2,
         "overlap_context_tokens": 0, "is_document_end": False},
        {**base, "chunk_index": 1, "row_id": f"{document_id}:ctx16384:1", "input_ids": [0, 11, 12, 13, 1],
         "token_start": 2, "source_token_start": 2, "token_end": 5, "source_token_end": 5,
         "overlap_context_tokens": 1, "is_document_end": True},
    ]
    # Correct the metadata to match the synthetic complete spans.
    rows[0]["document_token_count"] = rows[1]["document_token_count"] = 5
    metadata = {**base, "document_token_count": 5}
    assert pilot.validate_doc_rows(rows, metadata) == 5
    rows[1]["source_token_start"] = rows[1]["token_start"] = 3
    try:
        pilot.validate_doc_rows(rows, metadata)
    except ValueError as exc:
        assert "source_gap" in str(exc)
    else:
        raise AssertionError("gap was not rejected")


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print("4/4 representative pilot preparation tests passed")
