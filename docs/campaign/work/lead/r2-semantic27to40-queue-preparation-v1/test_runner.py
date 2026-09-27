#!/usr/bin/env python3
"""Small local tests for the 27--40 launch guards; no campaign data is read."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


PACKET = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("semantic27to40_run", PACKET / "run.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_exact_range() -> None:
    assert module.SELECTED == list(range(27, 41))
    assert len(module.SELECTED) == 14
    assert not set(module.SELECTED) & set(module.PRIOR)
    command = module.make_command(module.REPLAY_DEFAULT, module.OUTPUT_DEFAULT)
    assert command[:3] == ["timeout", "--signal=TERM", "--kill-after=30s"]
    shard_arg = command[command.index("--shards") + 1]
    assert [int(value) for value in shard_arg.split(",")] == list(range(27, 41))
    assert command[command.index("--max-workers") + 1] == "2"
    assert all(value >= 27 for value in [int(value) for value in shard_arg.split(",")])


def test_reusable_child_requires_content_binding() -> None:
    with tempfile.TemporaryDirectory(prefix="semantic27to40-test-") as name:
        root = Path(name)
        target = root / "shard-0027"
        target.mkdir()
        ledger = target / "semantic-ledger.jsonl"
        ledger.write_text('{"row_id":"r1"}\n')
        source = {
            "shard": 27,
            "binding_sha256": "binding",
            "queued_ids": ["r1"],
            "queued_rows": 1,
            "queued_ids_sha256": module.ids_sha(["r1"]),
            "provenance_ledger": {"sha256": "provenance"},
        }
        output = {
            "bytes": ledger.stat().st_size,
            "path": str(ledger),
            "rows": 1,
            "sha256": module.sha(ledger),
            "row_ids_sha256": module.ids_sha(["r1"]),
        }
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "sepalith.dat10.sourcewalk_roxy_semantic.v6",
                    "status": "complete_review_only",
                    "training_admission": False,
                    "code": {},
                    "streaming_binding": source,
                    "source_provenance_binding_sha256": "binding",
                    "queued_ids_sha256": module.ids_sha(["r1"]),
                    "output": output,
                    "output_binding": output,
                }
            )
        )
        reused = module.reusable_child(target, source, {})
        assert reused["status"] == "reused_independently_verified"
        ledger.write_text('{"row_id":"changed"}\n')
        try:
            module.reusable_child(target, source, {})
        except module.IntakeError:
            pass
        else:
            raise AssertionError("changed semantic bytes were reusable")


if __name__ == "__main__":
    test_exact_range()
    test_reusable_child_requires_content_binding()
    print("semantic27to40 runner tests: PASS")
