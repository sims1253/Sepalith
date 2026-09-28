#!/usr/bin/env python3
"""Small CPU-only tests for the P2 split comparer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("verify_rl03_p2.py")
SPEC = importlib.util.spec_from_file_location("verify_rl03_p2", MODULE_PATH)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def row(ordinal: int, source: str) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "source_id": source,
        "prompt_sha256": f"prompt-{ordinal}",
        "output_sha256": f"output-{ordinal}",
        "reward": float(ordinal),
        "failure": None,
    }


def checkpoint(adapter: str = "adapter", optimizer: str = "optimizer", rng: str = "rng") -> dict[str, object]:
    exact = {
        name: {"bytes": 1, "sha256": value}
        for name, value in {
            "adapter_model.safetensors": adapter,
            "optimizer.pt": optimizer,
            "scheduler.pt": "scheduler",
            "rng_state.pth": rng,
        }.items()
    }
    # Metadata is deliberately different between arms.  It is allowlisted.
    files = {**exact, "trainer_state.json": {"bytes": 1, "sha256": "run-specific"}}
    return {
        "file_hashes": files,
        "sampler": {"source_draw_schedule_sha256": VERIFY.SCHEDULE_SHA256, "source_draw_sequence_sha256": VERIFY.SEQUENCE_SHA256, "current_index": 1},
    }


class P2SplitComparerTests(unittest.TestCase):
    def setUp(self) -> None:
        first_row, second_row = row(0, "source-a"), row(1, "source-b")
        self.continuous = {"status": "pass", "join_sequence": [first_row, second_row], "checkpoint_states": {1: checkpoint(), 2: checkpoint()}}
        self.first = {"status": "pass", "join_sequence": [first_row], "checkpoint_states": {1: checkpoint()}}
        self.resumed = {"status": "pass", "join_sequence": [second_row], "checkpoint_states": {2: checkpoint()}}

    def test_allowed_metadata_difference_passes(self) -> None:
        self.first["checkpoint_states"][1]["file_hashes"]["trainer_state.json"] = {"bytes": 2, "sha256": "first-run"}
        self.resumed["checkpoint_states"][2]["file_hashes"]["trainer_state.json"] = {"bytes": 3, "sha256": "resume-run"}
        result = VERIFY.compare_runs(self.continuous, self.first, self.resumed)
        self.assertEqual(result["status"], "pass")
        self.assertIn("trainer_state.json", result["boundary_hashes"]["1"]["metadata_allowlist"])

    def test_optimizer_or_rng_change_fails(self) -> None:
        for field, value in (("optimizer.pt", "changed-optimizer"), ("rng_state.pth", "changed-rng")):
            with self.subTest(field=field):
                self.resumed["checkpoint_states"][2]["file_hashes"][field] = {"bytes": 1, "sha256": value}
                result = VERIFY.compare_runs(self.continuous, self.first, self.resumed)
                self.assertEqual(result["status"], "fail")
                self.assertTrue(any(item["kind"] == "exact_resume_bytes" and item["file"] == field for item in result["failures"]))
                self.resumed["checkpoint_states"][2] = checkpoint()

    def test_missing_resume_remains_pending(self) -> None:
        self.resumed["status"] = "pending"
        result = VERIFY.compare_runs(self.continuous, self.first, self.resumed)
        self.assertEqual(result["status"], "pending")


if __name__ == "__main__":
    unittest.main()
