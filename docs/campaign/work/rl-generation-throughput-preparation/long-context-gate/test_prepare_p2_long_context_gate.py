"""CPU-only contract tests for the prepared longest-TRAIN P2 gate."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import prepare_p2_long_context_gate as gate


class FakeRecord:
    def __init__(self, row_id: str, prompt_ids: list[int], family: str = "fixture"):
        self.row_id = row_id
        self.prompt_ids = tuple(prompt_ids)
        self.row = {"id": row_id, "split": "train", "family": family, "package_id": None}
        self.source_identity = {"candidate_file": "/source.jsonl", "candidate_line": 1}
        self.selection_geometry = {"overflow": False}

    def envelope(self):
        return {"prompt": {"text": "ignored", "ids": list(self.prompt_ids)}}


class LongContextGateTests(unittest.TestCase):
    def test_selection_is_stable_longest_distinct_stored_prompt(self):
        duplicate = [0, 10, 11, 12, 13]
        records = [
            FakeRecord("short", [0, 1]),
            FakeRecord("long-a", duplicate, family="a"),
            FakeRecord("long-a-duplicate", duplicate, family="b"),
            FakeRecord("long-b", [0, 20, 21, 22, 23], family="c"),
            FakeRecord("over-cap", [0] * 2049),
        ]
        selected = gate.select_longest_distinct(
            records, [record.row_id for record in records],
        )
        self.assertEqual([record.row_id for record in selected], ["long-a", "long-b"])
        self.assertEqual([len(record.prompt_ids) for record in selected], [5, 5])

    def test_prompt_batch_has_two_complete_g4_groups_in_selected_order(self):
        selected = [FakeRecord("a", [0, 100]), FakeRecord("b", [0, 200, 201])]
        prompts = gate.make_generation_prompts(selected)
        self.assertEqual(len(prompts), 8)
        self.assertEqual([row["ids"] for row in prompts[:4]], [[0, 100]] * 4)
        self.assertEqual([row["ids"] for row in prompts[4:]], [[0, 200, 201]] * 4)
        self.assertEqual(
            [len(prompts[index]["ids"]) for index in (0, 4)], [2, 3],
        )

    def test_live_argv_is_explicit_and_contains_no_dev_or_final_input(self):
        args = SimpleNamespace(
            execution_root="/exec",
            rows_path="/rows.jsonl",
            sidecar_path="/sidecar.jsonl",
            selected_ids_path="/selected.json",
            parent_manifest="/parent.json",
            native_base_path="/native-base",
            trainer_sha256="a" * 64,
            result_path="/result.json",
        )
        argv = gate._live_argv(args, args.result_path)
        self.assertIn("--live", argv)
        self.assertIn("--trainer-sha256", argv)
        self.assertNotIn("--dev-path", argv)
        self.assertNotIn("--final-path", argv)


if __name__ == "__main__":
    unittest.main()

