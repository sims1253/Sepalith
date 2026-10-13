"""CPU-only checks for the materialized corrected schedule and its seam."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_finish_corrected_schedule as builder  # noqa: E402


SCHEDULE = HERE / "finish-corrected-draws-200.json"
SIDECAR = HERE / "finish-corrected-sampler-metadata.json"


class CorrectedScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        cls.sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
        cls.sampler = builder._load_sampler_module()

    def test_actual_manifest_passes_immutable_validator_and_geometry(self) -> None:
        self.sampler.validate_draw_manifest(self.manifest)
        self.assertEqual(self.manifest["status"], "complete")
        self.assertEqual(self.manifest["draw_count"], 3200)
        self.assertEqual(self.manifest["max_steps"] * self.manifest["effective_batch"], 3200)
        self.assertEqual(self.manifest["policy"]["seed"], 3407)
        self.assertEqual(self.manifest["policy"]["effective_family_ceiling"], 800)
        self.assertLessEqual(max(self.manifest["exposure"]["family"].values()), 800)
        self.assertLessEqual(self.manifest["achieved_mixture"]["length"]["long"], 640)

    def test_physical_exclusion_and_updated_metadata_identity(self) -> None:
        rejected = set(self.sidecar["rejected_ids"])
        rows = {record["row_id"]: record for record in self.sidecar["records"]}
        self.assertEqual(len(rows), 11526)
        self.assertEqual(sum(record["correction_status"] == "eligible_repaired" for record in rows.values()), 4051)
        self.assertEqual(sum(record["correction_status"] == "unchanged" for record in rows.values()), 7475)
        self.assertEqual(len(rejected), 238)
        self.assertNotIn(next(iter(rejected)), rows)
        self.assertTrue(set(self.manifest["row_ids"]).isdisjoint(rejected))
        self.assertEqual(self.sidecar["corrected_rows"]["sha256"], self.manifest["token_rows_sha256"])

    def test_optimizer_prefixes_keep_all_families_and_noop_floor(self) -> None:
        for step in (50, 100, 200):
            draws = self.manifest["draws"][: step * 16]
            self.assertEqual(len(draws), step * 16)
            self.assertEqual(sum(item["semantic_noop"] for item in draws), step * 16 // 10)
            self.assertEqual({item["family"] for item in draws}, set(self.manifest["exposure"]["family"]))

    def test_row_builder_preserves_prompt_geometry_and_rejects_bad_bos(self) -> None:
        original = {
            "row_id": "r1",
            "family": "finish_block",
            "package_id": "pkg",
            "source_id": "source",
            "source_kind": "ordinary",
            "split": "train",
            "semantic_noop": False,
            "operation": "replace",
            "prompt_tokens": 3,
            "target_tokens": 3,
            "total_tokens": 6,
            "length_bucket": "short",
            "naturally_long": False,
            "provenance": "base",
        }
        row = {
            "id": "r1",
            "family": "finish_block",
            "package_id": "pkg",
            "split": "train",
            "input_ids": [0, 8, 9, 11, 12, 1, 1],
            "target_start": 3,
            "prompt_token_count": 2,
            "target_body_tokens": [11, 12],
            "target_body_token_count": 2,
            "target_terminal_tokens": [13],
            "target_terminal_token_count": 1,
            "target_token_count": 3,
            "bos_token_id": 0,
            "eos_token_id": 1,
        }
        record = builder._updated_sampler_record(row, original, correction_status="eligible_repaired")
        self.assertEqual(record["prompt_tokens"], 3)
        self.assertEqual(record["target_tokens"], 4)
        self.assertEqual(record["total_tokens"], 7)
        self.assertEqual(record["length_bucket"], "short")

        bad = copy.deepcopy(row)
        bad["input_ids"][0] = 99
        with self.assertRaises(builder.SchedulePreparationError):
            builder._updated_sampler_record(bad, original, correction_status="eligible_repaired")


if __name__ == "__main__":
    unittest.main()
