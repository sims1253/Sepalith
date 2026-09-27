"""CPU checks for the root-bindable expanded SFT recipe and draw schedule."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
RECIPE = HERE / "recipe.json"
SCHEDULE = HERE / "expanded-full-noop25-16000-draw-manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ExpandedScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads(RECIPE.read_text())
        cls.schedule = json.loads(SCHEDULE.read_text())

    def test_schedule_has_exact_batch_mixture_and_coverage(self):
        schedule = self.schedule
        draws = schedule["draws"]
        self.assertEqual(schedule["max_steps"], 1000)
        self.assertEqual(schedule["effective_batch"], 16)
        self.assertEqual(len(draws), 16000)
        self.assertEqual(sum(item["semantic_noop"] for item in draws), 4000)
        for offset in range(0, len(draws), 16):
            self.assertEqual(sum(item["semantic_noop"] for item in draws[offset:offset + 16]), 4)
        exposures = Counter(item["row_id"] for item in draws)
        self.assertEqual(len(exposures), 11505)
        self.assertEqual(min(exposures.values()), 1)
        self.assertEqual(max(exposures.values()), 4)
        self.assertTrue(all(item["group_id"] == item["source_id"] for item in draws))

    def test_each_pool_exhausts_before_its_first_repeat(self):
        draws = self.schedule["draws"]
        noops = [item["row_id"] for item in draws if item["semantic_noop"]]
        edits = [item["row_id"] for item in draws if not item["semantic_noop"]]
        self.assertEqual(len(set(noops[:1094])), 1094)
        self.assertEqual(noops[1094], noops[0])
        self.assertEqual(len(set(edits[:10411])), 10411)
        self.assertEqual(edits[10411], edits[0])

    def test_recipe_binds_schedule_and_fresh_merged_parent(self):
        recipe = self.recipe
        self.assertEqual(recipe["stage"], "expanded_task_sft_v1")
        self.assertEqual(recipe["identity"]["parent"]["kind"], "sft_merged")
        self.assertEqual(recipe["identity"]["parent"]["weights_sha256"],
                         "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c")
        self.assertEqual(recipe["identity"]["parent"]["previous_checkpoint_step"], 500)
        self.assertEqual(recipe["identity"]["policy"]["initialization"],
                         "new_lora_on_sft_merged_parent")
        self.assertEqual(recipe["parameters"]["train_max_target_tokens"], 1024)
        self.assertEqual(recipe["development_max_new_tokens"], 192)
        self.assertIsNone(recipe["resume_from"])
        self.assertEqual(recipe["draw_schedule"]["sha256"], digest(SCHEDULE))
        self.assertEqual(recipe["identity"]["data"]["draw_schedule_sha256"], digest(SCHEDULE))
        inputs = {record["path"] for record in recipe["inputs"]}
        for key in ("manifest", "previous_recipe", "previous_checkpoint_manifest"):
            record = recipe["sft_merged_parent"][key]
            self.assertIn(record["path"], inputs)
            self.assertEqual(record["sha256"], digest(Path(record["path"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
