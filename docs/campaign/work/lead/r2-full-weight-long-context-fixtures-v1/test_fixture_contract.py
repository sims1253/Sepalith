import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from materialize import SELECTED, SOURCE_SHA256

SOURCE = pathlib.Path("/mnt/e/sepalith/campaign-20260915/data-work/DAT10-novel-v1/roxygen-repair-materialization-v2-10017/repaired-full-file-token-rows.jsonl")
DRIVER = pathlib.Path("docs/campaign/work/lead/r2-full-weight-hybrid-smoke-root-v4/full_weight_smoke.py")


def canonical_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class FixtureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "fixture-manifest.json").read_text())
        wanted = {row_id for rows in SELECTED.values() for row_id in rows}
        cls.source_rows = {}
        with SOURCE.open() as source:
            for line in source:
                row = json.loads(line)
                if row.get("row_id") in wanted:
                    cls.source_rows[row["row_id"]] = row

    def test_source_and_all_selected_ids_are_exact(self):
        self.assertEqual(self.manifest["source"]["sha256"], SOURCE_SHA256)
        self.assertEqual(set(self.source_rows), {x for rows in SELECTED.values() for x in rows})

    def test_each_cohort_has_two_intact_train_rows_and_valid_boundaries(self):
        for cohort in self.manifest["cohorts"]:
            cap = cohort["cap"]
            rows = [json.loads(line) for line in pathlib.Path(cohort["path"]).read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["fixture_provenance"]["source_path"] for row in rows}), 2)
            for row in rows:
                original = self.source_rows[row["row_id"]]["token_row"]
                self.assertEqual(original["split"], "train")
                self.assertEqual(row["input_ids"], original["input_ids"])
                self.assertEqual(len(row["input_ids"]), len(row["labels"]))
                self.assertTrue(cap // 2 < len(row["input_ids"]) <= cap)
                start = original["target_start"]
                self.assertEqual(row["labels"][:start], [-100] * start)
                self.assertEqual(row["labels"][start:], original["input_ids"][start:])
                self.assertEqual(original["target_body_tokens"] + original["target_terminal_tokens"], original["input_ids"][start:-1])
                self.assertEqual(original["input_ids"][-1], original["eos_token_id"])
                self.assertEqual(row["supervised_tokens"], len(original["input_ids"]) - start)
                evidence = next(x for x in cohort["rows"] if x["row_id"] == row["row_id"])
                self.assertEqual(evidence["input_ids_sha256"], canonical_sha(row["input_ids"]))
                self.assertEqual(evidence["labels_sha256"], canonical_sha(row["labels"]))

    def test_frozen_smoke_driver_accepts_exact_caps_and_rejects_truncating_cap(self):
        spec = importlib.util.spec_from_file_location("frozen_full_weight_smoke", DRIVER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for cohort in self.manifest["cohorts"]:
            path = pathlib.Path(cohort["path"])
            rows = module.read_rows(path, cohort["cap"])
            self.assertEqual(len(rows), 2)
            with self.assertRaises(ValueError):
                module.read_rows(path, cohort["minimum_sequence_tokens"] - 1)


if __name__ == "__main__": unittest.main()
