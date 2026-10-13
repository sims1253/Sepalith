import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("audit_union", HERE / "audit_union.py")
audit = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(audit)


def row():
    return {
        "input_ids": [0, 10, 20, 30, 40, 1],
        "target_start": 2,
        "target_body_tokens": [20],
        "target_terminal_tokens": [30, 40],
        "target_body_token_count": 1,
        "target_terminal_token_count": 2,
        "prompt_token_count": 1,
        "target_token_count": 3,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "split": "train",
        "target_text": "x\n>>>>>>> UPDATED",
    }


class AuditContractTest(unittest.TestCase):
    def test_explicit_bos_and_eos_are_outside_stored_counts(self):
        ok, reasons, prompt_hash, target_hash = audit.token_contract(row())
        self.assertTrue(ok, reasons)
        self.assertEqual(len(prompt_hash), 64)
        self.assertEqual(len(target_hash), 64)

    def test_rejects_wrong_suffix_geometry(self):
        r = row(); r["input_ids"][2] = 99
        ok, reasons, _, _ = audit.token_contract(r)
        self.assertFalse(ok)
        self.assertIn("target_suffix_geometry", reasons)

    def test_rejects_dev_split(self):
        r = row(); r["split"] = "dev"
        ok, reasons, _, _ = audit.token_contract(r)
        self.assertFalse(ok)
        self.assertIn("split_not_train", reasons)

    def test_published_report_closes_all_rows(self):
        p = Path("/mnt/e/sepalith/campaign-20260915/data-work/Sourcewalk-SFT-admission-review-v1/review.json")
        o = json.loads(p.read_text())
        self.assertEqual(o["converted_audit"]["rows"], 86910)
        self.assertEqual(o["converted_audit"]["token_contract_failures"], 0)
        self.assertEqual(len(o["inputs"]["sourcewalk_token_rows"]), 41)
        self.assertEqual(sum(x["rows"] for x in o["inputs"]["sourcewalk_token_rows"]), 86910)
        self.assertEqual(o["preconversion"]["input_rows"], 200753)
        self.assertEqual(sum(o["preconversion"]["input_by_family"].values()), 200753)
        self.assertEqual(o["cross_pool"]["old_roxy10017_sourcewalk_id_overlap"], 10017)
        self.assertEqual(o["cross_pool"]["safe_vs_authoritative_current15006"]["new_unique_prompt_target_pairs"], 8595)
        self.assertEqual(o["inputs"]["current15006"]["sha256"], "3f551c446308575da84065ff2c499c09e1f631d9387ce995deb4920d72177d5e")


if __name__ == "__main__":
    unittest.main()
