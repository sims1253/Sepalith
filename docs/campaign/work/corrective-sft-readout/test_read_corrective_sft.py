#!/usr/bin/env python3
"""CPU tests for the corrective SFT readout's protocol and geometry gates."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("read_corrective_sft", HERE / "read_corrective_sft.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = module.load_module("test_corrective_protocol", module.DEFAULT_PROTOCOL)
        panel_path = Path(
            "/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/"
            "docs/campaign/work/lead/corrected-dev75-v1/dev75-corrected-finish-v1.jsonl"
        )
        cls.finish = None
        with panel_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] == "e623a61b5a4c066358a477f2":
                    cls.finish = module.PanelCase(
                        row, cls.protocol.PromptContext.from_mapping(row["context"]), False
                    )
                    break
        assert cls.finish is not None

    def test_canonical_and_cap_predictions_are_distinguished(self):
        case = self.finish
        valid_raw = self.protocol.serialize_target("replace", case.row["region_new"])
        valid = {
            "id": case.row["id"],
            "raw_output": valid_raw,
            "generated_ids": [42, 1],
            "generated_tokens": 2,
            "cap_hit": False,
        }
        outcome = module.classify_result(valid, case, self.protocol, 2)
        self.assertTrue(outcome["protocol_valid"])
        self.assertTrue(outcome["exact_region"])
        capped = dict(valid, raw_output="partial", generated_ids=[42, 43], generated_tokens=2, cap_hit=True)
        outcome = module.classify_result(capped, case, self.protocol, 2)
        self.assertFalse(outcome["protocol_valid"])
        self.assertTrue(outcome["cap_hit"])
        self.assertEqual(outcome["failure"], "generation_cap_without_canonical_eos")

    def test_utf16_document_application_and_r_parse(self):
        case = self.finish
        raw = self.protocol.serialize_target("replace", case.row["region_new"])
        result = {
            "id": case.row["id"],
            "raw_output": raw,
            "generated_ids": [42, 1],
            "generated_tokens": 2,
        }
        outcome = module.classify_result(result, case, self.protocol, 512)
        before, after, geometry = module.apply_prediction(case, outcome["_parsed"], self.protocol)
        self.assertEqual(geometry["before_sha256"], case.row["context"]["replacement_range"]["content_sha256"])
        self.assertEqual(geometry["after_sha256"], case.row["source_provenance"]["correction"]["new_post_document_sha256"])
        parser = module.make_r_parser()
        self.assertTrue(module.parse_r(parser, after))
        self.assertNotEqual(before, after)

    def test_incremental_reader_processes_results_without_loading_root_results(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stream:
            json.dump({"status": "partial", "step": 50, "results": [
                {"id": "a", "raw_output": "x"},
                {"id": "b", "raw_output": "y"},
            ]}, stream, indent=2)
            stream.flush()
            seen = []
            meta = module.stream_case_file(Path(stream.name), seen.append)
        self.assertEqual(meta["status"], "partial")
        self.assertEqual([item["id"] for item in seen], ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
