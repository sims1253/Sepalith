"""C09 TRAIN search panel exclusions (sepalith.quant.search_panel)."""
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.quant import search_panel as sp


def row(row_id, family="f"):
    return {"id": row_id, "family": family, "split": "train", "input_ids": [1, 2, 3], "target_start": 2}


class ExclusionTest(unittest.TestCase):
    def test_manifest_row_ids_excludes_every_reserved_row(self):
        manifest = {"calibration": {"selected_row_ids": ["cal"], "filler_row_ids": ["fill"]},
                    "kld": {"windows": [["kld", 0, 1024]]},
                    "exclusion": {"excluded_row_ids": {"dev": "dev_region_hash"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest))
            reserved = sp.manifest_row_ids(path)
        self.assertEqual(reserved, {"calibration": {"cal", "fill"}, "kld": {"kld"}, "dev_excluded": {"dev"}})
        rows = [row(i) for i in ("cal", "fill", "kld", "dev", "free")]
        panel = sp.build_panel(rows, per_family=10, seed=1, calibration_ids=reserved["calibration"],
                               kld_ids=reserved["kld"], dev_excluded_ids=reserved["dev_excluded"])
        self.assertEqual([case["id"] for case in panel], ["free"])


if __name__ == "__main__":
    unittest.main()
