"""Exercise the assembler's actual edit-row seam using tiny in-memory records."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_legacy_protocol import legacy_renderer

ROOT = Path(__file__).resolve().parents[3]
ASSEMBLER = ROOT / "experiments/post-processing/assemble_sft_v5.py"


class AssemblerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("assembler_protocol_fixture", ASSEMBLER)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load assembler fixture: {ASSEMBLER}")
        cls.assembler = importlib.util.module_from_spec(spec)
        # Import defines paths and functions; main is not called and no corpus
        # loaders run. Restore the script's source bootstrap after the import.
        original_path = list(sys.path)
        try:
            spec.loader.exec_module(cls.assembler)
        finally:
            sys.path[:] = original_path
        cls.legacy = staticmethod(legacy_renderer())

    def test_edit_row_matches_legacy_and_preserves_suffix_target(self):
        example = {
            "path": "R/fit.R", "prefix": ["fit <- function(x) {", ""],
            "region_old": ["  x <- na.omit(x)", "  mean(x)  ", "}"],
            "region_new": ["  x <- na.omit(x)", "  mean(x, na.rm = TRUE)", "}"],
            "suffix": ["", "# café  ", ""], "cursor_idx": 100,
            "event_diff": "```diff\nUser edited fit.R\n\n-mean(x)\n+mean(x, na.rm = TRUE)\n```\n",
            # These values belong to the scenario, not the legacy render input.
            "cursor_column": 999, "schema_version": "scenario.v5",
            "metadata": {"research_only": object()},
        }
        result = self.assembler.edit_row(example, "format_propagation", "fixture")
        expected_example = {**example, "cursor_idx": 0}
        self.assertEqual(result["prompt"].encode(), self.legacy(expected_example).encode())
        self.assertEqual(result["target"], "  mean(x, na.rm = TRUE)\n}\n>>>>>>> UPDATED")
        self.assertIn("  x <- na.omit(x)<|user_cursor|>\n", result["prompt"])
        self.assertEqual(example["cursor_idx"], 100)

    def test_no_op_empty_and_blank_targets_preserve_exact_bytes(self):
        for new_lines, expected_target in (([], "\n>>>>>>> UPDATED"),
                                           ([""], "\n\n>>>>>>> UPDATED")):
            example = {"path": "R/empty.R", "prefix": [], "region_old": ["}"],
                       "region_new": new_lines, "cursor_idx": 99}
            with self.subTest(new_lines=new_lines):
                result = self.assembler.edit_row(example, "no_op", "fixture", fd=0, cursor_after=0)
                legacy_example = {**example, "suffix": [], "cursor_idx": 0}
                self.assertEqual(result["prompt"], self.legacy(legacy_example))
                self.assertEqual(result["target"], expected_target)
                self.assertEqual(result["prompt"],
                                 "<[fim-suffix]>\n<[fim-prefix]><filename>edit_history\n"
                                 "<filename>R/empty.R\n<<<<<<< CURRENT\n}<|user_cursor|>\n"
                                 "=======\n<[fim-middle]>")

    def test_direct_help_from_other_directory_needs_no_install(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-B", str(ASSEMBLER), "--help"],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--out", result.stdout)
        self.assertIn("--data-root", result.stdout)
        self.assertIn("--finish-source", result.stdout)
        self.assertEqual(self.assembler.REPO, ROOT)


if __name__ == "__main__":
    unittest.main()
