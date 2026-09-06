import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sepalith.protocol import EditContext, render_context


class ContextCommandTests(unittest.TestCase):
    def test_legacy_migration_and_render_from_other_directory(self):
        source = Path(__file__).resolve().parents[1] / "src"
        record = {"path": "R/é.R", "prefix": [], "region_old": ["é <- 1"],
                  "suffix": [], "cursor_idx": 0, "case_id": "kept"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(source)}
            command = [sys.executable, "-m", "sepalith.context_cli"]
            normalized = subprocess.run(command + ["normalize", str(path), "--legacy"],
                                        cwd=directory, env=env, capture_output=True, check=True)
            self.assertEqual(json.loads(normalized.stdout)["case_id"], "kept")
            path.write_bytes(normalized.stdout)
            rendered = subprocess.run(command + ["render", str(path)], cwd=directory,
                                      env=env, capture_output=True, check=True)
            self.assertEqual(rendered.stdout, render_context(EditContext.from_legacy(record)).encode())

    def test_unversioned_input_requires_explicit_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text("{}", encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "sepalith.context_cli", "render", str(path)],
                                    capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"Missing context fields", result.stderr)
