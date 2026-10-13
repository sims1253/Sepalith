"""The checkpoint-root restriction is directory containment, not a string prefix."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from sepalith.training import paths
from sepalith.training.sft import full_weight_edit_sft as sft

ROOT = Path("/tmp/checkpoints")
PACKAGE = Path(paths.__file__).parent


def storage_recipe(trainer: str, archive: str) -> dict:
    return {
        "cohort": {"max_sequence_tokens": 16384},
        "development": {"final_set_access": False, "generation_budget": {
            "root_selected": True, "default_candidate": 1024, "prepared_max_new_tokens_bound": 8192}},
        "checkpoint_storage": {"mode": "e_same_filesystem_atomic", "c_hot_stage": "not_admitted",
                               "required_same_filesystem": True, "trainer_root": trainer, "archive_root": archive},
        "outputs": {"trainer": trainer, "archive": archive},
    }


class CheckpointRootTests(unittest.TestCase):
    def test_containment_rejects_siblings_the_root_itself_and_escapes(self):
        inside = ["/tmp/checkpoints/train", "/tmp/checkpoints/a/b/", "/tmp/checkpoints//train"]
        outside = ["/tmp/checkpoints-other/train", "/tmp/checkpoints", "/tmp/checkpoints/",
                   "/tmp/checkpoints/../other", "/tmp", "checkpoints/train", "tmp/checkpoints/train"]
        for value in inside:
            self.assertTrue(paths.under_checkpoint_root(value, ROOT), value)
            self.assertTrue(paths.under_checkpoint_root(Path(value), ROOT), value)
        for value in outside:
            self.assertFalse(paths.under_checkpoint_root(value, ROOT), value)

    def test_configured_root_is_normalized_and_validated(self):
        self.assertEqual(paths.checkpoint_root("/mnt/e/"), Path("/mnt/e"))
        self.assertEqual(paths.checkpoint_root("/tmp//checkpoints/."), ROOT)
        for value in ("", "/", "//", "relative/root", "."):
            with self.assertRaisesRegex(ValueError, "absolute directory"):
                paths.checkpoint_root(value)

    def test_default_and_environment_root(self):
        env = dict(os.environ, PYTHONPATH=str(PACKAGE.parents[1]))
        code = "from sepalith.training import paths; print(paths.CHECKPOINT_ROOT)"
        env.pop("SEPALITH_CHECKPOINT_ROOT", None)
        default = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(default.stdout.strip(), "/mnt/e")
        configured = subprocess.run([sys.executable, "-B", "-c", code], env=dict(env, SEPALITH_CHECKPOINT_ROOT="/tmp/checkpoints/"),
                                    capture_output=True, text=True, check=True)
        self.assertEqual(configured.stdout.strip(), "/tmp/checkpoints")
        refused = subprocess.run([sys.executable, "-B", "-c", code], env=dict(env, SEPALITH_CHECKPOINT_ROOT="/"),
                                 capture_output=True, text=True, check=False)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("absolute directory", refused.stderr)

    def test_sft_storage_validation_uses_containment(self):
        with mock.patch.object(paths, "CHECKPOINT_ROOT", ROOT):
            sft.validate_context_and_storage(storage_recipe("/tmp/checkpoints/train", "/tmp/checkpoints/archive"))
            for trainer, archive in (("/tmp/checkpoints-other/train", "/tmp/checkpoints/archive"),
                                     ("/tmp/checkpoints/train", "/tmp/checkpoints-other/archive")):
                with self.assertRaisesRegex(ValueError, "only E trainer/archive"):
                    sft.validate_context_and_storage(storage_recipe(trainer, archive))

    def test_no_module_compares_the_checkpoint_root_by_prefix(self):
        for source in sorted(PACKAGE.rglob("*.py")):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "startswith":
                    for arg in node.args:
                        prefix = arg.value if isinstance(arg, ast.Constant) else None
                        self.assertFalse(isinstance(arg, ast.Name) and arg.id == "CHECKPOINT_ROOT", source)
                        self.assertFalse(isinstance(prefix, str) and prefix.startswith("/mnt/e"), source)


if __name__ == "__main__":
    unittest.main()
