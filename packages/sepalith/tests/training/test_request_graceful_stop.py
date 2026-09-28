"""The save-and-stop request is written for whichever trainer binds the recipe."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sepalith.training.cpt import full_weight_cpt_trainer as cpt
from sepalith.training.sft import full_weight_edit_sft as sft
from sepalith.training.supervisors import request_graceful_stop as stop


class GracefulStopTests(unittest.TestCase):
    def bound(self, root, schema):
        recipe = root / "bound.json"
        recipe.write_text(json.dumps({"schema": schema, "outputs": {"graceful_stop": str(root / "control" / "stop.json")}}))
        return recipe

    def run_request(self, schema, module, other):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = self.bound(root, schema)
            loaded = json.loads(recipe.read_text())
            with mock.patch.object(module, "load_bound", return_value=loaded) as used, \
                    mock.patch.object(other, "load_bound", side_effect=AssertionError("wrong trainer")):
                result = stop.request(recipe)
            used.assert_called_once_with(recipe.resolve())
            written = json.loads((root / "control" / "stop.json").read_text())
            # Exactly the payload both trainers compare against at an optimizer boundary.
            self.assertEqual(written, {"action": "save_and_stop", "bound_recipe_sha256": module.sha256(recipe)})
            self.assertEqual(result["status"], "save_and_stop_requested")
            with self.assertRaises(FileExistsError), \
                    mock.patch.object(module, "load_bound", return_value=loaded):
                stop.request(recipe)

    def test_sft_recipe_uses_the_sft_trainer(self):
        self.run_request("sepalith.sft11.full-weight-edit-sft-eval-gate-bound.v1", sft, cpt)

    def test_cpt_recipe_still_uses_the_cpt_trainer(self):
        self.run_request("sepalith.sft11.full-weight-cpt-stage-transition-bound.v1", cpt, sft)

    def test_unknown_recipe_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = self.bound(Path(tmp), "sepalith.sft11.full-weight-edit-sft-eval-gate-template.v1")
            with self.assertRaisesRegex(ValueError, "no trainer accepts"):
                stop.request(recipe)
            self.assertFalse((Path(tmp) / "control" / "stop.json").exists())


if __name__ == "__main__":
    unittest.main()
