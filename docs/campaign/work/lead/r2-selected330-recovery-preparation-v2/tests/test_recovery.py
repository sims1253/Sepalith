#!/usr/bin/env python3
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
SRC = PACKET / "source/experiments/training"
sys.path[:0] = [str(PACKET), str(SRC)]
import prepare_recovery as P
import ordinary_canary_resume as O
import full_weight_cpt_trainer as T


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get("TMPDIR", "/mnt/e/sepalith/campaign-20260915/cache/tests"))
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="selected330-recovery-", dir=root)
        self.work = Path(self.temp.name)
        migration = json.loads((PACKET / "runtime-source-migration-admission.template.json").read_text())
        migration.update({
            "status": "admitted",
            "launch_authorized": True,
            "runtime_source_manifest_sha256": P.sha(PACKET / "source-manifest.json"),
        })
        self.migration = self.work / "migration.json"
        write(self.migration, migration)
        suffix = self.work.name
        self.trainer_root = Path("/home/m0hawk/.local/state/sepalith/campaign-20260915/training") / ("TEST-" + suffix) / "runtime"
        self.archive_root = Path("/mnt/e/sepalith/campaign-20260915/checkpoints") / ("TEST-" + suffix)
        self.out = self.work / "bound"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        return P.prepare(PACKET / "source-manifest.json", self.migration, self.trainer_root, self.archive_root, self.out)

    def test_recipe_changes_only_cadence_and_review_boundary(self):
        result = self.prepare()
        base = json.loads(P.BASE.read_text())
        recipe = json.loads((self.out / "runtime-recipe.json").read_text())
        old, new = T.identity(base), T.identity(recipe)
        self.assertNotEqual(P.canonical_sha(old), P.canonical_sha(new))
        for key in old:
            if key != "schedule":
                self.assertEqual(old[key], new[key])
        for key in old["schedule"]:
            if key not in {"checkpoint_every", "mandatory_stop_step"}:
                self.assertEqual(old["schedule"][key], new["schedule"][key])
        self.assertEqual((new["schedule"]["checkpoint_every"], new["schedule"]["mandatory_stop_step"]), (24, 354))
        self.assertEqual(result["first_stop_cursor"], 4608)

    def test_exact_save_and_stop_boundaries(self):
        self.prepare()
        runtime = json.loads((self.out / "runtime-recipe.json").read_text())["runtime"]
        observed = {step: T.milestone_action(step, 330, runtime, 66, 354) for step in range(331, 451)}
        saves = [step for step, action in observed.items() if action["save"]]
        self.assertEqual(saves, [354, 378, 402, 426, 450])
        self.assertEqual([step for step, action in observed.items() if action["stop"]], [354])
        self.assertFalse(observed[353]["save"])
        self.assertEqual(observed[354]["cursor"], 4608)

    def test_selected_checkpoint_transitions_to_exact_recovery_identity(self):
        result = self.prepare()
        recipe_path = self.out / "runtime-recipe.json"
        recipe = json.loads(recipe_path.read_text())
        transition = json.loads((self.out / "selected-packed330-transition.review.json").read_text())
        transition.update({"status": "admitted", "launch_authorized": True})
        transition_path = self.work / "transition.json"
        write(transition_path, transition)
        observed, lineage = O.resolve_resume_identity(recipe_path, T.identity(recipe), P.SOURCE_CHECKPOINT, transition_path)
        self.assertEqual(O.canonical_sha(observed), transition["source_identity_sha256"])
        self.assertEqual(lineage["destination_identity_sha256"], result["destination_identity_sha256"])
        changed = copy.deepcopy(recipe)
        changed["runtime"]["learning_rate"] = 4e-6
        with self.assertRaisesRegex(ValueError, "more than recovery cadence"):
            O.resolve_resume_identity(recipe_path, T.identity(changed), P.SOURCE_CHECKPOINT, transition_path)

    def test_root_admissions_remain_fail_closed(self):
        self.prepare()
        for name in ("selected-packed330-transition.review.json", "continuation-330.review.json", "execution-stop-354.review.json"):
            value = json.loads((self.out / name).read_text())
            self.assertFalse(value["launch_authorized"])
            self.assertEqual(value["status"], "ROOT_MUST_SET_admitted")

    def test_emitted_frontdoors_bind_fresh_recipe_and_resume(self):
        self.prepare()
        cpu = json.loads((self.out / "cpu-preflight-command.json").read_text())
        gpu = json.loads((self.out / "training-command.json").read_text())
        guard = json.loads((self.out / "guard-command.json").read_text())
        self.assertIn("CUDA_VISIBLE_DEVICES=", cpu)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", gpu)
        self.assertIn(str((self.out / "runtime-recipe.json").resolve()), cpu)
        self.assertIn(str(P.SOURCE_CHECKPOINT), cpu)
        self.assertIn(str((PACKET / "run_with_allocator.py").resolve()), gpu)
        self.assertIn(str((self.out / "training-command.json").resolve()), guard)
        self.assertTrue(any(str(value).endswith("SFT11-selected330-recovery-cadence24-v2-host-supervision") for value in guard))

    def test_actual_load_bound_accepts_only_admitted_schedule_delta(self):
        self.prepare()
        recipe_path = self.out / "runtime-recipe.json"
        recipe = json.loads(recipe_path.read_text())
        import native_runtime_contract as N

        def attested_identity(path, bundle):
            path = Path(path)
            if path.name == "manifest.json":
                return {"sha256": recipe["cohort"]["streaming_cache"]["manifest_sha256"]}
            if path == Path(recipe["cohort"]["rows"]["path"]):
                return {"sha256": recipe["cohort"]["rows"]["sha256"]}
            if path == Path(recipe["cohort"]["draw_schedule"]["path"]):
                return {"sha256": recipe["cohort"]["draw_schedule"]["sha256"]}
            raise AssertionError(path)

        with mock.patch.object(N, "require_attested_identity", side_effect=attested_identity), mock.patch.object(N, "require_attested", return_value={}):
            loaded = T.load_bound(recipe_path)
            self.assertEqual(loaded["runtime"]["checkpoint_every"], 24)
            changed = copy.deepcopy(recipe)
            changed["runtime"]["learning_rate"] = 4e-6
            write(recipe_path, changed)
            with self.assertRaisesRegex(ValueError, "recovery destination identity"):
                T.load_bound(recipe_path)

    def test_native_peak_capacity_bound(self):
        checkpoint = 17_250_934_153
        # During publication the source checkpoint is renamed to E staging;
        # at most source330 + current serialization + one transient predecessor
        # may coexist on native storage. Root must evict verified old hot copies.
        self.assertLessEqual(3 * checkpoint, 70 * 1024**3)
        self.assertGreater(70 * 1024**3 - 3 * checkpoint, 20 * 1024**3)

    def test_source_manifest_matches_bytes(self):
        manifest = json.loads((PACKET / "source-manifest.json").read_text())
        self.assertEqual(manifest["files_count"], len(manifest["files"]))
        for record in manifest["files"]:
            path = PACKET / record["path"]
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
