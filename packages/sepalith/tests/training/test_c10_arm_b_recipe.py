"""Card C10: the arm-B bound recipe equals arm A's apart from its cohort and schedule, and its night jobs are valid."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.ops import night_queue
from sepalith.training.sft import full_weight_edit_sft as trainer
from sepalith.training.sft import recipe_v2

REPO = Path(__file__).resolve().parents[4]
DEPLOY = Path("/home/m0hawk/sandbox/Sepalith")
PATHS = recipe_v2.PathMap(DEPLOY, REPO)
PLAN = REPO / "docs/training/plan-20260927"
ARM_A = PLAN / "c04/bound/sft-v2-lr1e-5.json"
ARM_B = PLAN / "c10/bound/sft-v2-arm-b-lr1e-5.json"
GATES = [342, 684, 1026, 1368]
# Top-level keys and nested paths that may differ between the arms; everything else must be identical.
DIFFERENT_TOP = {"arm", "cohort", "data_queue", "outputs", "checkpoint_storage", "root_admission", "template", "output_tag"}
DIFFERENT_RUNTIME = {"max_steps", "checkpoint_every", "evaluation_steps", "selected_milestones", "mandatory_dev_gate_steps",
                     "gate_evidence_directory"}
DIFFERENT_GATE = {"mandatory_steps", "evidence_directory", "override_file_pattern"}


def localized(path):
    return PATHS.localize(json.loads(Path(path).read_text()))


class ArmBRecipeTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = localized(ARM_A), localized(ARM_B)

    def test_only_cohort_schedule_and_outputs_differ_from_arm_a(self):
        a, b = self.a, self.b
        for key in (set(a) | set(b)) - DIFFERENT_TOP - {"runtime", "dev_gate", "pre_registered"}:
            self.assertEqual(a.get(key), b.get(key), key)
        self.assertEqual({k: v for k, v in a["runtime"].items() if k not in DIFFERENT_RUNTIME},
                         {k: v for k, v in b["runtime"].items() if k not in DIFFERENT_RUNTIME})
        self.assertEqual({k: v for k, v in a["dev_gate"].items() if k not in DIFFERENT_GATE},
                         {k: v for k, v in b["dev_gate"].items() if k not in DIFFERENT_GATE})
        self.assertEqual({k: v for k, v in a["pre_registered"].items() if k != "gate_updates"},
                         {k: v for k, v in b["pre_registered"].items() if k != "gate_updates"})
        self.assertEqual(a["source"], b["source"])  # same code manifest as arm A
        self.assertEqual(a["parent"], b["parent"])
        self.assertEqual((b["arm"]["learning_rate"], b["runtime"]["optimizer"]["side_lr"]), (1e-5, 1e-6))
        self.assertEqual(b["retention"]["policy"], "retain_all")

    def test_arm_b_binds_the_accepted_v4_cohort_and_schedule(self):
        b = self.b
        recipe_v2.validate_pre_registered(b)
        recipe_v2.verify_source(b["source"])
        for key in ("root_admission", "template"):
            self.assertEqual(recipe_v2.sha256(b[key]["path"]), b[key]["sha256"], key)
        self.assertEqual(b["cohort"]["rows"]["sha256"], "256a418b914014f12b7a848e4d899e210407435cbf98c243f98d50225e8b92ec")
        self.assertEqual(b["cohort"]["draw_schedule"]["sha256"], "1e89713e95b8ab789b56077b9c01ad78c6b469014cd803b40ea7add0e04598d2")
        self.assertEqual((b["cohort"]["unique_rows"], b["cohort"]["updates"]), (18867, 1368))
        self.assertEqual(b["dev_gate"]["mandatory_steps"], GATES)
        self.assertEqual(b["cohort"]["schedule_semantics"]["exposure_histogram_by_pool"],
                         {"edit": {"1": 17762, "2": 11}, "noop": {"3": 272, "4": 822}})
        recipe_v2.verify_data_queue_v2(b["data_queue"], b["cohort"])
        kwargs = trainer.training_arguments_kwargs(b, Path("/tmp/x"))
        self.assertEqual((kwargs["warmup_steps"], kwargs["lr_scheduler_type"], kwargs["lr_scheduler_kwargs"], kwargs["max_steps"]),
                         (32, "cosine_with_min_lr", {"min_lr_rate": 0.1}, 1368))
        self.assertNotEqual(b["outputs"]["archive"], self.a["outputs"]["archive"])
        self.assertNotEqual(json.dumps(trainer.identity(b), sort_keys=True), json.dumps(trainer.identity(self.a), sort_keys=True))

    @unittest.skipUnless(Path("/mnt/e/sepalith/campaign-20260915/data-work/C10-roxygen-admission-v1/arm-b-v4").is_dir(),
                         "campaign data is not mounted")
    def test_load_and_preflight_with_a_stand_in_parent(self):
        # The real parent is 5 GB; the cohort, schedule, panels and token denominators are verified for real.
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            parent.mkdir()
            recipe = copy.deepcopy(self.b)
            for name in recipe["parent"]["files"]:
                (parent / name).write_text(name)
                recipe["parent"]["files"][name] = recipe_v2.sha256(parent / name)
            recipe["parent"]["path"] = str(parent)
            path = Path(tmp) / "recipe.json"
            path.write_text(json.dumps(recipe))
            self.assertEqual(trainer.load_bound(path)["schema"], recipe_v2.BOUND_SCHEMA)
            report = trainer.preflight(path)
            self.assertEqual((report["unique_rows"], report["draws"], report["updates"]), (18867, 21888, 1368))


class ArmBJobTests(unittest.TestCase):
    def test_jobs_are_valid_and_point_at_the_arm_b_recipe(self):
        jobs = {path.stem: json.loads(path.read_text()) for path in sorted((PLAN / "c10/jobs").glob("C10-arm-b-*.json"))}
        self.assertEqual(sorted(jobs), ["C10-arm-b-lr1e-5", "C10-arm-b-lr1e-5-full"])
        for name, job in jobs.items():
            self.assertEqual(job["id"], name)
            night_queue.validate_job(copy.deepcopy(job))
            command = job["command"]
            self.assertIn("sepalith.training.sft.night_arm", command)
            self.assertEqual(Path(command[command.index("--recipe") + 1]), DEPLOY / "docs/training/plan-20260927/c10/bound/sft-v2-arm-b-lr1e-5.json")
            self.assertTrue(job["resumable"])
            self.assertEqual(job["cwd"], str(DEPLOY))
        short = jobs["C10-arm-b-lr1e-5"]["command"]
        self.assertEqual(int(short[short.index("--stop-after-gate") + 1]), 684)
        self.assertIn(684, GATES)
        self.assertNotIn("--stop-after-gate", jobs["C10-arm-b-lr1e-5-full"]["command"])


if __name__ == "__main__":
    unittest.main()
