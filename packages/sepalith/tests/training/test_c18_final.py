"""C18 user-stop end: DEV250 of the user's checkpoint through the driver's evaluator, the pilot summary, the verdict."""
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.training.grpo_exec import final, jobs, night, rules
from sepalith.training.sft import night_arm

from test_c18_grpo_exec import FakeLauncher, dev_counts, driver_recipe


class UserStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recipe_path = driver_recipe(self.root)
        self.launcher = FakeLauncher(self)
        self.launcher.cpt.update({50: 1.012, 100: 1.024})
        self.launcher.val.update({50: (0.31, 0.40), 100: (0.32, 0.42)})
        self.launcher.dev[100] = dev_counts(score=106)
        self.clock = {"now": 1000.0}

        def hook(target):  # the pilot reaches update 100 and stops for the user on the CPT row
            if target >= 100:
                self.clock["now"] = 10**7
        self.launcher.on_train = hook

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, deadline=10**6 + 1000.0):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, eval_budget_seconds=100.0, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: self.clock["now"], deadline=deadline)
        return night.ExecGRPOArm(settings, lambda s, w, stop: self.launcher, verify_files=False)

    def reach_100(self):
        self.arm().run()  # trains to 50 (continue), to 100; the deadline defers the evaluation
        self.clock["now"] = 1000.0
        self.launcher.on_train = None
        self.assertEqual(self.arm().gate_step(100), rules.STOP_FOR_USER)
        arm = self.arm()
        (arm.evidence_dir / "user-stop-step-100.json").write_text(json.dumps({
            "schema": final.USER_STOP_SCHEMA, "step": 100, "outcome": "stop_and_evaluate", "reason": "user: stop at 100",
            "selected_checkpoint": str(arm.checkpoint(100))}))
        return arm

    def test_dev250_runs_once_and_the_summary_feeds_the_verdict(self):
        arm = self.reach_100()
        with self.assertRaises(night_arm.DriverError):
            final.summary(arm, 100)  # no DEV250 yet
        self.assertEqual(final.run_dev250(arm, 100), night_arm.EXIT_DONE)
        dev = [a for m, a in self.launcher.calls if m == night.DEV_EVALUATOR and a[a.index("--panel") + 1] == "dev250"]
        self.assertEqual(len(dev), 1)
        self.assertTrue(dev[0][dev[0].index("--checkpoint") + 1].endswith("checkpoint-100"))
        summary = json.loads((arm.archive / "pilot-summary.json").read_text())
        self.assertEqual((summary["status"], summary["selected"]["step"]), ("user_stop", 100))
        self.assertIn("not the pre-registered", summary["selection_rule"])
        self.assertEqual(summary["selected"]["dev250"]["score250"], 106)
        self.assertAlmostEqual(summary["selected"]["cpt_rise_vs_start"], 0.024)
        self.assertEqual(final.run_dev250(self.arm(), 100), night_arm.EXIT_DONE)  # idempotent: no second evaluation
        self.assertEqual(len([a for m, a in self.launcher.calls if m == night.DEV_EVALUATOR and "dev250" in a]), 1)
        metrics = self.root / "metrics.json"
        metrics.write_text(json.dumps({
            jobs.START_LABEL: {"overall": {"protocol_valid": 0.973}},
            jobs.CANDIDATE_LABEL: {"overall": {"protocol_valid": 0.99}, f"vs_{jobs.START_LABEL}": {
                "all_pass": {"mean": 0.02, "ci95": [0.008, 0.03]}, "reward": {"mean": 0.03, "ci95": [0.01, 0.05]},
                "exact_match": {"mean": 0.01, "ci95": [0.0, 0.02]}}}}))
        verdict = json.loads(json.dumps(jobs.verdict(arm.archive / "pilot-summary.json", metrics)))
        self.assertTrue(verdict["primary"]["met"])
        self.assertTrue(verdict["secondary"]["dev250_score250"]["non_inferior"])
        self.assertTrue(verdict["adopt"])

    def test_dev250_defers_near_the_deadline_and_requires_the_user_decision(self):
        arm = self.reach_100()
        late = self.arm(deadline=1000.0 + 60)
        self.assertEqual(final.run_dev250(late, 100), night_arm.EXIT_DEFERRED)
        (arm.evidence_dir / "user-stop-step-100.json").unlink()
        with self.assertRaises(night_arm.DriverError):
            final.run_dev250(self.arm(), 100)

    def test_job_file(self):
        job = json.loads(json.dumps(final.dev250_job(Path("/x/wt"), Path("/x/recipe.json"), 100, Path("/x/py"))))
        from sepalith.ops import night_queue

        night_queue.validate_job(job)
        self.assertIn("PYTHONPATH=/x/wt/packages/sepalith/src", job["command"])
        self.assertEqual(job["command"][-5:], ["dev250", "--recipe", "/x/recipe.json", "--step", "100"])


if __name__ == "__main__":
    unittest.main()
