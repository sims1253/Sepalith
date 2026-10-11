"""Card C04: rule R1 on synthetic metrics, and the night driver's gate flow.

The driver tests replace the trainer and DEV-generation processes with a fake
launcher that writes checkpoints and evaluator results, so they exercise the
real decision records, gate evidence (prepare_dev_gate / milestone_gate),
override precedence and deadline deferral without a GPU.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.training.eval import milestone_gate
from sepalith.training.sft import gate_rule as R1
from sepalith.training.sft import night_arm
from sepalith.training.sft.recipe_v2 import BOUND_SCHEMA

REPO = Path(__file__).resolve().parents[4]
DEV250_SHA = "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22"


def metrics(valid=70, cap=0, edit=20, nofp=3, score250=120, loss=0.5):
    return {"valid": valid, "cap": cap, "edit": edit, "nofp": nofp, "score250": score250, "loss": loss}


class RuleR1Tests(unittest.TestCase):
    parent = {"valid": 72, "cap": 1, "edit": 12, "nofp": 2, "score250": 90}

    def test_first_gate_continue_and_each_floor(self):
        self.assertEqual(R1.apply_r1(metrics(), first=True, reference=self.parent)["outcome"], R1.CONTINUE)
        for current in (metrics(valid=59), metrics(cap=3), metrics(edit=12), metrics(loss=float("nan")), metrics(loss=None)):
            result = R1.apply_r1(current, first=True, reference=self.parent)
            self.assertEqual(result["outcome"], R1.STOP_FOR_USER, current)
        # The boundaries themselves pass.
        self.assertEqual(R1.apply_r1(metrics(valid=60, cap=2, edit=13), first=True, reference=self.parent)["outcome"], R1.CONTINUE)

    def test_later_gate_bands(self):
        previous = metrics(valid=72, cap=0, edit=25, nofp=2, score250=150)
        cases = [
            ({}, R1.CONTINUE),
            ({"edit": 23}, R1.CONTINUE), ({"edit": 22}, R1.STOP_FOR_USER), ({"edit": 18}, R1.STOP_FOR_USER), ({"edit": 17}, R1.ROLLBACK),
            ({"nofp": 3}, R1.CONTINUE), ({"nofp": 4}, R1.STOP_FOR_USER), ({"nofp": 5}, R1.STOP_FOR_USER), ({"nofp": 6}, R1.ROLLBACK),
            ({"valid": 71}, R1.CONTINUE), ({"valid": 70}, R1.STOP_FOR_USER), ({"valid": 68}, R1.STOP_FOR_USER),
            ({"valid": 67}, R1.ROLLBACK),
            ({"cap": 1}, R1.CONTINUE), ({"cap": 2}, R1.STOP_FOR_USER), ({"cap": 4}, R1.STOP_FOR_USER), ({"cap": 5}, R1.ROLLBACK),
            ({"score250": 146}, R1.CONTINUE), ({"score250": 145}, R1.STOP_FOR_USER), ({"score250": 141}, R1.STOP_FOR_USER),
            ({"score250": 140}, R1.ROLLBACK),
            ({"edit": 40, "score250": 200, "valid": 75}, R1.CONTINUE),  # improvements never stop
            ({"edit": 22, "cap": 5}, R1.ROLLBACK),  # the worst row wins
            ({"loss": float("inf")}, R1.STOP_FOR_USER),
        ]
        for change, expected in cases:
            current = dict(previous, **change)
            self.assertEqual(R1.apply_r1(current, first=False, reference=previous)["outcome"], expected, change)

    def test_valid_below_68_is_rollback_even_without_a_drop(self):
        previous = metrics(valid=64)
        self.assertEqual(R1.apply_r1(metrics(valid=66), first=False, reference=previous)["outcome"], R1.ROLLBACK)

    def test_metrics_from_evaluator_summaries(self):
        dev75 = {"counts": {"protocol_valid": 70, "cap_hit": 1, "edit_exact": 21, "strict_noop_false_suggestions": 4}}
        dev250 = {"counts": {"edit_exact": 100, "strict_noop_correct": 41}}
        self.assertEqual(R1.metrics_from_summaries(dev75, dev250), {"valid": 70, "cap": 1, "edit": 21, "nofp": 4, "score250": 141})
        with self.assertRaises(ValueError):
            R1.metrics_from_summaries(dict(dev75, counts=dict(dev75["counts"], cap_hit=-1)), dev250)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeLauncher(night_arm.Launcher):
    """Stands in for the trainer and DEV evaluator processes."""

    def __init__(self, test, arm_root):
        self.test, self.root = test, arm_root
        self.after_train = None     # hook run after each simulated trainer exit
        self.after_run = None       # hook run after each simulated process
        self.calls = []
        self.counts = {}            # (step, panel) -> counts dict
        self.stop_at = None         # simulated deadline: trainer stops at this update
        self.fail_panel = None
        self.loss = 0.5

    def run(self, label, module_args, *, budget, on_stop):
        code = self._run(label, module_args)
        if self.after_run:
            self.after_run(label, module_args)
        return code

    def _run(self, label, module_args):
        module = module_args[0]
        args = dict(zip(module_args[1::2], module_args[2::2], strict=False)) if module_args[1] != "run" else {}
        self.calls.append((module, label))
        if module.endswith("full_weight_edit_sft"):
            code = self.train(module_args)
            if self.after_train:
                self.after_train()
            return code
        output = Path(args["--output"])
        panel = args.get("--panel", "dev75")
        if panel == self.fail_panel:
            return 1
        step = 0 if module.endswith("parent_baseline") else int(Path(args["--checkpoint"]).name.removeprefix("checkpoint-"))
        recipe = json.loads(Path(args["--bound-recipe"]).read_text())
        bound = recipe["dev_gate"]["panels"][panel]
        counts = self.counts[(step, panel)]
        full = {"protocol_valid": 0, "cap_hit": 0, "edit_exact": 0, "strict_noop_false_suggestions": 0, "strict_noop_correct": 0}
        full.update(counts)
        manifest = Path(args["--checkpoint"]) / "campaign-manifest.json"
        result = {"schema": night_arm.RESULT_SCHEMA, "status": "complete", "step": step,
                  "bound_recipe": {"path": args["--bound-recipe"], "sha256": sha(args["--bound-recipe"])},
                  "checkpoint": {"path": args["--checkpoint"], "campaign_manifest_sha256": sha(manifest)},
                  "panel": {"path": bound["path"], "sha256": bound["sha256"]},
                  "summary": {"denominators": {"cases": bound["cases"]}, "counts": full, "panel_sha256": bound["sha256"],
                              "generation_max_new_tokens": 1024, "target_nll": 0.3},
                  "final_set_access": False}
        output.write_text(json.dumps(result))
        return 0

    def train(self, module_args):
        recipe = json.loads(Path(module_args[module_args.index("--recipe") + 1]).read_text())
        start = 0
        if "--resume" in module_args:
            start = int(Path(module_args[module_args.index("--resume") + 1]).name.removeprefix("checkpoint-"))
        else:
            self.test.assertFalse((self.root / "archive").exists(), "fresh run needs a fresh archive")
        gates = recipe["dev_gate"]["mandatory_steps"]
        target = min(g for g in gates if g > start)
        if "--stop-after-step" in module_args:
            target = min(target, int(module_args[module_args.index("--stop-after-step") + 1]))
        if self.stop_at is not None and start < self.stop_at < target:
            target = self.stop_at
        checkpoint = self.root / "archive" / "full" / f"checkpoint-{target}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "campaign-manifest.json").write_text(json.dumps({"step": target}))
        with (self.root / "archive" / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "log", "step": target, "logs": {"loss": self.loss}}) + "\n")
        return 0


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        panels = {}
        for name, cases, digest in (("dev75", 75, milestone_gate.PANEL_SHA), ("dev250", 250, DEV250_SHA)):
            panels[name] = {"path": f"/panels/{name}.jsonl", "sha256": digest, "cases": cases}
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "campaign-manifest.json").write_text("{}")
        self.recipe = {
            "schema": BOUND_SCHEMA, "arm": {"name": "lr1e-5", "learning_rate": 1e-5},
            "parent": {"path": str(parent), "files": {"model.safetensors": "a" * 64}},
            "outputs": {"archive": str(self.root / "archive"), "trainer": str(self.root / "runtime"),
                        "graceful_stop": str(self.root / "control" / "stop.json")},
            "dev_gate": {"mandatory_steps": [2, 4, 6, 8], "evidence_directory": str(self.root / "gates"),
                         "panel": {"path": panels["dev75"]["path"], "sha256": panels["dev75"]["sha256"]},
                         "panels": panels, "parent_baseline": {"path": str(self.root / "baseline" / "baseline.json")},
                         "generation_max_new_tokens": 1024},
            "runtime": {"evaluation_steps": [2, 4, 6, 8], "max_steps": 8},
            "source": {"repository_root": str(REPO)},
        }
        self.recipe_path = self.root / "recipe.json"
        self.recipe_path.write_text(json.dumps(self.recipe))
        self.clock = [1000.0]

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, deadline=None, stop_after_gate=None, eval_budget=100.0):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, deadline=deadline,
                                      stop_after_gate=stop_after_gate, eval_budget_seconds=eval_budget,
                                      min_train_seconds=10.0, job_result=self.root / "job-result.json",
                                      clock=lambda: self.clock[0])
        if not hasattr(self, "launcher"):
            self.launcher = FakeLauncher(self, self.root)
        launcher = self.launcher
        return night_arm.Arm(settings, lambda s, w, stop: launcher)

    def scores(self, step, valid=70, cap=0, edit=20, nofp=2, score=120):
        self.launcher.counts[(step, "dev75")] = {"protocol_valid": valid, "cap_hit": cap, "edit_exact": edit,
                                                 "strict_noop_false_suggestions": nofp}
        self.launcher.counts[(step, "dev250")] = {"edit_exact": score - 40, "strict_noop_correct": 40}

    def baseline(self, edit=10):
        arm = self.arm()
        self.scores(0, edit=edit)
        self.assertEqual(night_arm.run_baseline(arm), night_arm.EXIT_DONE)
        self.assertEqual(json.loads((self.root / "baseline" / "baseline.json").read_text())["metrics"]["edit"], edit)

    def decision(self, step):
        return json.loads(self.arm().decisions(step)[-1].read_text())

    def test_full_arm_continues_through_every_gate_with_evidence(self):
        self.baseline()
        for step, edit in ((2, 20), (4, 21), (6, 22), (8, 22)):
            self.scores(step, edit=edit, score=120 + step)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        for step in (2, 4, 6, 8):
            self.assertTrue((self.root / "gates" / f"dev-gate-step-{step}.json").is_file())
            record = self.decision(step)
            self.assertEqual((record["status"], record["continue_training"], record["decided_by"]),
                             ("admitted_for_continuation", True, "rule_R1"))
            self.assertEqual(record["rule"]["kind"], "first_gate" if step == 2 else "later_gate")
        self.assertEqual(self.decision(2)["rule"]["inputs"]["parent"]["edit"], 10)
        self.assertEqual(self.decision(4)["rule"]["inputs"]["previous"]["edit"], 20)
        # The evidence is exactly what the trainer's continuation_plan verifies before resuming.
        milestone_gate.verify_gate_evidence(self.recipe, self.recipe_path, 4, self.root / "archive/full/checkpoint-4")
        result = json.loads((self.root / "archive" / "arm-result.json").read_text())
        self.assertEqual((result["status"], result["gate"]), ("complete", 8))
        job = json.loads((self.root / "job-result.json").read_text())
        self.assertEqual(len(job["rule_decisions"]), 4)

    def test_first_gate_without_improvement_stops_for_user_then_override_continues(self):
        self.baseline(edit=20)
        self.scores(2, edit=20)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        record = self.decision(2)
        self.assertEqual((record["outcome"], record["continue_training"]), ("stop_for_user", False))
        self.assertFalse((self.root / "gates" / "dev-gate-step-2.json").exists())
        self.assertIn("override-step-2.json", json.loads((self.root / "job-result.json").read_text())["needs_user"])
        # A retry without an override stays stopped and does not re-evaluate.
        calls = len(self.launcher.calls)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertEqual(len(self.launcher.calls), calls)
        # The human override takes precedence over the rule.
        override = {"schema": night_arm.OVERRIDE_SCHEMA, "step": 2, "outcome": "continue",
                    "bound_recipe_sha256": sha(self.recipe_path), "reason": "accept: loss curve is healthy", "author": "user"}
        (self.root / "gates" / "override-step-2.json").write_text(json.dumps(override))
        for step in (4, 6, 8):
            self.scores(step, edit=20)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        record = json.loads(self.arm().decisions(2)[-1].read_text())
        self.assertEqual((record["outcome"], record["rule_outcome"], record["decided_by"]),
                         ("continue", "stop_for_user", "human_override"))
        self.assertEqual(len(self.arm().decisions(2)), 2)

    def test_override_wins_over_a_passing_rule_and_wrong_recipe_override_is_refused(self):
        self.baseline()
        self.scores(2, edit=30)
        (self.root / "gates").mkdir()
        bad = {"schema": night_arm.OVERRIDE_SCHEMA, "step": 2, "outcome": "rollback", "bound_recipe_sha256": "0" * 64,
               "reason": "x"}
        (self.root / "gates" / "override-step-2.json").write_text(json.dumps(bad))
        arm = self.arm()
        with self.assertRaisesRegex(night_arm.DriverError, "another recipe"):
            arm.run()
        bad.update(bound_recipe_sha256=sha(self.recipe_path), outcome="stop_for_user")
        (self.root / "gates" / "override-step-2.json").write_text(json.dumps(bad))
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        record = self.decision(2)
        self.assertEqual((record["rule_outcome"], record["outcome"], record["decided_by"]),
                         ("continue", "stop_for_user", "human_override"))

    def test_later_gate_rollback_names_the_last_good_checkpoint(self):
        self.baseline()
        self.scores(2, edit=25)
        self.scores(4, edit=15)  # edit drop 10 -> rollback
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        result = json.loads((self.root / "archive" / "arm-result.json").read_text())
        self.assertEqual((result["status"], result["gate"]), ("rolled_back", 4))
        self.assertEqual(result["result_checkpoint"], str(self.root / "archive/full/checkpoint-2"))
        self.assertEqual(self.decision(4)["last_good_checkpoint"], result["result_checkpoint"])

    def test_pilot_stops_at_its_first_gate_and_the_winner_resumes(self):
        self.baseline()
        self.scores(2)
        self.assertEqual(self.arm(stop_after_gate=2).run(), night_arm.EXIT_DONE)
        self.assertEqual(json.loads((self.root / "archive" / "arm-result.json").read_text())["status"], "pilot_gate_passed")
        self.assertEqual(self.launcher.calls[-1][0], "sepalith.training.eval.run_dev_generation")
        for step in (4, 6, 8):
            self.scores(step)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        train_calls = [label for module, label in self.launcher.calls if module.endswith("full_weight_edit_sft")]
        self.assertEqual(train_calls, ["train-from-0", "train-from-2", "train-from-4", "train-from-6"])

    def test_pilot_failing_the_floor_finishes_without_needing_the_user(self):
        self.baseline(edit=30)
        self.scores(2, edit=20)
        self.assertEqual(self.arm(stop_after_gate=2).run(), night_arm.EXIT_DONE)
        self.assertEqual(json.loads((self.root / "archive" / "arm-result.json").read_text())["status"], "pilot_gate_failed_r1")
        self.assertIsNone(json.loads((self.root / "job-result.json").read_text())["needs_user"])

    def test_deadline_defers_training_and_gate_evaluation(self):
        self.baseline()
        self.scores(2)
        self.launcher.stop_at = 1
        self.clock[0] = 1000.0
        deadline = 1005.0
        # Trainer saves at update 1 when the deadline passes; the driver exits resumable.
        arm = self.arm(deadline=deadline)

        def expire():
            self.clock[0] = deadline + 1
        self.launcher.after_train = expire
        self.clock[0] = deadline - 20
        self.assertEqual(arm.run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(arm.durable_steps(), [1])
        # Next night: resume to the gate, but too little time remains to evaluate it.
        self.launcher.after_train = None
        self.launcher.stop_at = None
        self.clock[0] = 2000.0
        self.assertEqual(self.arm(deadline=2050.0, eval_budget=100.0).run(), night_arm.EXIT_DEFERRED)
        self.assertEqual(self.arm().durable_steps(), [1, 2])
        self.assertEqual(self.arm().decisions(2), [])
        # The night after: the gate is evaluated first, then training continues.
        for step in (4, 6, 8):
            self.scores(step)
        self.clock[0] = 3000.0
        self.assertEqual(self.arm(deadline=10_000.0).run(), night_arm.EXIT_DONE)
        resumed = [label for module, label in self.launcher.calls if module.endswith("full_weight_edit_sft")]
        self.assertEqual(resumed, ["train-from-0", "train-from-1", "train-from-2", "train-from-4", "train-from-6"])

    def test_evaluator_crash_is_stop_for_user_and_retry_re_evaluates(self):
        self.baseline()
        self.scores(2)
        self.launcher.fail_panel = "dev250"
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        record = self.decision(2)
        self.assertFalse(record["evaluation_complete"])
        self.assertIn("evaluator crash", record["rule"]["problems"][0])
        self.launcher.fail_panel = None
        for step in (4, 6, 8):
            self.scores(step)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)

    def test_pilot_evaluator_crash_still_needs_the_user(self):
        self.baseline()
        self.scores(2)
        self.launcher.fail_panel = "dev75"
        self.assertEqual(self.arm(stop_after_gate=2).run(), night_arm.EXIT_NEEDS_USER)

    def test_wrong_panel_hash_in_a_result_is_stop_for_user(self):
        self.baseline()
        self.scores(2)

        def tamper(label, module_args):
            if "--output" in module_args and "dev250" in label:
                path = Path(module_args[module_args.index("--output") + 1])
                value = json.loads(path.read_text())
                value["panel"]["sha256"] = "0" * 64
                path.write_text(json.dumps(value))
        self.launcher.after_run = tamper
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertIn("dev250 result panel binding differs", self.decision(2)["rule"]["problems"][0])

    def test_missing_baseline_is_stop_for_user(self):
        self.launcher = FakeLauncher(self, self.root)
        self.scores(2)
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertIn("parent baseline missing", self.decision(2)["rule"]["problems"][0])

    def test_sigterm_during_training_writes_the_recipe_bound_stop_file(self):
        arm = self.arm()
        arm.request_trainer_stop(None)
        self.assertEqual(json.loads((self.root / "control" / "stop.json").read_text()),
                         {"action": "save_and_stop", "bound_recipe_sha256": sha(self.recipe_path)})
        arm.request_trainer_stop(None)  # idempotent
        arm.clear_stale_stop_file()
        self.assertFalse((self.root / "control" / "stop.json").exists())


if __name__ == "__main__":
    unittest.main()
