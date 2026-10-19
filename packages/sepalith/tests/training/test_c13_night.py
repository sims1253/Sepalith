"""Card C13: the offline night driver's gate flow, R5 and the DPO guard, with a fake launcher (no GPU).

The fake launcher stands in for ``offline.trainer`` and ``offline.dev_generation``:
it writes checkpoints, telemetry and evaluator results, so the real decision
records, gate evidence (prepare_dev_gate / milestone_gate), overrides, R5
records and the DPO telemetry guard are exercised.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.training.eval import campaign_eval, milestone_gate
from sepalith.training.offline import night, recipe as offline_recipe, rules
from sepalith.training.sft import night_arm

REPO = Path(__file__).resolve().parents[4]
DEV250_SHA = "75f3a5ee216400fdfe1e2fb1fb63224374f418d1c248a1c278ae0d09be487f22"
BATCH = campaign_eval.generation_batching(32, "dynamic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    return json.loads(json.dumps(value))


def counts(valid=74, cap=1, edit=27, nofp=8, score=117, noedit=1):
    dev75 = {"protocol_valid": valid, "cap_hit": cap, "edit_exact": edit, "strict_noop_false_suggestions": nofp,
             "predicted_noop": 24, "strict_noop_correct": 24}
    dev250 = {"edit_exact": score - 38, "strict_noop_correct": 38, "predicted_noop": 38 + noedit}
    return dev75, dev250


def result(path, step, recipe_path, manifest, panel, values):
    value = {"schema": night_arm.RESULT_SCHEMA, "status": "complete", "step": step,
             "bound_recipe": {"path": str(recipe_path), "sha256": sha(recipe_path) if Path(recipe_path).is_file() else "0" * 64},
             "checkpoint": {"path": str(Path(manifest).parent), "campaign_manifest_sha256": sha(manifest)},
             "panel": {"path": panel["path"], "sha256": panel["sha256"]},
             "summary": {"denominators": {"cases": panel["cases"]}, "counts": values, "panel_sha256": panel["sha256"],
                         "generation_max_new_tokens": 1024, "generation_runtime": BATCH},
             "generation_runtime": {"dtype": "bf16", "cuda_visible_devices": "0", **BATCH}, "final_set_access": False}
    Path(path).write_text(json.dumps(value))


class FakeLauncher(night_arm.Launcher):
    def __init__(self, test, root):
        self.test, self.root = test, root
        self.calls, self.counts = [], {}
        self.chosen_log_ratio = 0.1
        self.chosen_log_ratio_at = None  # optional function of the update

    def run(self, label, module_args, *, budget, on_stop):
        module = module_args[0]
        self.calls.append((module, label))
        if module.endswith("offline.trainer"):
            return self.train(module_args)
        self.test.assertTrue(module.endswith("offline.dev_generation"), module)
        args = dict(zip(module_args[1::2], module_args[2::2], strict=False))
        recipe = json.loads(Path(args["--bound-recipe"]).read_text())
        panel = recipe["dev_gate"]["panels"][args["--panel"]]
        step = int(Path(args["--checkpoint"]).name.removeprefix("checkpoint-"))
        dev75, dev250 = self.counts[step]
        result(args["--output"], step, args["--bound-recipe"], Path(args["--checkpoint"]) / "campaign-manifest.json", panel,
               dev75 if args["--panel"] == "dev75" else dev250)
        return 0

    def train(self, module_args):
        recipe = json.loads(Path(module_args[module_args.index("--recipe") + 1]).read_text())
        start = int(Path(module_args[module_args.index("--resume") + 1]).name.removeprefix("checkpoint-")) if "--resume" in module_args else 0
        target = min(g for g in recipe["dev_gate"]["mandatory_steps"] if g > start)
        checkpoint = self.root / "archive" / "full" / f"checkpoint-{target}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "campaign-manifest.json").write_text(json.dumps({"step": target}))
        with (self.root / "archive" / "telemetry.jsonl").open("a") as stream:
            for step in range(start + 1, target + 1):
                ratio = self.chosen_log_ratio_at(step) if self.chosen_log_ratio_at is not None else self.chosen_log_ratio
                stream.write(json.dumps({"event": "dpo_update", "step": step, "chosen_log_ratio": ratio}) + "\n")
            stream.write(json.dumps({"event": "log", "step": target, "logs": {"loss": 0.4}}) + "\n")
        return 0


class OfflineDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.panels = {"dev75": {"path": "/panels/dev75.jsonl", "sha256": milestone_gate.PANEL_SHA, "cases": 75},
                       "dev250": {"path": "/panels/dev250.jsonl", "sha256": DEV250_SHA, "cases": 250}}
        parent = self.root / "start" / "checkpoint-536"
        parent.mkdir(parents=True)
        (parent / "campaign-manifest.json").write_text("{}")
        results = {}
        start75, start250 = counts()
        for name, values in (("dev75", start75), ("dev250", start250)):
            path = self.root / "start" / f"{name}.json"
            result(path, 536, self.root / "sft-recipe.json", parent / "campaign-manifest.json", self.panels[name], values)
            results[name] = {"path": str(path), "sha256": sha(path)}
        self.gates = [2, 4]
        self.recipe = plain({
            "schema": offline_recipe.BOUND_SCHEMA, "kind": "rft", "arm": {"name": "C13-rft-1-test", "learning_rate": 3.33333e-06},
            "parent": {"path": str(parent), "files": {"model.safetensors": "a" * 64}, "tag": "test"},
            "parent_reference": {"results": results, "metrics": rules.offline_metrics({"counts": start75}, {"counts": start250})},
            "outputs": {"archive": str(self.root / "archive"), "trainer": str(self.root / "runtime"),
                        "graceful_stop": str(self.root / "control" / "stop.json")},
            "dev_gate": {"mandatory_steps": self.gates, "evidence_directory": str(self.root / "gates"),
                         "panel": {"path": self.panels["dev75"]["path"], "sha256": self.panels["dev75"]["sha256"]},
                         "panels": self.panels, "generation_max_new_tokens": 1024,
                         "generation_batching": {"batch_size": 32, "kv_cache": "dynamic"}},
            "runtime": {"evaluation_steps": self.gates, "max_steps": 4},
            "source": {"repository_root": str(REPO)},
        })
        self.write()
        self.launcher = FakeLauncher(self, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.recipe_path = self.root / "recipe.json"
        self.recipe_path.write_text(json.dumps(self.recipe))

    def arm(self):
        settings = night_arm.Settings(recipe_path=self.recipe_path, guard=False, eval_budget_seconds=100.0, min_train_seconds=10.0,
                                      job_result=self.root / "job-result.json", clock=lambda: 1000.0)
        return night.OfflineArm(settings, lambda s, w, stop: self.launcher)

    def decision(self, step):
        return json.loads(self.arm().decisions(step)[-1].read_text())

    def r5(self):
        return json.loads((self.root / "gates" / "r5-decision.json").read_text())

    def test_start_metrics_come_from_the_bound_start_results(self):
        self.assertEqual(self.arm().start_metrics(),
                         {"valid": 74, "cap": 1, "edit": 27, "nofp": 8, "score250": 117, "noedit": 1, "edit_total": 106})

    def test_unchanged_first_gate_continues_and_r5_adopts_a_five_point_gain(self):
        self.launcher.counts = {2: counts(), 4: counts(score=122, edit=28)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        first = self.decision(2)
        self.assertEqual((first["outcome"], first["rule"]["kind"], first["decided_by"]),
                         ("continue", "offline_first_gate", "rule_R1_offline"))
        self.assertEqual(first["rule"]["inputs"]["start_model"]["score250"], 117)
        self.assertEqual(self.decision(4)["rule"]["inputs"]["previous"]["score250"], 117)
        milestone_gate.verify_gate_evidence(self.recipe, self.recipe_path, 2, self.root / "archive/full/checkpoint-2")
        record = self.r5()
        self.assertEqual((record["outcome"], record["result_step"]), ("adopt", 4))
        self.assertEqual(json.loads((self.root / "archive" / "arm-result.json").read_text())["status"], "r5_adopt")
        # A retry reuses the R5 record and runs nothing new.
        calls = len(self.launcher.calls)
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        self.assertEqual(len(self.launcher.calls), calls)

    def test_r5_band_one_to_four_stops_for_the_user_and_no_gain_discards(self):
        self.launcher.counts = {2: counts(), 4: counts(score=120)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        self.assertEqual(self.r5()["outcome"], "stop_for_user")
        self.assertIn("R5", json.loads((self.root / "job-result.json").read_text())["needs_user"])

    def test_r5_discards_when_nofp_rises_by_two(self):
        self.launcher.counts = {2: counts(), 4: counts(score=125, nofp=10)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)  # gate 4: nofp rise 2 vs gate 2 is R1 stop-for-user
        override = {"schema": night_arm.OVERRIDE_SCHEMA, "step": 4, "outcome": "continue", "bound_recipe_sha256": sha(self.recipe_path),
                    "reason": "test"}
        (self.root / "gates" / "override-step-4.json").write_text(json.dumps(override))
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        self.assertEqual(self.r5()["outcome"], "discard")

    def test_noedit_dodge_at_the_first_gate_stops_for_the_user(self):
        self.launcher.counts = {2: counts(noedit=4)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        rows = {row["name"]: row["outcome"] for row in self.decision(2)["rule"]["rows"]}
        self.assertEqual(rows["noedit_dodge"], "stop_for_user")
        self.assertFalse((self.root / "gates" / "r5-decision.json").exists())

    def test_rollback_at_the_last_gate_applies_r5_to_the_previous_gate(self):
        self.launcher.counts = {2: counts(score=123), 4: counts(score=110)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        self.assertEqual(self.decision(4)["outcome"], "rollback")
        record = self.r5()
        self.assertEqual((record["outcome"], record["result_step"]), ("adopt", 2))

    def test_rollback_at_the_first_gate_discards(self):
        self.launcher.counts = {2: counts(score=105)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        record = self.r5()
        self.assertEqual((record["outcome"], record["result_checkpoint"]), ("discard", None))

    def dpo(self):
        self.recipe["kind"] = "dpo"
        pairs = self.root / "pairs.jsonl"
        pairs.write_text("{}\n")
        logps = self.root / "ref" / "reference-logps.jsonl"
        logps.parent.mkdir()
        logps.write_text("{}\n")
        self.recipe["cohort"] = {"rows": {"path": str(pairs), "sha256": sha(pairs)}}
        self.recipe["reference"] = {"logps": str(logps), "manifest": str(self.root / "ref" / "reference-manifest.json")}
        self.recipe["pre_registered"] = {"beta": 0.1}
        manifest = {"schema": offline_recipe.REFERENCE_SCHEMA, "status": "complete", "parent_weights_sha256": "a" * 64,
                    "pairs_sha256": sha(pairs), "logps_sha256": sha(logps)}
        self.write()
        return manifest

    def test_dpo_needs_its_reference_and_stops_on_a_chosen_collapse(self):
        manifest = self.dpo()
        with self.assertRaisesRegex(night_arm.DriverError, "run the reference job first"):
            self.arm().run()
        Path(self.recipe["reference"]["manifest"]).write_text(json.dumps(manifest))
        self.gates = [200, 400]
        self.recipe["dev_gate"]["mandatory_steps"] = self.recipe["runtime"]["evaluation_steps"] = self.gates
        self.recipe["runtime"]["max_steps"] = 400
        self.write()
        self.launcher.chosen_log_ratio_at = lambda step: 0.5 - step / 100  # falls steadily below zero
        self.launcher.counts = {200: counts()}
        self.assertEqual(self.arm().run(), night_arm.EXIT_NEEDS_USER)
        rows = {row["name"]: row["outcome"] for row in self.decision(200)["rule"]["rows"]}
        self.assertEqual(rows["dpo_chosen_collapse"], "stop_for_user")

    def test_dpo_with_healthy_chosen_log_ratios_continues(self):
        manifest = self.dpo()
        Path(self.recipe["reference"]["manifest"]).write_text(json.dumps(manifest))
        self.launcher.counts = {2: counts(), 4: counts(score=124)}
        self.assertEqual(self.arm().run(), night_arm.EXIT_DONE)
        self.assertEqual(self.r5()["outcome"], "adopt")


if __name__ == "__main__":
    unittest.main()
