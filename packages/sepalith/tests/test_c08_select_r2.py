"""Card C08 rule R2 selector (docs/training/plan-20260927/c08/select_r2.py).

Covers every R2 branch on synthetic decisions, the binding checks, the
continuation job, the real gate-268 numbers of the two finished arms, and the
night runner picking up the job that the selector enqueues during the night.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import ModuleType

from sepalith.ops.night_queue import Queue, validate_job
from sepalith.ops.night_runner import Clock, NightRunner, Probes, Window

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "docs/training/plan-20260927/c08/select_r2.py"
SELECT_JOB = REPO / "docs/training/plan-20260927/c08/jobs/C08-r2-select.json"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c08_select_r2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r2 = _load()

PARENT = {"cap": 64, "edit": 0, "nofp": 0, "score250": 0, "valid": 0}
# The real gate-268 metrics of the two arms that finished on 2026-10-12.
REAL = {
    "lr1e-5": {"cap": 2, "edit": 26, "loss": 0.7168424129486084, "nofp": 11, "score250": 110, "valid": 73},
    "lr3e-6": {"cap": 1, "edit": 26, "loss": 0.6943989992141724, "nofp": 10, "score250": 104, "valid": 74},
}
LR = {"lr3e-6": 3e-06, "lr1e-5": 1e-05, "lr3e-5": 3e-05}
PRIORITY = {"lr1e-5": 20, "lr3e-6": 21, "lr3e-5": 22}
LIVE_STATE = Path.home() / ".local/state/sepalith"
LIVE_GATES = LIVE_STATE / "campaign-20260915/evaluation-gates"
LIVE_CHECKPOINTS = Path("/mnt/e/sepalith/campaign-20260915/checkpoints")


def metrics(score250, valid=74, cap=1, edit=26, nofp=10, loss=0.7):
    return {"cap": cap, "edit": edit, "loss": loss, "nofp": nofp, "score250": score250, "valid": valid}


def arm(name, score250, floor=(), lr=None, cpt=None):
    return {"arm": name, "learning_rate": LR.get(name, lr) if lr is None else lr, "score250": score250,
            "floor_failures": list(floor), "cpt_2k_validation_loss": cpt}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n" if not isinstance(value, str) else value)
    return path


# A stand-in for `night_arm arm`. With --stop-after-gate it plays the pilot arm
# according to <root>/arm-mode.json; without it, it is the C09 continuation and
# records its argv.
FAKE_ARM = """
import json, pathlib, shutil, sys
argv = sys.argv[1:]
root = pathlib.Path(__file__).resolve().parent
recipe = pathlib.Path(argv[argv.index("--recipe") + 1])
name = recipe.stem.removeprefix("sft-v2-")
if "--stop-after-gate" not in argv:
    (root / f"continued-{name}.json").write_text(json.dumps(argv))
    sys.exit(0)
mode = json.loads((root / "arm-mode.json").read_text()).get(name, "done")
if mode == "defer":
    sys.exit(75)
if mode == "fail":
    sys.exit(1)
staged = root / "staged" / name
if staged.is_dir():
    for path in staged.rglob("*"):
        if path.is_file():
            target = pathlib.Path("/" + str(path.relative_to(staged)))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.rmtree(staged)
sys.exit(0)
"""


class Tree:
    """A synthetic checkout, campaign state and night queue for the three pilot arms."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.queue_root = root / "state/night-queue"
        self.queue = Queue(self.queue_root)
        self.receipt = root / "state/campaign-20260915/C08-r2-selection.json"
        self.fake_arm = write(root / "fake_arm.py", textwrap.dedent(FAKE_ARM))
        write(root / "arm-mode.json", {})

    def recipe_path(self, lr):
        return self.repo / f"docs/training/plan-20260927/c04/bound/sft-v2-{lr}.json"

    def evidence(self, lr):
        return self.root / f"gates/SFT-v2-r4-{lr}"

    def archive(self, lr):
        return self.root / f"checkpoints/SFT-v2-r4-{lr}"

    def job(self, lr):
        return json.loads((self.repo / f"docs/training/plan-20260927/c04/jobs/C08-arm-{lr}.json").read_text())

    def decision_path(self, lr, attempt=1):
        return self.evidence(lr) / f"decisions/decision-step-268-{attempt}.json"

    def add_arm(self, lr, values, *, state="done", staged=False, parent=None):
        """Recipe, C08 job, queue entry, checkpoint, results and (unless staged) the decision."""
        parent = dict(PARENT if parent is None else parent)
        recipe = {"schema": r2.BOUND_SCHEMA, "arm": {"name": lr, "learning_rate": LR[lr]},
                  "parent": {"step": 11586}, "seed": 3407, "data_queue": {"schedule": "v2"},
                  "pre_registered": {}, "cohort": {}, "development": {},
                  "outputs": {"archive": str(self.archive(lr))},
                  "dev_gate": {"evidence_directory": str(self.evidence(lr)), "mandatory_steps": [268, 536, 803, 1071]}}
        recipe_path = write(self.recipe_path(lr), recipe)
        job = {"id": f"C08-arm-{lr}", "cwd": str(self.root), "env": [], "priority": PRIORITY[lr], "resumable": True,
               "estimated_minutes": 120, "requires": ["C07-baseline"],
               "command": [sys.executable, "-B", str(self.fake_arm), "arm", "--recipe", str(recipe_path),
                           "--stop-after-gate", "268"], "note": f"C08 pilot {lr}"}
        write(self.repo / f"docs/training/plan-20260927/c04/jobs/C08-arm-{lr}.json", job)
        self.queue.enqueue(job, environ={})
        if state != "pending":
            self.queue.move(job["id"], "pending", state)
        manifest = write(self.archive(lr) / "full/checkpoint-268/campaign-manifest.json", f"manifest {lr}\n")
        results = {panel: write(self.evidence(lr) / f"results/dev-{panel}-step-268.json", {"panel": panel, "arm": lr})
                   for panel in ("dev75", "dev250")}
        floor = r2.first_gate_floor(values, parent["edit"])
        outcome = r2.CONTINUE if not floor else r2.STOP_FOR_USER
        decision = {
            "schema": r2.DECISION_SCHEMA, "status": "admitted_for_continuation", "step": 268, "outcome": outcome,
            "rule_outcome": outcome, "decided_by": "rule_R1", "override": None, "evaluation_complete": True,
            "rule": {"rule": "R1", "kind": "first_gate", "outcome": outcome, "rows": [], "problems": [],
                     "inputs": {"current": dict(values), "parent": parent}},
            "metrics": dict(values), "bound_recipe_sha256": sha(recipe_path), "checkpoint_manifest_sha256": sha(manifest),
            "panel_sha256": "7c" * 32, "dev250": {"sha256": "75" * 32, "cases": 250},
            "generation_runtime": {"batch_size": 32, "dtype": "bf16", "max_new_tokens": 1024},
            "results": {panel: {"path": str(path), "sha256": sha(path)} for panel, path in results.items()},
            "at": "2026-10-13T01:20:00+0200",
        }
        files = {self.decision_path(lr): decision}
        if outcome == r2.CONTINUE:
            files[self.evidence(lr) / "dev-gate-step-268.json"] = {"step": 268}
        for path, value in files.items():
            write(self.root / "staged" / lr / path.relative_to("/") if staged else path, value)
        return decision

    def standard(self, third=None, *, third_state="done", staged=False):
        self.add_arm("lr1e-5", REAL["lr1e-5"])
        self.add_arm("lr3e-6", REAL["lr3e-6"])
        self.add_arm("lr3e-5", third or metrics(100), state=third_state, staged=staged)

    def edit_decision(self, lr, change):
        path = self.decision_path(lr)
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def select(self, *, dry_run=False, environ=None):
        return r2.select(self.repo, self.queue_root, self.receipt, None, dry_run=dry_run, environ=environ or {})

    def main(self, *extra, environ=None):
        argv = ["--repo-root", str(self.repo), "--queue-root", str(self.queue_root), "--receipt", str(self.receipt), *extra]
        saved = dict(os.environ)
        os.environ.update(environ or {})
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = r2.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        return code, err.getvalue()


class RuleTests(unittest.TestCase):
    def test_r1_first_gate_floor(self):
        self.assertEqual(r2.first_gate_floor(REAL["lr1e-5"], 0), [])
        self.assertEqual(r2.first_gate_floor(metrics(100, valid=60, cap=2, edit=1), 0), [])
        self.assertIn("valid 59 < 60", r2.first_gate_floor(metrics(100, valid=59), 0))
        self.assertIn("cap 3 > 2", r2.first_gate_floor(metrics(100, cap=3), 0))
        self.assertIn("edit 5 does not beat parent edit 5", r2.first_gate_floor(metrics(100, edit=5), 5))
        self.assertTrue(r2.first_gate_floor(metrics(100, loss=float("nan")), 0))
        self.assertTrue(r2.first_gate_floor(metrics(100, loss=None), 0))
        self.assertTrue(r2.first_gate_floor({**metrics(100), "score250": None}, 0))
        self.assertTrue(r2.first_gate_floor(metrics(100), None))

    def test_step2_highest_score_outside_the_band(self):
        result = r2.apply_r2([arm("lr3e-6", 104), arm("lr1e-5", 110), arm("lr3e-5", 90)])
        self.assertEqual((result["outcome"], result["winner"]), ("continue", "lr1e-5"))
        self.assertEqual(result["steps"][-1]["applies"], False)
        self.assertEqual(result["steps"][-1]["gap"], 6)

    def test_step3_within_five_picks_the_lower_learning_rate(self):
        result = r2.apply_r2([arm("lr3e-6", 104), arm("lr1e-5", 109), arm("lr3e-5", 90)])  # gap 5: inside
        self.assertEqual(result["winner"], "lr3e-6")
        result = r2.apply_r2([arm("lr3e-6", 104), arm("lr1e-5", 110), arm("lr3e-5", 113)])  # 113 vs 110
        self.assertEqual(result["winner"], "lr1e-5")
        self.assertEqual(result["steps"][-1]["band"], ["lr3e-5", "lr1e-5"])

    def test_step3_uses_the_top_two_only(self):
        # lr3e-6 is within 5 of the top but ranks third, so it is not in "the top two".
        result = r2.apply_r2([arm("lr3e-5", 110), arm("lr1e-5", 107), arm("lr3e-6", 105)])
        self.assertEqual(result["winner"], "lr1e-5")
        # A tie for second place keeps both tied arms in the band.
        result = r2.apply_r2([arm("lr3e-5", 110), arm("lr1e-5", 106), arm("lr3e-6", 106)])
        self.assertEqual(result["winner"], "lr3e-6")

    def test_step1_discards_floor_failures(self):
        result = r2.apply_r2([arm("lr3e-6", 104), arm("lr1e-5", 110), arm("lr3e-5", 140, floor=["cap 5 > 2"])])
        self.assertEqual(result["winner"], "lr1e-5")
        self.assertEqual(result["steps"][0]["discarded"], {"lr3e-5": ["cap 5 > 2"]})
        result = r2.apply_r2([arm("lr3e-6", 104, floor=["x"]), arm("lr1e-5", 110, floor=["y"]), arm("lr3e-5", 90)])
        self.assertEqual(result["winner"], "lr3e-5")
        self.assertIn("only arm", result["reason"])

    def test_step5_no_survivor_stops_for_the_user(self):
        result = r2.apply_r2([arm(name, 100, floor=["valid 50 < 60"]) for name in LR])
        self.assertEqual((result["outcome"], result["winner"]), ("stop_for_user", None))
        self.assertEqual(result["steps"][-1]["step"], 5)

    def test_step4_still_tied_needs_the_cpt_loss(self):
        # Only reachable with two arms at the same learning rate; the pilot arms differ.
        tied = [arm("a", 110, lr=1e-5), arm("b", 108, lr=1e-5), arm("c", 90, lr=3e-6)]
        result = r2.apply_r2(tied)
        self.assertEqual(result["outcome"], "stop_for_user")
        self.assertIn("does not compute it", result["reason"])
        result = r2.apply_r2([arm("a", 110, lr=1e-5, cpt=0.95), arm("b", 108, lr=1e-5, cpt=0.94)])
        self.assertEqual(result["winner"], "b")
        result = r2.apply_r2([arm("a", 110, lr=1e-5, cpt=0.95), arm("b", 108, lr=1e-5, cpt=0.95)])
        self.assertEqual(result["outcome"], "stop_for_user")

    def test_select_job_file_is_cpu_only_and_runs_right_after_lr3e_5(self):
        job = validate_job(json.loads(SELECT_JOB.read_text()))
        pilot = json.loads((REPO / "docs/training/plan-20260927/c04/jobs/C08-arm-lr3e-5.json").read_text())
        self.assertEqual(job["requires"], ["C08-arm-lr3e-5"])
        self.assertEqual(job["priority"], pilot["priority"] + 1)
        self.assertFalse(job["resumable"])
        self.assertLessEqual(job["estimated_minutes"], 10)
        self.assertIn("CUDA_VISIBLE_DEVICES=", job["command"])
        self.assertEqual(job["command"][-1], "/home/m0hawk/sandbox/Sepalith/docs/training/plan-20260927/c08/select_r2.py")
        self.assertNotIn("--dry-run", job["command"])


class BindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tree = Tree(Path(temp.name))

    def assertStops(self, fragment, pending=()):
        code, receipt = self.tree.select()
        self.assertEqual(code, r2.EXIT_STOP_FOR_USER)
        self.assertIn(fragment, receipt["question_for_user"])
        self.assertEqual(self.tree.queue.summary()["pending"], list(pending))
        return receipt

    def test_real_numbers_with_a_lower_third_select_lr1e_5_and_enqueue_c09(self):
        self.tree.standard(metrics(100))
        code, receipt = self.tree.select(environ={"SEPALITH_NIGHT_JOB_ID": "C08-r2-select"})
        self.assertEqual(code, 0)
        self.assertEqual(receipt["winner"]["arm"], "lr1e-5")
        self.assertEqual(receipt["winner"]["resume_from"]["path"], str(self.tree.archive("lr1e-5") / "full/checkpoint-268"))
        job = self.tree.queue.load("pending", "C09-arm-a-lr1e-5")
        c08 = self.tree.job("lr1e-5")
        self.assertEqual(job["command"], c08["command"][:-2])
        self.assertNotIn("--stop-after-gate", job["command"])
        self.assertEqual((job["resumable"], job["estimated_minutes"], job["cwd"], job["env"]),
                         (True, 240, c08["cwd"], c08["env"]))
        self.assertEqual(job["requires"], ["C07-baseline", "C08-arm-lr1e-5", "C08-r2-select"])
        self.assertFalse(receipt["continuation"]["already_queued"])

    def test_projected_outcomes_for_the_third_arm(self):
        cases = {116: "lr3e-5", 115: "lr1e-5", 113: "lr1e-5", 110: "lr1e-5", 50: "lr1e-5"}
        for score, winner in cases.items():
            with self.subTest(score250=score), tempfile.TemporaryDirectory() as other:
                tree = Tree(Path(other))
                tree.standard(metrics(score))
                code, receipt = tree.select(dry_run=True)
                self.assertEqual((code, receipt["winner"]["arm"]), (0, winner))
                self.assertEqual(tree.queue.summary()["pending"], [])
        with tempfile.TemporaryDirectory() as other:
            tree = Tree(Path(other))
            tree.standard(metrics(140, valid=50))
            code, receipt = tree.select(dry_run=True)
            self.assertEqual((code, receipt["winner"]["arm"]), (0, "lr1e-5"))
            self.assertEqual(receipt["inputs"]["lr3e-5"]["floor_failures"], ["valid 50 < 60"])

    def test_main_writes_the_receipt_and_job_result_and_is_idempotent(self):
        self.tree.standard(metrics(100))
        result = self.tree.root / "result.json"
        code, err = self.tree.main(environ={"SEPALITH_JOB_RESULT": str(result)})
        self.assertEqual(code, 0, err)
        receipt = json.loads(self.tree.receipt.read_text())
        self.assertEqual((receipt["outcome"], receipt["winner"]["arm"]), ("continue", "lr1e-5"))
        self.assertEqual(json.loads(result.read_text())["rule_decisions"][0]["winner"], "lr1e-5")
        code, _ = self.tree.main()  # a retry finds the identical job and does not duplicate it
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(self.tree.receipt.read_text())["continuation"]["already_queued"])
        self.assertEqual(len(list(self.tree.receipt.parent.glob("C08-r2-selection.superseded-*.json"))), 1)
        self.assertEqual(self.tree.queue.summary()["pending"], ["C09-arm-a-lr1e-5"])

    def test_main_exit_3_and_needs_user_on_stop(self):
        self.tree.standard(metrics(100))
        shutil.rmtree(self.tree.evidence("lr1e-5"))
        code, err = self.tree.main(environ={"SEPALITH_JOB_RESULT": str(self.tree.root / "result.json")})
        self.assertEqual(code, 3)
        self.assertIn("stop-for-user", err)
        self.assertTrue(json.loads((self.tree.root / "result.json").read_text())["needs_user"])

    def test_no_survivor_stops(self):
        self.tree.add_arm("lr1e-5", metrics(110, cap=3))
        self.tree.add_arm("lr3e-6", metrics(104, valid=40))
        self.tree.add_arm("lr3e-5", metrics(100, edit=0))
        receipt = self.assertStops("no pilot arm passed")
        self.assertIn("lr1e-5: score250 110, floor fail", receipt["question_for_user"])

    def test_missing_or_unfinished_third_arm_stops(self):
        self.tree.standard(metrics(100), third_state="deferred", staged=True)
        self.assertStops("must be in done/")

    def test_missing_decision_stops(self):
        self.tree.standard(metrics(100), staged=True)
        self.assertStops("no gate-268 decision")

    def test_refuses_decisions_not_bound_to_their_arm(self):
        mutations = {
            "bound_recipe_sha256": lambda d: d.update(bound_recipe_sha256="0" * 64),
            "step": lambda d: d.update(step=536),
            "schema": lambda d: d.update(schema="other"),
            "human_override": lambda d: d.update(decided_by="human_override"),
            "override": lambda d: d.update(override={"outcome": "continue"}),
            "evaluation_complete": lambda d: d.update(evaluation_complete=False),
            "rule_outcome": lambda d: d.update(rule_outcome="stop_for_user"),
            "disagrees with the R1 floor": lambda d: d["metrics"].update(valid=50) or d["rule"]["inputs"]["current"].update(valid=50),
            "metrics differ": lambda d: d["metrics"].update(score250=200),
            "R1 recorded problems": lambda d: d["rule"].update(problems=["evaluator crash"]),
            "first-gate R1 outcome": lambda d: d.update(outcome="rollback", rule_outcome="rollback"),
            "checkpoint manifest": lambda d: d.update(checkpoint_manifest_sha256="0" * 64),
            "dev250 result": lambda d: d["results"]["dev250"].update(sha256="0" * 64),
        }
        for fragment, change in mutations.items():
            with self.subTest(fragment), tempfile.TemporaryDirectory() as other:
                self.tree = Tree(Path(other))
                self.tree.standard(metrics(100))
                self.tree.edit_decision("lr1e-5", change)
                self.assertStops(fragment)

    def test_latest_decision_attempt_is_the_one_used(self):
        self.tree.standard(metrics(100))
        first = json.loads(self.tree.decision_path("lr1e-5").read_text())
        write(self.tree.decision_path("lr1e-5", 2), {**first, "bound_recipe_sha256": "0" * 64})
        self.assertStops("decision-step-268-2.json")

    def test_refuses_other_odd_states(self):
        cases = {
            "latest durable checkpoint is 536": lambda t: write(
                t.archive("lr3e-6") / "full/checkpoint-536/campaign-manifest.json", "x"),
            "gate evidence": lambda t: (t.evidence("lr1e-5") / "dev-gate-step-268.json").unlink(),
            "differ in shared_decision": lambda t: t.edit_decision("lr3e-5", lambda d: d.update(panel_sha256="1" * 64)),
            "differs from": lambda t: t.queue.path("done", "C08-arm-lr3e-6").write_text(json.dumps(
                {**t.queue.load("done", "C08-arm-lr3e-6"), "command": ["true"]})),
            "bound_recipe_sha256": lambda t: write(t.recipe_path("lr3e-6"), {**json.loads(
                t.recipe_path("lr3e-6").read_text()), "note": "edited after the run"}),
            "learning rate": lambda t: write(t.recipe_path("lr3e-6"), {**json.loads(
                t.recipe_path("lr3e-6").read_text()), "arm": {"name": "lr3e-6", "learning_rate": 1e-05}}),
        }
        for fragment, change in cases.items():
            with self.subTest(fragment), tempfile.TemporaryDirectory() as other:
                self.tree = Tree(Path(other))
                self.tree.standard(metrics(100))
                change(self.tree)
                self.assertStops(fragment)

    def test_differing_parent_metrics_stop(self):
        self.tree.add_arm("lr1e-5", REAL["lr1e-5"])
        self.tree.add_arm("lr3e-6", REAL["lr3e-6"])
        self.tree.add_arm("lr3e-5", metrics(100), parent={**PARENT, "edit": 1})
        self.assertStops("parent_metrics")

    def test_conflicting_continuation_job_stops(self):
        self.tree.standard(metrics(100))
        self.tree.queue.enqueue({"id": "C09-arm-a-lr3e-6", "command": ["true"], "cwd": "/tmp"}, environ={})
        receipt = self.assertStops("C09-arm-a-lr3e-6 is already in pending/", pending=["C09-arm-a-lr3e-6"])
        self.assertEqual(receipt["winner"]["arm"], "lr1e-5")
        with tempfile.TemporaryDirectory() as other:
            self.tree = Tree(Path(other))
            self.tree.standard(metrics(100))
            self.tree.queue.enqueue({"id": "C09-arm-a-lr1e-5", "command": ["true"], "cwd": "/tmp"}, environ={})
            self.assertStops("different definition", pending=["C09-arm-a-lr1e-5"])


def live_at_gate_268() -> bool:
    """True while the real arms' latest durable checkpoint is still gate 268 (C08's night-2 state)."""
    steps = [int(path.parent.name.removeprefix("checkpoint-"))
             for path in LIVE_CHECKPOINTS.glob("SFT-v2-r4-*/full/checkpoint-*/campaign-manifest.json")
             if path.parent.name.removeprefix("checkpoint-").isdigit()]
    return bool(steps) and max(steps) == 268


@unittest.skipUnless((LIVE_GATES / "SFT-v2-r4-lr1e-5/decisions/decision-step-268-1.json").is_file()
                     and (LIVE_GATES / "SFT-v2-r4-lr3e-6/decisions/decision-step-268-1.json").is_file()
                     and (LIVE_CHECKPOINTS / "SFT-v2-r4-lr3e-5/full/checkpoint-268/campaign-manifest.json").is_file()
                     and live_at_gate_268(),
                     "needs the real C08 gate-268 decisions and checkpoints on this PC, before any arm trained past 268")
class LiveDecisionTests(unittest.TestCase):
    """The two real decisions (read only, copied) and a synthetic lr3e-5 decision bound to the real recipe."""

    def build(self, root: Path, third: dict) -> tuple[Path, Path]:
        gates, queue_root = root / "gates", root / "night-queue"
        queue = Queue(queue_root)
        for lr in ("lr1e-5", "lr3e-6"):
            source = LIVE_GATES / f"SFT-v2-r4-{lr}"
            (gates / source.name / "decisions").mkdir(parents=True)
            shutil.copy2(source / "decisions/decision-step-268-1.json", gates / source.name / "decisions")
            shutil.copy2(source / "dev-gate-step-268.json", gates / source.name)
        recipe = REPO / "docs/training/plan-20260927/c04/bound/sft-v2-lr3e-5.json"
        template = json.loads((gates / "SFT-v2-r4-lr1e-5/decisions/decision-step-268-1.json").read_text())
        decision = copy.deepcopy(template)
        results = {panel: write(root / f"results/dev-{panel}.json", {"synthetic": panel}) for panel in ("dev75", "dev250")}
        floor = r2.first_gate_floor(third, 0)
        outcome = "continue" if not floor else "stop_for_user"
        decision.update(metrics=third, outcome=outcome, rule_outcome=outcome, bound_recipe_sha256=sha(recipe),
                        checkpoint_manifest_sha256=sha(LIVE_CHECKPOINTS / "SFT-v2-r4-lr3e-5/full/checkpoint-268/campaign-manifest.json"),
                        results={p: {"path": str(path), "sha256": sha(path)} for p, path in results.items()})
        decision["rule"].update(outcome=outcome, inputs={"current": third, "parent": template["rule"]["inputs"]["parent"]})
        write(gates / "SFT-v2-r4-lr3e-5/decisions/decision-step-268-1.json", decision)
        if outcome == "continue":
            write(gates / "SFT-v2-r4-lr3e-5/dev-gate-step-268.json", {"synthetic": True})
        for lr in ("lr1e-5", "lr3e-6", "lr3e-5"):
            job = json.loads((REPO / f"docs/training/plan-20260927/c04/jobs/C08-arm-{lr}.json").read_text())
            queue.enqueue(job, environ={})
            queue.move(job["id"], "pending", "done")
        return gates, queue_root

    def test_live_two_arms_with_synthetic_third(self):
        cases = {100: "lr1e-5", 113: "lr1e-5", 115: "lr1e-5", 116: "lr3e-5"}
        for score, winner in cases.items():
            with self.subTest(score250=score), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                gates, queue_root = self.build(root, metrics(score))
                code, receipt = r2.select(REPO, queue_root, root / "receipt.json", gates, dry_run=False, environ={})
                self.assertEqual(code, 0, receipt["question_for_user"])
                self.assertEqual(receipt["winner"]["arm"], winner)
                self.assertEqual(receipt["inputs"]["lr1e-5"]["metrics"], REAL["lr1e-5"])
                self.assertEqual(receipt["inputs"]["lr3e-6"]["metrics"], REAL["lr3e-6"])
                job = Queue(queue_root).load("pending", f"C09-arm-a-{winner}")
                self.assertEqual(job["command"][-2:], ["--recipe", f"/home/m0hawk/sandbox/Sepalith/docs/training/"
                                                       f"plan-20260927/c04/bound/sft-v2-{winner}.json"])


class FakeClock(Clock):
    def __init__(self, now):
        self.value = float(now)

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(0.0, seconds)
        time.sleep(0.005)


class IdleProbes(Probes):
    def gpu(self):
        return {"used_mib": 500, "compute_pids": []}

    def host_memory(self):
        return {"windows_available_mib": 30000, "committed_bytes": 10, "commit_limit_bytes": 100,
                "linux_available_mib": 20000}

    def disk_free(self, path):
        return 10**12

    def cuda_lock_free(self, path):
        return True


class RunnerTests(unittest.TestCase):
    """The night runner with the real selector: requirements, mid-night enqueue, failure modes."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tree = Tree(Path(temp.name))
        # lr3e-5 sits in deferred/ as it does after the night of 2026-10-12; its decision is written when it runs.
        self.tree.standard(metrics(100), third_state="deferred", staged=True)
        job = json.loads(SELECT_JOB.read_text())
        job.update(cwd=str(self.tree.root), command=[sys.executable, "-I", "-B", str(SCRIPT), "--repo-root",
                                                     str(self.tree.repo), "--queue-root", str(self.tree.queue_root),
                                                     "--receipt", str(self.tree.receipt)])
        self.tree.queue.enqueue(job, environ={})
        self.tree.queue.enqueue({"id": "C07-baseline", "command": ["true"], "cwd": str(self.tree.root)}, environ={})
        self.tree.queue.move("C07-baseline", "pending", "done")

    def night(self, label="2026-10-13"):
        window = Window(label, label, 1000.0, 100000.0, 100100.0, 100200.0)
        environ = {"PATH": os.environ["PATH"], "HOME": str(self.tree.root)}
        with redirect_stdout(io.StringIO()):
            runner = NightRunner(self.tree.root / "state", window, clock=FakeClock(window.start), probes=IdleProbes(),
                                 environ=environ, poll_seconds=1.0, min_start_seconds=60.0)
            return runner.run()

    def test_selector_runs_after_lr3e_5_and_its_continuation_runs_the_same_night(self):
        record = self.night()
        self.assertEqual([job["id"] for job in record["jobs"]], ["C08-arm-lr3e-5", "C08-r2-select", "C09-arm-a-lr1e-5"])
        self.assertEqual([job["outcome"] for job in record["jobs"]], ["done"] * 3)
        self.assertEqual(record["stop_reason"], "no_eligible_job")
        self.assertEqual(record["needs_user"], [])
        argv = json.loads((self.tree.root / "continued-lr1e-5.json").read_text())
        self.assertNotIn("--stop-after-gate", argv)
        self.assertEqual(record["jobs"][1]["result"]["rule_decisions"][0]["winner"], "lr1e-5")
        self.assertEqual(Queue(self.tree.queue_root).load("done", "C09-arm-a-lr1e-5")["requires"],
                         ["C07-baseline", "C08-arm-lr1e-5", "C08-r2-select"])

    def test_lr3e_5_deferred_again_holds_the_selector_until_the_next_night(self):
        write(self.tree.root / "arm-mode.json", {"lr3e-5": "defer"})
        record = self.night()
        self.assertEqual([job["id"] for job in record["jobs"]], ["C08-arm-lr3e-5"])
        self.assertEqual(record["skipped"], [{"id": "C08-r2-select", "reason": "waiting_for_requirements"}])
        self.assertEqual(record["stop_reason"], "no_eligible_job")
        self.assertEqual(self.tree.queue.locate("C08-r2-select"), "pending")
        write(self.tree.root / "arm-mode.json", {})
        record = self.night("2026-10-14")
        self.assertEqual([job["id"] for job in record["jobs"]], ["C08-arm-lr3e-5", "C08-r2-select", "C09-arm-a-lr1e-5"])

    def test_lr3e_5_failure_skips_the_selector_and_flags_the_user(self):
        write(self.tree.root / "arm-mode.json", {"lr3e-5": "fail"})
        record = self.night()
        self.assertEqual([job["id"] for job in record["jobs"]], ["C08-arm-lr3e-5"])
        self.assertEqual(record["skipped"], [{"id": "C08-r2-select", "reason": "requires_failed_job"}])
        self.assertEqual(self.tree.queue.locate("C08-r2-select"), "pending")
        self.assertTrue(any("C08-arm-lr3e-5 failed" in item for item in record["needs_user"]))

    def test_stop_for_user_fails_the_selector_and_enqueues_nothing(self):
        for lr in ("lr1e-5", "lr3e-6"):
            self.tree.edit_decision(lr, lambda d: d.update(bound_recipe_sha256="0" * 64))
        record = self.night()
        self.assertEqual([job["id"] for job in record["jobs"]], ["C08-arm-lr3e-5", "C08-r2-select"])
        self.assertEqual((record["jobs"][1]["outcome"], record["jobs"][1]["exit_code"]), ("failed", 3))
        self.assertTrue(any(item.startswith("C08-r2-select: R2 was not applied") for item in record["needs_user"]))
        self.assertEqual(self.tree.queue.summary()["pending"], [])
        self.assertEqual(json.loads(self.tree.receipt.read_text())["outcome"], "stop_for_user")


if __name__ == "__main__":
    unittest.main()
