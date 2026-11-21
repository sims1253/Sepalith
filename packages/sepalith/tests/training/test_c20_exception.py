"""Synthetic evidence for the C20 checkpoint-300 exception."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sepalith.training.c20_exception import final
from sepalith.training.sft.c20 import run


class ExceptionPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.recipe_path = self.root / "pilot.json"
        self.recipe = {"mode": "pilot", "outputs": {"evidence": str(self.root)}}
        self.write(self.recipe_path, self.recipe)
        self.binding = {"step": 300, "recipe_sha256": run.sha256(self.recipe_path),
                        "checkpoint_manifest_sha256": "abc"}
        self.summary_path = self.root / "pilot-summary.json"
        self.dev250_path = self.root / "dev250.json"
        self.dev250 = {"binding": self.binding,
                       "metrics": {"score250": 107, "dev250_edit_exact": 72, "dev250_noop_correct": 35}}
        self.write(self.dev250_path, self.dev250)
        self.start_rows = [
            {"task_id": str(i), "function_id": "f", "family": "finish_block", "target_lines": 3, "all_pass": False}
            for i in range(10)
        ] + [{"task_id": "rename", "function_id": "g", "family": "rename_propagation",
              "target_lines": 1, "all_pass": True}]
        self.selected_rows = [{**row, "all_pass": True} for row in self.start_rows]
        self.baseline = self.gate(0, self.start_rows, 0, 1.0)
        self.winner = self.gate(300, self.selected_rows, 0, 1.01)
        self.gates = [{"binding": {"step": step}, "criteria": {"accepted": False}} for step in (100, 200)]
        self.gates.append(self.winner)
        self.summary = {
            "status": "keep_c19", "result_checkpoint": str(run.config.PARENT),
            "recipe": run.record(self.recipe_path), "dev250_result": run.record(self.dev250_path),
            "dev250": self.dev250["metrics"], "gates": self.gates,
        }
        self.receipt = {
            "schema": final.RECEIPT_SCHEMA, "waived_criterion": final.WAIVED_CRITERION,
            "adopt_criteria": final.ADOPT_CRITERIA, "observed_dev250": self.dev250["metrics"],
            "binding": self.binding, "dev250_sha256": run.sha256(self.dev250_path),
        }
        self.write_summary()

    def write(self, path: Path, content: dict) -> None:
        path.write_text(json.dumps(content))

    def gate(self, step: int, rows: list[dict], nofp: int, nll: float) -> dict:
        directory = run.evidence(self.recipe, step)
        directory.mkdir(exist_ok=True)
        validation = directory / "records.jsonl"
        validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = {
            "binding": {**self.binding, "step": step}, "inputs": {"validation": run.record(validation)},
            "dev75": {"nofp": nofp}, "cpt": {"nll": nll},
        }
        if step:
            report["criteria"] = run.promote(self.start_rows, rows, nofp, nll - 1.0)
        self.write(directory / "gate.json", report)
        return report

    def write_summary(self) -> None:
        self.write(self.summary_path, self.summary)
        self.receipt["pilot_summary_sha256"] = run.sha256(self.summary_path)

    def verify(self) -> dict:
        with (patch.object(run.sft, "load_bound", return_value=self.recipe),
              patch.object(run, "binding", side_effect=lambda _, __, step: {**self.binding, "step": step}),
              patch.object(run, "checkpoint", return_value=self.root / "checkpoint-300"),
              patch.object(run, "verify_saved_checkpoint") as verify_checkpoint):
            _, winner = final.verify_pilot(self.receipt, self.summary_path, self.dev250_path)
            verify_checkpoint.assert_called_once_with(self.recipe, 300)
        return winner

    def test_receipt_hashes_must_match(self) -> None:
        for field in ("pilot_summary_sha256", "dev250_sha256"):
            with self.subTest(field=field), patch.dict(self.receipt, {field: "wrong"}):
                with self.assertRaisesRegex(ValueError, "input differs"):
                    self.verify()

    def test_dev250_score_below_108_can_reach_final(self) -> None:
        self.assertEqual(self.verify(), self.winner)
        with patch.dict(self.receipt, waived_criterion="any failed criterion"):
            with self.assertRaisesRegex(ValueError, "exception scope differs"):
                self.verify()

    def test_failed_promotion_flag_rejects(self) -> None:
        self.winner["criteria"]["accepted"] = False
        self.write_summary()
        with self.assertRaisesRegex(ValueError, "no passing pilot gate"):
            self.verify()

    def test_acceptance_flag_cannot_waive_failed_criteria(self) -> None:
        bad_rename = [*self.selected_rows[:-1], {**self.selected_rows[-1], "all_pass": False}]
        for name, rows, nofp, nll in (
            ("finish gain", self.start_rows, 0, 1.01),
            ("rename drop", bad_rename, 0, 1.01),
            ("DEV75 nofp", self.selected_rows, 3, 1.01),
            ("CPT NLL", self.selected_rows, 0, 1.04),
        ):
            with self.subTest(criterion=name):
                report = self.gate(300, rows, nofp, nll)
                self.assertFalse(report["criteria"]["accepted"])
                report["criteria"]["accepted"] = True
                self.write(run.evidence(self.recipe, 300) / "gate.json", report)
                self.gates[-1] = report
                self.write_summary()
                with self.assertRaisesRegex(ValueError, "pilot promotion criteria differ"):
                    self.verify()

    def test_step_binding_must_match(self) -> None:
        with patch.dict(self.receipt, binding={**self.binding, "step": 200}):
            with self.assertRaisesRegex(ValueError, "exception step-300 binding differs"):
                self.verify()

    def test_holdout_defers_and_resumes_with_pinned_baseline(self) -> None:
        out = self.root / "holdout"
        cp = self.root / "checkpoint-300"
        baseline = self.root / "baseline.jsonl"
        baseline.write_text("synthetic completions\n")
        spec = {"completions": run.record(baseline), "tasks": {"path": "synthetic"}, "n_tasks": 1}
        with patch.object(run, "exec_run", return_value=run.EXIT_DEFERRED):
            self.assertEqual(final.evaluate(out, spec, cp, self.winner, self.receipt), run.EXIT_DEFERRED)
        seeded = out / f"completions-{run.config.START}.jsonl"
        self.assertEqual(seeded.read_bytes(), baseline.read_bytes())
        self.assertFalse((out / "verdict.json").exists())

        receipt_path = out / "receipt.json"
        receipt = run.data.read(receipt_path)
        self.assertEqual(receipt["models"][run.config.START]["status"], "generated")
        coverage = {"tasks_scored": 1, "tasks_total": 1}
        model = {"coverage": coverage, "by_bucket": {"medium": {}, "long": {}}}
        for lower, cpt, adopt in ((0.001, 0.03, True), (0, 0.03, False), (0.001, 0.031, False)):
            with self.subTest(lower=lower, cpt=cpt), patch.object(run, "emit", return_value=0):
                metrics = {
                    run.config.START: model,
                    run.config.SELECTED: {
                        **model, f"vs_{run.config.START}": {"all_pass": {"n": 1, "ci95": [lower, 0.05]}},
                    },
                }
                self.write(out / "metrics.json", metrics)
                self.write(receipt_path, {**receipt, "status": "complete"})
                self.winner["criteria"]["cpt_nll_rise"] = cpt
                self.assertEqual(final.evaluate(out, spec, cp, self.winner, self.receipt), 0)
                verdict = run.data.read(out / "verdict.json")
                self.assertEqual(verdict["adopt"], adopt)
                self.assertEqual(verdict["exception"], self.receipt)
                self.assertEqual(verdict["result_checkpoint"], str(cp if adopt else run.config.PARENT))
