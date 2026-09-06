"""Small real-process tests; no models, network, NAS or training dependencies."""
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from sepalith.runner import Runner, RunnerError, _write_json, main, validate_recipe


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
        (self.repo / "job.py").write_text("import pathlib, sys\npathlib.Path(sys.argv[1]).write_text('captured')\n")
        self.runner = Runner(self.root / "state")
        self.snapshot = self.runner.snapshot(self.repo, ["job.py"])

    def recipe(self, name="example", **changes):
        recipe = {"schema_version": 1, "id": name, "snapshot": self.snapshot,
                  "steps": [{"id": "write", "argv": ["{python}", "{source}/job.py", "{run}/output.txt"],
                             "artifacts": ["output.txt"]}]}
        recipe.update(changes)
        return recipe

    def test_snapshot_includes_untracked_bytes_and_is_idempotent(self):
        self.assertEqual(self.snapshot, self.runner.snapshot(self.repo, ["job.py"]))
        (self.repo / "job.py").write_text("raise RuntimeError('new working tree')")
        self.runner.enqueue(self.recipe())
        self.assertIsNone(self.runner.run_next())  # new queues start paused
        self.runner.resume()
        attempt = self.runner.run_next()
        self.assertEqual("captured", (self.runner.root / "attempts" / attempt / "output.txt").read_text())
        self.assertEqual("succeeded", self.runner.plan()["jobs"][0]["status"])

    def test_snapshot_tampering_fails_before_launch(self):
        self.runner.enqueue(self.recipe())
        path = self.runner.root / "snapshots" / self.snapshot / "source" / "job.py"
        path.chmod(0o644)
        path.write_text("raise SystemExit(99)")
        self.runner.resume()
        attempt = self.runner.run_next()
        state = self.runner.plan()["attempts"][0]
        self.assertEqual("failed", state["status"])
        self.assertIn("Snapshot content changed", state["detail"])
        self.assertFalse((self.runner.root / "attempts" / attempt / "00-write.log").exists())

    def test_dependencies_and_restart_do_not_repeat_success(self):
        self.runner.enqueue(self.recipe("second", depends_on=["first"]))
        self.runner.enqueue(self.recipe("first"))
        self.runner.resume()
        self.runner.run_next()
        restarted = Runner(self.runner.root)
        self.assertEqual(["queued", "succeeded"], [j["status"] for j in restarted.plan()["jobs"]])
        restarted.run_next()
        self.assertIsNone(restarted.run_next())
        self.assertEqual(2, len(restarted.plan()["attempts"]))

    def test_failure_stops_steps_and_blocks_dependents(self):
        fail = {"id": "fail", "argv": ["{python}", "-c", "raise SystemExit(7)"]}
        recipe = self.recipe("first")
        recipe["steps"].insert(0, fail)
        self.runner.enqueue(recipe)
        self.runner.enqueue(self.recipe("second", depends_on=["first"]))
        self.runner.resume()
        attempt = self.runner.run_next()
        self.assertEqual("failed", self.runner.plan()["jobs"][0]["status"])
        self.assertFalse((self.runner.root / "attempts" / attempt / "output.txt").exists())
        self.assertIsNone(self.runner.run_next())
        self.runner.retry("first")
        self.runner.run_next()
        self.assertEqual(2, len(self.runner.plan()["attempts"]))

    def test_each_source_include_must_match(self):
        with self.assertRaisesRegex(RunnerError, "No Git-visible files.*missing.py"):
            self.runner.snapshot(self.repo, ["job.py", "missing.py"])

    def test_later_step_cannot_invalidate_recorded_artifact(self):
        for operation in ("p.write_text('overwritten')", "p.unlink()"):
            with self.subTest(operation=operation):
                name = "mutate" if "write_text" in operation else "delete"
                code = "from pathlib import Path; p=Path('{run}/output.txt'); " + operation
                steps = [*self.recipe()["steps"],
                         {"id": "archive", "argv": ["{python}", "-c", code]}]
                self.runner.enqueue(self.recipe(name, steps=steps))
                self.runner.resume()
                self.runner.run_next()
                state = self.runner.plan()["attempts"][-1]
                self.assertEqual("failed", state["status"])
                self.assertIn("Previously verified artifact changed", state["detail"])

    def test_missing_artifact_is_not_success(self):
        recipe = self.recipe()
        recipe["steps"][0]["artifacts"] = ["missing.txt"]
        self.runner.enqueue(recipe)
        self.runner.resume()
        self.runner.run_next()
        self.assertEqual("failed", self.runner.plan()["jobs"][0]["status"])

    def test_changed_input_fails_without_launch(self):
        source = self.root / "data.txt"
        source.write_text("old")
        recipe = self.recipe(inputs=[{"path": str(source), "sha256": hashlib.sha256(b"old").hexdigest()}])
        self.runner.enqueue(recipe)
        source.write_text("new")
        self.runner.resume()
        self.runner.run_next()
        self.assertIn("Input hash mismatch", self.runner.plan()["attempts"][0]["detail"])

    def test_cycle_and_path_escape_rejected(self):
        self.runner.enqueue(self.recipe("first", depends_on=["second"]))
        with self.assertRaisesRegex(RunnerError, "Dependency cycle"):
            self.runner.enqueue(self.recipe("second", depends_on=["first"]))
        recipe = self.recipe()
        recipe["steps"][0]["artifacts"] = ["../escape"]
        with self.assertRaises(RunnerError):
            validate_recipe(recipe)
        with self.assertRaises(RunnerError):
            self.runner.snapshot(self.repo, ["../outside"])

    def test_missing_executable_is_failed_not_unknown_launch(self):
        self.runner.enqueue(self.recipe(steps=[{"id": "missing", "argv": ["/no-such-sepalith-executable"]}]))
        self.runner.resume()
        self.runner.run_next()
        self.assertEqual("failed", self.runner.plan()["attempts"][0]["status"])

    def test_log_open_failure_is_failed_and_does_not_block_next_job(self):
        self.runner.enqueue(self.recipe("first"))
        self.runner.enqueue(self.recipe("second"))
        self.runner.resume()
        original_open = Path.open

        def fail_log(path, *args, **kwargs):
            if path.name == "00-write.log":
                raise PermissionError("log is unavailable")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", fail_log), patch("sepalith.runner.subprocess.Popen") as launch:
            self.runner.run_next()
            launch.assert_not_called()
        self.assertEqual("failed", self.runner.plan()["attempts"][0]["status"])
        self.runner.run_next()
        self.assertEqual(["failed", "succeeded"], [j["status"] for j in self.runner.plan()["jobs"]])

    def test_json_publication_does_not_follow_stale_temp_symlink(self):
        outside = self.root / "outside.txt"
        outside.write_text("preserve")
        record = self.root / "record.json"
        record.with_name("record.json.tmp").symlink_to(outside)
        _write_json(record, {"complete": True})
        self.assertEqual("preserve", outside.read_text())
        self.assertEqual({"complete": True}, json.loads(record.read_text()))

    def test_failed_json_publication_preserves_previous_record(self):
        record = self.root / "record.json"
        _write_json(record, {"previous": True})
        with self.assertRaises(ValueError):
            _write_json(record, {"invalid": float("nan")})
        self.assertEqual({"previous": True}, json.loads(record.read_text()))
        self.assertEqual([], list(self.root.glob("record.json.*.tmp")))

    def test_artifact_symlink_cannot_escape_attempt(self):
        outside = self.root / "outside.txt"
        outside.write_text("preserve")
        code = "from pathlib import Path; Path(%r).symlink_to(%r)" % ("{run}/output.txt", str(outside))
        self.runner.enqueue(self.recipe(steps=[{"id": "link", "argv": ["{python}", "-c", code],
                                               "artifacts": ["output.txt"]}]))
        self.runner.resume()
        self.runner.run_next()
        attempt = self.runner.plan()["attempts"][0]
        self.assertEqual("failed", attempt["status"])
        self.assertIn("escapes its root", attempt["detail"])
        self.assertEqual("preserve", outside.read_text())

    def start_dispatcher(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.Popen([sys.executable, "-m", "sepalith.runner", "--state", str(self.runner.root), "run-next"],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.poll() is None and (proc.kill(), proc.wait()))
        return proc

    def wait_for_file(self, path):
        deadline = time.monotonic() + 5
        while not path.exists():
            if time.monotonic() > deadline:
                self.fail(f"Toy process did not write {path}")
            time.sleep(0.01)

    def test_pause_drains_current_experiment_including_later_steps(self):
        marker, release = self.root / "started", self.root / "release"
        code = "import pathlib,time; pathlib.Path(%r).touch();\nwhile not pathlib.Path(%r).exists(): time.sleep(.01)" % (str(marker), str(release))
        steps = [{"id": "wait", "argv": ["{python}", "-c", code]}, *self.recipe()["steps"]]
        self.runner.enqueue(self.recipe("first", steps=steps))
        self.runner.enqueue(self.recipe("second"))
        self.runner.resume()
        proc = self.start_dispatcher()
        self.wait_for_file(marker)
        self.runner.pause()
        with self.assertRaisesRegex(RunnerError, "Another dispatcher"):
            self.runner.run_next()
        release.touch()
        proc.communicate(timeout=5)
        self.assertEqual(0, proc.returncode)
        self.assertEqual(["succeeded", "queued"], [j["status"] for j in self.runner.plan()["jobs"]])
        self.assertIsNone(self.runner.run_next())

    def test_crashed_dispatcher_does_not_duplicate_surviving_child(self):
        marker, release = self.root / "started", self.root / "release"
        code = "import pathlib,time; pathlib.Path(%r).touch();\nwhile not pathlib.Path(%r).exists(): time.sleep(.01)" % (str(marker), str(release))
        self.runner.enqueue(self.recipe(steps=[{"id": "wait", "argv": ["{python}", "-c", code]}]))
        self.runner.resume()
        proc = self.start_dispatcher()
        self.wait_for_file(marker)
        attempt = self.runner.plan()["attempts"][0]
        child = attempt["child_group"]
        self.addCleanup(lambda: release.touch())
        proc.kill()
        proc.communicate(timeout=5)
        with self.assertRaisesRegex(RunnerError, "unfinished attempt"):
            self.runner.run_next()
        with self.assertRaisesRegex(RunnerError, "still alive"):
            self.runner.recover(attempt["id"])
        release.touch()
        deadline = time.monotonic() + 5
        while True:
            try:
                self.runner.recover(attempt["id"])
                break
            except RunnerError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        self.assertEqual("interrupted", self.runner.plan()["jobs"][0]["status"])
        self.runner.retry("example")
        self.runner.run_next()
        self.assertEqual(["interrupted", "succeeded"], [a["status"] for a in self.runner.plan()["attempts"]])
        self.assertIsNotNone(child)

    def test_read_only_snapshot_mode_changes_are_detected(self):
        path = self.runner.root / "snapshots" / self.snapshot / "source" / "job.py"
        path.chmod(0o555)
        with self.assertRaisesRegex(RunnerError, "content changed"):
            self.runner.verify_snapshot(self.snapshot)

    def test_nonregular_input_is_rejected_without_blocking(self):
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        self.runner.enqueue(self.recipe(inputs=[{"path": str(fifo), "sha256": "0" * 64}]))
        self.runner.resume()
        self.runner.run_next()
        self.assertIn("not a regular file", self.runner.plan()["attempts"][0]["detail"])

    def test_source_changes_between_steps_fail(self):
        mutate = "from pathlib import Path; p=Path('job.py'); p.chmod(0o644); p.write_text('changed')"
        self.runner.enqueue(self.recipe(steps=[{"id": "mutate", "argv": ["{python}", "-c", mutate]},
                                              *self.recipe()["steps"]]))
        self.runner.resume()
        attempt = self.runner.run_next()
        self.assertEqual("failed", self.runner.plan()["jobs"][0]["status"])
        self.assertIn("source was modified", self.runner.plan()["attempts"][0]["detail"])
        self.assertFalse((self.runner.root / "attempts" / attempt / "output.txt").exists())

    def test_invalid_environment_and_empty_executable_rejected(self):
        with self.assertRaises(RunnerError):
            validate_recipe(self.recipe(env={"": "bad"}))
        with self.assertRaises(RunnerError):
            validate_recipe(self.recipe(steps=[{"id": "bad", "argv": [""]}]))

    def unknown_launch(self):
        self.runner.enqueue(self.recipe())
        self.runner.resume()
        # No real child is created. Simulate losing control in the interval
        # where the dispatcher cannot establish whether Popen launched one.
        with patch("sepalith.runner.subprocess.Popen", side_effect=KeyboardInterrupt):
            with self.assertRaisesRegex(RunnerError, "needs reconciliation"):
                self.runner.run_next()
        self.runner.pause()
        attempt = self.runner.plan()["attempts"][0]
        self.assertEqual("launching", attempt["phase"])
        return attempt

    def process_audit(self, receipt, **changes):
        audit = {"schema_version": 1, "attempt": receipt["id"], "operator": "test-operator",
                 "decision": "interrupted", "checked_at": datetime.now(timezone.utc).isoformat(),
                 "all_processes_stopped": True,
                 "findings": "The fixture prevented Popen from creating a child.",
                 "evidence": "Test fixture: Popen raised KeyboardInterrupt before launch."}
        audit.update(changes)
        return audit

    def test_operator_resolution_preserves_receipts_and_requires_explicit_retry(self):
        before = self.unknown_launch()
        work = self.runner.root / "attempts" / before["id"]
        original = {p.name: p.read_bytes() for p in work.glob("*.json")}
        audit = self.process_audit(before)
        with patch("sepalith.runner.subprocess.Popen") as launch, patch("sepalith.runner.os.killpg") as signal:
            record = self.runner.resolve_unknown(before["id"], audit)
            launch.assert_not_called()
            signal.assert_not_called()
        self.assertEqual(before, record["attempt_before"])
        self.assertEqual(self.recipe(), record["recipe"])
        self.assertEqual(audit, record["audit"])
        self.assertEqual(original, {p.name: p.read_bytes() for p in work.glob("*.json")})
        restarted = Runner(self.runner.root)
        self.assertEqual([record], restarted.plan()["operator_resolutions"])
        self.assertEqual("interrupted", restarted.plan()["jobs"][0]["status"])
        restarted.resume()
        self.assertIsNone(restarted.run_next())
        restarted.retry("example")
        restarted.run_next()
        self.assertEqual(["interrupted", "succeeded"], [a["status"] for a in restarted.plan()["attempts"]])
        self.assertEqual([record], restarted.plan()["operator_resolutions"])
        with self.assertRaisesRegex(RunnerError, "unknown launch"):
            restarted.pause()
            restarted.resolve_unknown(before["id"], audit)
        for statement in ("UPDATE operator_resolutions SET record='{}'", "DELETE FROM operator_resolutions"):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                with restarted._db() as db:
                    db.execute(statement)

    def test_operator_resolution_rejects_incomplete_mismatched_or_unchecked_audits(self):
        before = self.unknown_launch()
        audits = [{}, self.process_audit(before, attempt="different"),
                  self.process_audit(before, operator=" "), self.process_audit(before, evidence=""),
                  self.process_audit(before, decision="succeeded"),
                  self.process_audit(before, all_processes_stopped=False),
                  self.process_audit(before, all_processes_stopped=1),
                  self.process_audit(before, checked_at="2026-01-01T00:00:00"),
                  self.process_audit(before, checked_at="2000-01-01T00:00:00Z"),
                  self.process_audit(before, checked_at="9999-01-01T00:00:00Z")]
        for audit in audits:
            with self.subTest(audit=audit), self.assertRaises(RunnerError):
                self.runner.resolve_unknown(before["id"], audit)
        self.assertEqual(before, self.runner.plan()["attempts"][0])
        self.assertEqual([], self.runner.plan()["operator_resolutions"])
        with self.assertRaisesRegex(RunnerError, "manual process audit"):
            self.runner.recover(before["id"])

    def test_operator_resolution_requires_pause_lock_and_no_known_live_group(self):
        before = self.unknown_launch()
        audit = self.process_audit(before)
        self.runner.resume()
        with self.assertRaisesRegex(RunnerError, "Pause"):
            self.runner.resolve_unknown(before["id"], audit)
        self.runner.pause()
        with self.runner._dispatch_lock(), self.assertRaisesRegex(RunnerError, "Another dispatcher"):
            self.runner.resolve_unknown(before["id"], audit)
        self.runner._update_attempt(before["id"], child_group=12345)
        with patch("sepalith.runner._group_alive", return_value=True) as alive:
            with self.assertRaisesRegex(RunnerError, "still alive"):
                self.runner.resolve_unknown(before["id"], audit)
            alive.assert_called_once_with(12345)
        self.assertEqual("running", self.runner.plan()["attempts"][0]["status"])
        self.assertEqual([], self.runner.plan()["operator_resolutions"])

    def test_operator_resolution_transaction_rolls_back_audit_and_status_together(self):
        before = self.unknown_launch()
        with self.runner._db() as db:
            db.executescript("""
                CREATE TRIGGER test_reject_resolution BEFORE UPDATE ON attempts
                WHEN NEW.status='interrupted' BEGIN
                    SELECT RAISE(ABORT, 'simulated transaction failure');
                END;
            """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated transaction failure"):
            self.runner.resolve_unknown(before["id"], self.process_audit(before))
        self.assertEqual(before, self.runner.plan()["attempts"][0])
        self.assertEqual([], self.runner.plan()["operator_resolutions"])

    def test_operator_resolution_cli_records_failed_decision(self):
        before = self.unknown_launch()
        path = self.root / "audit.json"
        path.write_text(json.dumps(self.process_audit(before, decision="failed")))
        with patch("builtins.print"):
            result = main(["--state", str(self.runner.root), "resolve-unknown", before["id"], "--audit", str(path)])
        self.assertEqual(0, result)
        self.assertEqual("failed", self.runner.plan()["jobs"][0]["status"])
        self.assertEqual("failed", self.runner.plan()["operator_resolutions"][0]["audit"]["decision"])


if __name__ == "__main__":
    unittest.main()
