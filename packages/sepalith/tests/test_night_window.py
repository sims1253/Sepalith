"""Night queue and runner tests with fake jobs, a fake clock and fake probes."""
from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout

from sepalith.ops import night_queue, night_runner
from sepalith.ops.night_queue import Queue, QueueError, validate_job
from sepalith.ops.night_runner import (BERLIN, Clock, Limits, NightRunner, Probes, Window, night_window, parse_gpu,
                                       post_window_check, preflight, test_window)


class FakeClock(Clock):
    def __init__(self, now):
        self.value = float(now)

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(0.0, seconds)
        time.sleep(0.005)


class FakeProbes(Probes):
    def __init__(self, gpu_used=500, compute=(), windows_free=30000, disk=10**12, lock_free=True, fail=None):
        self.gpu_used, self.compute, self.windows_free = gpu_used, list(compute), windows_free
        self.disk, self.lock_free, self.fail = disk, lock_free, fail

    def gpu(self):
        if self.fail == "gpu":
            raise RuntimeError("nvidia-smi missing")
        return {"used_mib": self.gpu_used, "compute_pids": self.compute}

    def host_memory(self):
        return {"windows_available_mib": self.windows_free, "committed_bytes": 10, "commit_limit_bytes": 100,
                "linux_available_mib": 20000}

    def disk_free(self, path):
        return self.disk

    def cuda_lock_free(self, path):
        return self.lock_free


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


# A job that honours the stop request: it saves and exits 75.
HONOUR = """
import json, os, pathlib, signal, sys, time
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps({k: v for k, v in os.environ.items()}))
while not stop and not pathlib.Path(os.environ["SEPALITH_STOP_REQUEST_FILE"]).exists():
    time.sleep(0.01)
pathlib.Path(os.environ["SEPALITH_JOB_RESULT"]).write_text(json.dumps(
    {"checkpoints": ["/tmp/checkpoint-7"], "summary": "saved at step 7"}))
sys.exit(75)
"""

# A job that ignores SIGTERM and starts a grandchild in its own session.
IGNORE = """
import pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"],
    start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(600)
"""

# A job that exits at once but leaves a detached process behind.
LEAVE = """
import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
"""


class NightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = Queue(self.root / "night-queue")
        self.window = Window("2026-10-01", "2026-10-01", 1000.0, 2000.0, 2100.0, 2200.0)

    def script(self, name, source):
        path = self.root / f"{name}.py"
        path.write_text(textwrap.dedent(source))
        return str(path)

    def add(self, identifier, command, **changes):
        job = {"id": identifier, "command": command, "cwd": str(self.root), "estimated_minutes": 1, **changes}
        return self.queue.enqueue(job, environ={})

    def runner(self, window=None, clock=None, probes=None, **options):
        options.setdefault("poll_seconds", 5.0)
        options.setdefault("min_start_seconds", 60.0)
        options.setdefault("preflight_retry_seconds", 300.0)
        environ = {"PATH": os.environ["PATH"], "HOME": str(self.root), "ALLOWED": "yes", "HF_TOKEN": "hf_secretvalue"}
        window = window or self.window
        with redirect_stdout(io.StringIO()):
            runner = NightRunner(self.root, window, clock=clock or FakeClock(window.start), probes=probes or FakeProbes(),
                                 environ=environ, **options)
        return runner

    def run_quietly(self, runner):
        with redirect_stdout(io.StringIO()):
            return runner.run()

    # Queue -----------------------------------------------------------------
    def test_validation_rejects_bad_jobs(self):
        good = {"id": "a", "command": ["true"], "cwd": "/tmp"}
        self.assertEqual(validate_job(good)["priority"], 50)
        for change in ({"id": "../x"}, {"command": []}, {"cwd": "relative"}, {"env": {"HF_TOKEN": "x"}},
                       {"env": ["BAD-NAME"]}, {"priority": -1}, {"estimated_minutes": 0}, {"requires": ["a"]},
                       {"env_values": {}}):
            with self.subTest(change=change), self.assertRaises(QueueError):
                validate_job({**good, **change})

    def test_enqueue_refuses_secret_values_and_duplicates(self):
        with self.assertRaisesRegex(QueueError, "HF_TOKEN"):
            self.queue.enqueue({"id": "leak", "command": ["echo", "hf_abcdefgh123"], "cwd": "/tmp"},
                               environ={"HF_TOKEN": "hf_abcdefgh123"})
        self.add("a", ["true"])
        with self.assertRaisesRegex(QueueError, "already exists"):
            self.add("a", ["true"])

    def test_cli_enqueue_list_cancel_retry(self):
        root = str(self.root / "night-queue")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(night_queue.main(["--root", root, "enqueue", "--id", "job1", "--cwd", str(self.root),
                                               "--env", "ALLOWED", "--resumable", "--priority", "3", "--",
                                               "python3", "-c", "print(1)"]), 0)
        job = self.queue.load("pending", "job1")
        self.assertEqual((job["command"], job["env"], job["priority"], job["resumable"]),
                         (["python3", "-c", "print(1)"], ["ALLOWED"], 3, True))
        out = io.StringIO()
        with redirect_stdout(out):
            night_queue.main(["--root", root, "list"])
            night_queue.main(["--root", root, "cancel", "job1"])
        self.assertIn("job1", out.getvalue())
        self.assertEqual(self.queue.locate("job1"), "failed")
        with redirect_stdout(io.StringIO()):
            night_queue.main(["--root", root, "retry", "job1"])
        self.assertEqual(self.queue.locate("job1"), "pending")
        self.assertNotIn("outcome", self.queue.load("pending", "job1"))

    # Window and checks -----------------------------------------------------
    def test_night_window_uses_berlin_time_across_dst(self):
        noon = datetime(2026, 10, 24, 12, 0, tzinfo=BERLIN).timestamp()
        window = night_window(noon)  # after 09:00, so the next night
        self.assertEqual(window.night, "2026-10-25")
        self.assertEqual(datetime.fromtimestamp(window.start, BERLIN).hour, 1)
        # Clocks go back at 03:00 that night: the window lasts nine hours.
        self.assertEqual(window.hard - window.start, 9 * 3600)
        self.assertEqual(window.deadline - window.start, 8 * 3600 + 40 * 60)
        inside = datetime(2026, 9, 29, 3, 0, tzinfo=BERLIN).timestamp()
        self.assertEqual(night_window(inside).night, "2026-09-29")
        self.assertEqual(night_window(inside).hard - night_window(inside).start, 8 * 3600)
        compressed = test_window(inside, 10, 12.5, 15)
        self.assertEqual((compressed.deadline - inside, compressed.hard - inside), (600, 900))
        self.assertTrue(compressed.label.startswith("2026-09-29-test-"))

    def test_parse_gpu_and_preflight(self):
        self.assertEqual(parse_gpu("4762\n", ""), {"used_mib": 4762, "compute_pids": []})
        self.assertEqual(parse_gpu("10\n", "123\n")["compute_pids"], ["123"])
        passing = preflight(FakeProbes(), Limits())
        self.assertTrue(all(check["ok"] for check in passing), passing)
        for probes, failing in ((FakeProbes(gpu_used=4762), "gpu_idle"), (FakeProbes(compute=["9"]), "gpu_idle"),
                                (FakeProbes(windows_free=9000), "host_memory"),
                                (FakeProbes(disk=149 * 10**9), "disk_free"),
                                (FakeProbes(lock_free=False), "cuda_lock_free"),
                                (FakeProbes(fail="gpu"), "gpu_idle")):
            with self.subTest(failing=failing):
                checks = {check["name"]: check["ok"] for check in preflight(probes, Limits())}
                self.assertEqual([name for name, ok in checks.items() if not ok], [failing])

    # Runner ----------------------------------------------------------------
    def test_finished_job_is_done_with_minimal_environment(self):
        env_file = self.root / "env.json"
        self.add("quick", [sys.executable, "-c", "import json, os, sys; open(sys.argv[1], 'w').write("
                           "json.dumps(dict(os.environ)))", str(env_file)], env=["ALLOWED"])
        record = self.run_quietly(self.runner())
        self.assertEqual(record["stop_reason"], "no_eligible_job")
        self.assertEqual([(job["id"], job["outcome"], job["exit_code"]) for job in record["jobs"]],
                         [("quick", "done", 0)])
        env = json.loads(env_file.read_text())
        self.assertEqual(env["ALLOWED"], "yes")
        self.assertNotIn("HF_TOKEN", env)
        self.assertEqual(env["SEPALITH_DEADLINE_EPOCH"], "2000")
        self.assertEqual(env["SEPALITH_HARD_STOP_EPOCH"], "2200")
        self.assertEqual(self.queue.locate("quick"), "done")
        report = json.loads((self.root / "night-reports/2026-10-01.json").read_text())
        self.assertEqual(report["runs"][0]["jobs"][0]["id"], "quick")
        self.assertIn("| quick | 1 | done |", (self.root / "night-reports/2026-10-01.md").read_text())
        self.assertNotIn("hf_secretvalue", (self.root / "night-reports/2026-10-01.json").read_text())

    def test_honouring_job_is_deferred_then_resumes_next_night(self):
        self.add("train", [sys.executable, self.script("honour", HONOUR), str(self.root / "env.json")],
                 resumable=True, estimated_minutes=600)
        record = self.run_quietly(self.runner())
        job = record["jobs"][0]
        self.assertEqual((job["outcome"], job["reason"], job["exit_code"]),
                         ("deferred", "stopped_at_safe_boundary", 75))
        self.assertIsNotNone(job["graceful_stop_at"])
        self.assertIsNone(job["hard_kill_at"])
        self.assertEqual(job["result"]["checkpoints"], ["/tmp/checkpoint-7"])
        self.assertEqual(record["needs_user"], [])
        self.assertEqual(self.queue.locate("train"), "deferred")
        self.assertIn("/tmp/checkpoint-7", (self.root / "night-reports/2026-10-01.md").read_text())
        # The next night picks the deferred job up again as attempt 2.
        (self.root / "honour.py").write_text("import sys; sys.exit(0)\n")
        tomorrow = Window("2026-10-02", "2026-10-02", 90000.0, 91000.0, 91100.0, 91200.0)
        record = self.run_quietly(self.runner(window=tomorrow))
        self.assertEqual([(job["attempt"], job["outcome"]) for job in record["jobs"]], [(2, "done")])
        self.assertEqual(len(self.queue.load("done", "train")["attempts"]), 2)

    def test_ignoring_job_and_its_detached_child_are_killed_at_hard_stop(self):
        pid_file = self.root / "child.pid"
        self.add("stubborn", [sys.executable, self.script("ignore", IGNORE), str(pid_file)], resumable=True)
        record = self.run_quietly(self.runner())
        job = record["jobs"][0]
        self.assertEqual((job["outcome"], job["reason"]), ("failed", "hard_stop_killed"))
        self.assertIsNotNone(job["graceful_stop_at"])
        self.assertIsNotNone(job["hard_kill_at"])
        self.assertEqual(job["exit_code"], -9)
        child = int(pid_file.read_text())
        self.assertIn(child, job["killed_pids"])
        self.assertFalse(alive(child))
        self.assertTrue(any("stubborn failed (hard_stop_killed)" in item for item in record["needs_user"]))
        self.assertEqual(self.queue.locate("stubborn"), "failed")

    def test_processes_left_behind_are_killed_and_reported(self):
        pid_file = self.root / "leftover.pid"
        self.add("leaky", [sys.executable, self.script("leave", LEAVE), str(pid_file)])
        record = self.run_quietly(self.runner())
        child = int(pid_file.read_text())
        self.assertEqual(record["jobs"][0]["outcome"], "done")
        self.assertIn(child, record["jobs"][0]["leftover_pids"])
        self.assertFalse(alive(child))
        self.assertTrue(any("left processes behind" in item for item in record["needs_user"]))

    def test_operator_stop_requests_graceful_stop_then_kills(self):
        self.add("stubborn", [sys.executable, self.script("ignore", IGNORE), str(self.root / "child.pid")])
        runner = self.runner(operator_grace_seconds=30)

        class StopSoon(FakeClock):
            def sleep(inner, seconds):
                super().sleep(seconds)
                if inner.value >= 1100:
                    runner.request_stop()

        runner.clock = StopSoon(self.window.start)
        record = self.run_quietly(runner)
        job = record["jobs"][0]
        self.assertEqual(record["stop_reason"], "operator_stop")
        self.assertEqual(job["reason"], "hard_stop_killed")
        killed = datetime.fromisoformat(job["hard_kill_at"]).timestamp()
        self.assertLess(killed, 1200)

    def test_failed_preflight_retries_then_stops_for_user(self):
        self.add("gpu-job", ["true"])
        probes = FakeProbes(gpu_used=4762)
        record = self.run_quietly(self.runner(probes=probes))
        self.assertEqual(record["stop_reason"], "preflight_failed")
        self.assertEqual(len(record["preflight"]), 4)  # at 1000, 1300, 1600 and 1900
        self.assertEqual(record["jobs"], [])
        self.assertEqual(self.queue.locate("gpu-job"), "pending")
        self.assertIn("gpu_idle", record["needs_user"][0])

    def test_order_requirements_and_fit(self):
        order = self.root / "order.txt"
        append = [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write(sys.argv[2])", str(order)]
        self.add("second", [*append, "2"], priority=10, requires=["first"])
        self.add("first", [*append, "1"], priority=20)
        self.add("too-long", ["true"], priority=0, estimated_minutes=30)  # 1800 s > 1000 s left
        self.add("blocked", ["true"], requires=["broken"])
        self.add("broken", ["false"], priority=90)
        record = self.run_quietly(self.runner())
        self.assertEqual(order.read_text(), "12")
        self.assertEqual([job["id"] for job in record["jobs"]], ["first", "second", "broken"])
        self.assertEqual({item["id"]: item["reason"] for item in record["skipped"]},
                         {"too-long": "does_not_fit_before_deadline", "blocked": "requires_failed_job"})

    def test_no_start_close_to_deadline_or_before_window(self):
        self.add("late", ["true"], resumable=True)
        record = self.run_quietly(self.runner(clock=FakeClock(1950)))
        self.assertEqual((record["stop_reason"], record["jobs"]), ("too_close_to_deadline", []))
        record = self.run_quietly(self.runner(clock=FakeClock(0)))
        self.assertEqual(record["stop_reason"], "before_window")
        self.assertEqual(self.queue.locate("late"), "pending")

    def test_missing_environment_fails_without_starting(self):
        self.add("needs-key", ["true"], env=["ZAI_API_KEY"])
        record = self.run_quietly(self.runner())
        self.assertEqual(record["jobs"][0]["reason"], "missing_env:ZAI_API_KEY")
        self.assertIsNone(record["jobs"][0]["exit_code"])
        self.assertEqual(self.queue.locate("needs-key"), "failed")

    def test_stale_running_job_is_recovered(self):
        self.add("lost", ["true"])
        job = self.queue.load("pending", "lost")
        job["attempts"] = [{"attempt": 1, "pgid": 2**22 + 12345}]
        self.queue.move("lost", "pending", "running", job)
        record = self.run_quietly(self.runner())
        self.assertEqual(self.queue.locate("lost"), "failed")
        self.assertIn("lost runner", record["needs_user"][0])

    def test_second_runner_is_refused(self):
        import fcntl
        self.queue.ensure()
        with open(self.queue.root / "runner.lock", "a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with self.assertRaisesRegex(QueueError, "Another night runner"):
                self.run_quietly(self.runner())

    def test_post_window_check_is_logged_into_the_report(self):
        now = datetime(2026, 10, 1, 9, 5, tzinfo=BERLIN).timestamp()
        self.run_quietly(self.runner())
        check = post_window_check(self.root, now, probes=FakeProbes())
        self.assertTrue(check["ok"])
        report = json.loads((self.root / "night-reports/2026-10-01.json").read_text())
        self.assertTrue(report["post_window_check"]["ok"])
        self.assertIn("Post-window check", (self.root / "night-reports/2026-10-01.md").read_text())
        self.assertEqual(len((self.root / "night-reports/checks.jsonl").read_text().splitlines()), 1)

    def test_runner_cli_prints_window(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(night_runner.main(["window"]), 0)
        self.assertIn('"deadline"', out.getvalue())


if __name__ == "__main__":
    unittest.main()
