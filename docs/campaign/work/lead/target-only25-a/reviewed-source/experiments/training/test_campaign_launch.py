"""Exercise the real independent timeout against an owned CPU process group."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

from campaign_control import utc_deadline
from campaign_launch import supervised_command


class LaunchTests(unittest.TestCase):
    def test_absolute_cutoff_overrides_attempt_duration_and_rejects_expired_budget(self):
        recipe = {"deadline": "1970-01-01T00:01:40Z", "max_attempt_seconds": 100,
                  "termination_grace_seconds": 5, "checkpoint_reserve_seconds": 10}
        argv, record = supervised_command(recipe, [sys.executable, "-c", "pass"], now=70)
        self.assertEqual(utc_deadline(record["hard_deadline"]), 100)
        self.assertEqual(utc_deadline(record["soft_deadline"]), 95)
        self.assertIn("25.000000s", argv)
        with self.assertRaisesRegex(ValueError, "No attempt budget"):
            supervised_command(recipe, [sys.executable], now=85)

    def test_timeout_kills_stalled_parent_and_descendant_without_cuda(self):
        code = """import json,os,signal,subprocess,sys,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
print(json.dumps({'parent':os.getpid(),'child':child.pid}),flush=True)
time.sleep(30)
"""
        recipe = {"deadline": datetime.fromtimestamp(time.time() + 10, timezone.utc).isoformat(),
                  "max_attempt_seconds": 1.2, "termination_grace_seconds": 0.3,
                  "checkpoint_reserve_seconds": 0.1}
        argv, _ = supervised_command(recipe, [sys.executable, "-c", code])
        started = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5, start_new_session=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 4)
        identities = json.loads(result.stdout)
        for pid in identities.values():
            status = Path(f"/proc/{pid}/stat")
            if status.exists():
                self.assertIn(status.read_text().rsplit(")", 1)[1].split()[0], ("Z", "X"))


if __name__ == "__main__":
    unittest.main()
