import json
from pathlib import Path
import subprocess
import tempfile
import unittest

PLAN = Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb")
E = PLAN / "docs/campaign/work/lead/r2-expanded-sft-e"
D_SOURCE = PLAN / "docs/campaign/work/lead/r2-expanded-sft-d/source"
RUNNER = Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/packages/sepalith/src/sepalith/runner.py")
PYTHON = Path("/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python")
SOURCE_ID = "bb2f9fdc1d016679533beab7d9906c9b996c9ce59d36e54ff98101adbaeed384"


class SnapshotReplayTest(unittest.TestCase):
    def test_exact_d_source_replays_to_frozen_runner_identity(self):
        with tempfile.TemporaryDirectory(prefix="sft11-expanded-e-snapshot-replay-") as state:
            result = subprocess.run(
                [str(PYTHON), "-B", str(RUNNER), "--state", state, "snapshot", "--repo", str(D_SOURCE), "--include", "experiments", "--include", "packages"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(json.loads(result.stdout), SOURCE_ID)
            enqueue = subprocess.run(
                [str(PYTHON), "-B", str(RUNNER), "--state", state, "enqueue", str(E / "runner-recipe.json")],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertTrue(json.loads(enqueue.stdout))


if __name__ == "__main__":
    unittest.main()
