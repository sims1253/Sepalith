"""Exec-env hold-out freeze: verification, manifest and read-only copy (no R needed)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from sepalith.exec_env import freeze

COMMIT = "c" * 40
PRINT = "f" * 16


def write_function(slice_dir: Path, fid: str, *, verdict: str = "pass", commit: str = COMMIT,
                   equivalent: str | None = None, skip: str | None = None, gate: bool = True) -> Path:
    d = slice_dir / fid
    (d / "probes").mkdir(parents=True)
    (d / "source.R").write_text("f <- function(x) {\n  x\n}\n")
    (d / "meta.json").write_text(json.dumps({"id": fid, "package": "pkg", "name": "f", "path": "R/f.R"}))
    if skip:
        (d / "SKIP").write_text(skip)
        return d
    (d / "test.R").write_text('check_identical("a", f(1), 1)\n')
    (d / "probes" / "p1.R").write_text("f <- function(x) {\n  x + 1\n}\n")
    if equivalent is not None:
        (d / "equivalent.txt").write_text(equivalent)
    if gate:
        time.sleep(0.01)
        (d / "gate.json").write_text(json.dumps({
            "verdict": verdict, "problems": [] if verdict == "pass" else ["kill_rate"], "checks": 12,
            "hand_written_checks": 12, "categories": {"a": 12},
            "gate_version": {"name": "v3", "commit": commit, "exec_env_modified": False, "fingerprint": PRINT},
            "mutants": {"generated": 10, "sampled": 10, "valid": 9, "killed": 9, "declared_equivalent": 1,
                        "stale_equivalent_ids": [], "declared_equivalent_but_killed": []}}))
    return d


class FreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "authors"
        for slice_id in ("H000", "H001"):
            (self.root / slice_id / ".claim").mkdir(parents=True)
            (self.root / slice_id / ".claim" / "owner").write_text("someone")
        write_function(self.root / "H000", "a1", equivalent=(
            "m1 SCOPED[nrow(x) != ncol(x)]: rectangular branch broken\n"
            "m2 QUIRK[isTRUE(x[1] > 0)] ATTR[iter]: incidental\nm3 plain reason\n"))
        write_function(self.root / "H000", "a2", skip="output only: behaviour is printed text")
        write_function(self.root / "H001", "b1")

    def tearDown(self) -> None:
        for path in Path(self.tmp.name).rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        self.tmp.cleanup()

    def run_freeze(self, *extra: str) -> int:
        return freeze.main(["--slices", "H000..H001", "--gate", COMMIT, "--fingerprint", PRINT,
                            "--root", str(self.root), "--out", str(Path(self.tmp.name) / "freeze.json"),
                            "--copy-to", str(Path(self.tmp.name) / "frozen"), *extra])

    def test_expand_slices(self):
        self.assertEqual(freeze.expand_slices("H000..H003,S010"), ["H000", "H001", "H002", "H003", "S010"])

    def test_freeze_writes_manifest_and_read_only_copy(self):
        self.assertEqual(self.run_freeze(), 0)
        manifest = json.loads((Path(self.tmp.name) / "freeze.json").read_text())
        self.assertEqual(manifest["counts"], {"functions": 3, "frozen": 2, "skipped": 1})
        self.assertEqual(manifest["gate"], {"commit": COMMIT, "fingerprint": PRINT})
        self.assertEqual(manifest["skipped"][0]["reason"], "output only: behaviour is printed text")
        self.assertEqual([u["mutant"] for u in manifest["untested_branches"]], ["m1"])
        a1 = next(f for f in manifest["functions"] if f["function"] == "a1")
        self.assertEqual({d["tag"] for d in a1["conditioned_declarations"]}, {"SCOPED", "QUIRK"})
        self.assertEqual(set(a1["sha256"]), {"source.R", "meta.json", "test.R", "equivalent.txt", "gate.json"})
        self.assertIn("p1.R", a1["probes"])
        # fake gate.json has no differential record: counted as low replay coverage, limitation noted
        self.assertEqual(len(manifest["low_replay_coverage"]), 2)
        self.assertTrue(manifest["known_limitations"])
        copy = Path(self.tmp.name) / "frozen"
        self.assertTrue((copy / "H000" / "a1" / "test.R").is_file())
        self.assertFalse((copy / "H000" / ".claim").exists())
        if manifest["read_only_copy"]["write_permission_removed"]:
            self.assertFalse(os.access(copy / "H000" / "a1" / "test.R", os.W_OK))
        self.assertEqual(self.run_freeze(), 1)  # never overwrites a freeze

    def test_check_writes_nothing(self):
        self.assertEqual(self.run_freeze("--check"), 0)
        self.assertFalse((Path(self.tmp.name) / "freeze.json").exists())
        self.assertFalse((Path(self.tmp.name) / "frozen").exists())

    def test_problems_block_the_freeze(self):
        cases = {
            "wrong_gate": dict(commit="d" * 40),
            "failing": dict(verdict="fail"),
            "ungated": dict(gate=False),
        }
        for fid, kwargs in cases.items():
            write_function(self.root / "H001", fid, **kwargs)
        stale = self.root / "H001" / "b1" / "test.R"
        later = time.time() + 30
        os.utime(stale, (later, later))
        manifest, problems = freeze.build_manifest(self.root, ["H000", "H001"], COMMIT, PRINT)
        text = "\n".join(problems)
        self.assertIn("wrong_gate: gated at " + "d" * 40, text)
        self.assertIn("failing: gate verdict fail", text)
        self.assertIn("ungated: test.R was never gated", text)
        self.assertIn("b1: changed after gating: test.R", text)
        self.assertEqual(self.run_freeze(), 1)
        self.assertFalse((Path(self.tmp.name) / "freeze.json").exists())

    def test_fingerprint_matches_gate_version_at_a_commit(self):
        # the recomputation from git must agree with what the gate records for a clean tree
        from sepalith.exec_env.gate import gate_version
        live = gate_version()
        if live["commit"] is None or live["exec_env_modified"]:
            self.skipTest("needs a clean git checkout")
        self.assertEqual(freeze.fingerprint_at(live["commit"]), live["fingerprint"])


if __name__ == "__main__":
    unittest.main()
