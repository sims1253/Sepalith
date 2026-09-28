#!/usr/bin/env python3
"""Static checks for the root-owned b4 editor preparation packet.

These checks deliberately do not import VS Code, spawn a sidecar, inspect model
weights, or contact the network.  The real extension-host checks are executed
only by the root launcher on the notebook.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
PLAN = ROOT.parents[1]
EXEC = pathlib.Path("/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912")
EXT = EXEC / "extensions/vscode-sepalith"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class B4EditorPreparationTests(unittest.TestCase):
    def test_packet_files_are_json_and_harness_is_real_vscode(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        self.assertEqual(package["engines"]["vscode"], "^1.85.0")
        harness = (ROOT / "extension.js").read_text()
        self.assertIn('require("vscode")', harness)
        self.assertIn('sepalith.startServer', harness)
        self.assertIn('editor.action.inlineSuggest.commit', harness)
        self.assertIn("SEPALITH_B4_MANUAL_TRIGGER", harness)
        self.assertNotIn("http.createServer", harness)
        self.assertNotIn("controlled_server", harness)

    def test_b4_route_settings_are_explicit_and_case_independent(self) -> None:
        harness = (ROOT / "extension.js").read_text()
        for value in [
            'settings.manifest_url === ""',
            'backend === "cpu"',
            'settings.threads === 6',
            'settings.context_size === 8192',
            'settings.scope_context === false',
            'max_tokens: 320',
            'stop_count: 7',
        ]:
            self.assertIn(value, harness)
        self.assertIn("request settings are fixed across cases", harness)
        self.assertIn("operation/family labels", harness)

    def test_pinned_sources_and_artifacts_match_acceptance_receipts(self) -> None:
        snapshot = json.loads((PLAN / "receipts/RUN-01-acceptance-source-snapshot.json").read_text())
        expected = {item["path"]: item["sha256"] for item in snapshot["files"]}
        self.assertEqual(sha256(EXT / "src/extension.ts"), expected["src/extension.ts"])
        self.assertEqual(sha256(EXT / "src/runtime.ts"), expected["src/runtime.ts"])
        self.assertEqual(sha256(EXT / "package.json"), expected["package.json"])
        vsix = pathlib.Path(snapshot["VSIX"]["path"])
        self.assertEqual(sha256(vsix), snapshot["VSIX"]["sha256"])
        self.assertEqual(vsix.stat().st_size, snapshot["VSIX"]["bytes"])

        fallback = json.loads((PLAN / "receipts/PRE-08-fallback-ledger.json").read_text())
        b4 = fallback["RESULT"]["matched_banked_artifact"]
        self.assertEqual(b4["sha256"], "e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d")
        self.assertEqual(fallback["RESULT"]["matched_runtime_identities"]["local_cpu_fallback"]["sha256"],
                         "123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804")

    def test_launcher_only_adopts_a_root_owned_sidecar(self) -> None:
        launcher = (ROOT / "run_b4_editor_cycle.mjs").read_text()
        self.assertIn("no_server_started_by_launcher: true", launcher)
        self.assertIn("fresh_external", launcher)
        self.assertIn("existing_external", launcher)
        self.assertNotIn('"llama-server", [', launcher)
        self.assertIn("--extensionDevelopmentPath=${HERE}", launcher)
        result = subprocess.run(
            ["node", str(ROOT / "run_b4_editor_cycle.mjs"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--mode fresh_external|existing_external", result.stdout)


if __name__ == "__main__":
    unittest.main()
