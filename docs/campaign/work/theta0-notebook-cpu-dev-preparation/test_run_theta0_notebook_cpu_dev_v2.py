#!/usr/bin/env python3
"""Bounded static tests for the v2 split CPU comparator capsule."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("theta0_notebook_cpu_v2", HERE / "run_theta0_notebook_cpu_dev_v2.py")
assert SPEC and SPEC.loader
CAPSULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPSULE)


class NotebookCpuV2Tests(unittest.TestCase):
    def test_matched_profile_uses_quality_cap_512_and_cpu_server(self) -> None:
        self.assertEqual(CAPSULE.PROFILE["context"], 4096)
        self.assertEqual(CAPSULE.PROFILE["quality_cap"], 512)
        self.assertEqual(CAPSULE.PROFILE["ngl"], 0)
        argv = CAPSULE.server_argv("/bin/llama-server", "/model.gguf", 18413)
        self.assertIn(("-c", "4096"), set(zip(argv, argv[1:])))
        self.assertIn(("-ngl", "0"), set(zip(argv, argv[1:])))
        self.assertIn(("-b", "256"), set(zip(argv, argv[1:])))
        self.assertIn(("-ub", "256"), set(zip(argv, argv[1:])))

    def test_scorer_and_tokenizer_are_explicitly_local(self) -> None:
        argv = CAPSULE.client_argv(Path("/tmp/run"), 18413)
        self.assertIn(str(CAPSULE.LOCAL_PYTHON), argv)
        self.assertIn("--panel", argv)
        self.assertIn(str(CAPSULE.PANEL), argv)
        self.assertIn("--tokenizer-dir", argv)
        self.assertIn(str(CAPSULE.TOKENIZER), argv)
        self.assertIn("--execution-root", argv)
        self.assertIn(str(CAPSULE.EXECUTION_ROOT), argv)
        self.assertIn("--deadline-seconds", argv)
        self.assertNotIn("--cap", argv)
        self.assertIn("http://127.0.0.1:18413", argv)

    def test_ssh_command_is_remote_server_only(self) -> None:
        command = CAPSULE.ssh_server_argv(
            "/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/x.staging",
            "/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/x", 18413)
        self.assertIn("-L", command)
        self.assertIn("python3", command)
        self.assertIn(CAPSULE.REMOTE_HELPER.name, command[command.index("python3") + 1])
        self.assertNotIn(str(CAPSULE.PLAN), command)
        self.assertIn(CAPSULE.REMOTE_HOST, command)

    def test_environment_is_cpu_and_clears_backend_overrides(self) -> None:
        old = {"GGML_BACKEND_PATH": "/bad", "GGML_CUDA_GRAPH_OPT": "1",
               "SEPALITH_VK_TRACE": "1"}
        saved = {key: os.environ.get(key) for key in old}
        try:
            os.environ.update(old)
            env, removed = CAPSULE.cpu_environment()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertNotIn("GGML_BACKEND_PATH", env)
        self.assertNotIn("SEPALITH_VK_TRACE", env)
        self.assertIn("GGML_CUDA_GRAPH_OPT", removed)

    def test_import_has_no_framework_or_launch_side_effect(self) -> None:
        self.assertNotIn("torch", CAPSULE.__dict__)
        self.assertNotIn("transformers", CAPSULE.__dict__)
        self.assertIn("preflight", CAPSULE.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
