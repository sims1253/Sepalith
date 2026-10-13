#!/usr/bin/env python3
"""CPU-only tests for the notebook theta0 CPU replay capsule."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("theta0_notebook_cpu_tested", HERE / "run_theta0_notebook_cpu_dev.py")
assert SPEC is not None and SPEC.loader is not None
capsule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capsule)


class NotebookCpuPreparationTests(unittest.TestCase):
    def test_profile_is_cpu_4096_512_t6_ngl0(self) -> None:
        self.assertEqual(capsule.PROFILE, {
            "backend": "cpu", "context": 4096, "cap": 192,
            "threads": 6, "threads_batch": 6, "threads_http": 2,
            "parallel": 1, "batch": 256, "ubatch": 256, "ngl": 0,
        })
        argv = capsule.server_argv(Path("/bin/llama-server"), Path("/model.gguf"), 18413)
        pairs = set(zip(argv, argv[1:]))
        self.assertIn(("-c", "4096"), pairs)
        self.assertIn(("-b", "256"), pairs)
        self.assertIn(("-ub", "256"), pairs)
        self.assertIn(("-ngl", "0"), pairs)
        self.assertIn(("-t", "6"), pairs)

    def test_cpu_environment_removes_vulkan_and_cuda_backend_overrides(self) -> None:
        old = {
            "GGML_BACKEND_PATH": "/bad", "GGML_CUDA_GRAPH_OPT": "1",
            "GGML_VK_FORCE_MMVQ": "1", "SEPALITH_VK_TRACE": "1",
        }
        import os
        saved = {key: os.environ.get(key) for key in old}
        try:
            os.environ.update(old)
            env, removed = capsule.cpu_environment()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OMP_NUM_THREADS"], "6")
        self.assertNotIn("GGML_BACKEND_PATH", env)
        self.assertNotIn("GGML_CUDA_GRAPH_OPT", env)
        self.assertNotIn("SEPALITH_VK_TRACE", env)
        self.assertIn("GGML_VK_FORCE_MMVQ", removed)

    def test_memory_parser_and_dri_policy_are_explicit(self) -> None:
        self.assertEqual(capsule.MEMORY_FLOOR_MIB, 2048)
        self.assertFalse(capsule.PROFILE["ngl"])
        audit = capsule.device_audit(os.getpid())
        self.assertEqual(audit["vulkan_mappings"], [])
        self.assertEqual(audit["dri_fds"], [])

    def test_import_is_framework_and_launch_free(self) -> None:
        self.assertNotIn("torch", capsule.__dict__)
        self.assertNotIn("transformers", capsule.__dict__)
        self.assertIn("preflight", capsule.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
