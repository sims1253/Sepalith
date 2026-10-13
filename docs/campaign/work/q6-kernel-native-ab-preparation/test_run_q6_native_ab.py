#!/usr/bin/env python3
"""CPU-only policy tests for the root-owned Q6 native A/B capsule."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("q6_native_ab_tested", HERE / "run_q6_native_ab.py")
assert SPEC is not None and SPEC.loader is not None
capsule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capsule)


class Q6NativeABPolicyTests(unittest.TestCase):
    def test_profile_has_required_vulkan_shape_and_server_argv(self) -> None:
        argv = capsule.profile_argv(Path("/x/llama-server"), Path("/x/model-Q6_K.gguf"), 18401)
        self.assertEqual(capsule.PROFILE, {
            "context": 4096, "batch": 256, "ubatch": 256, "threads": 6,
            "threads_http": 2, "parallel": 1, "ngl": 99, "cap": 192,
        })
        self.assertEqual(argv[0], "/x/llama-server")
        self.assertIn(("-c", "4096"), list(zip(argv, argv[1:])))
        self.assertIn(("-b", "256"), list(zip(argv, argv[1:])))
        self.assertIn(("-ub", "256"), list(zip(argv, argv[1:])))
        self.assertIn(("-ngl", "99"), list(zip(argv, argv[1:])))
        self.assertIn(("-t", "6"), list(zip(argv, argv[1:])))
        self.assertIn(("--parallel", "1"), list(zip(argv, argv[1:])))

    def test_trace_and_timing_environment_remove_graph_poison_and_force_switches(self) -> None:
        base = {
            "CUDA_VISIBLE_DEVICES": "0", "SEPALITH_VK_TRACE": "old",
            "GGML_BACKEND_PATH": "/bad", "GGML_CUDA_GRAPH_OPT": "1",
            "GGML_VK_Q6K_EXACT_DIV_POISON": "1", "GGML_VK_FORCE_MMVQ": "1",
            "GGML_VK_DISABLE_MMVQ": "1",
        }
        trace, trace_removed = capsule.child_environment(trace=True, base=base)
        timing, timing_removed = capsule.child_environment(trace=False, base=base)
        self.assertEqual(trace["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(trace[capsule.TRACE_ENV], "1")
        self.assertNotIn("GGML_BACKEND_PATH", trace)
        self.assertNotIn("GGML_CUDA_GRAPH_OPT", trace)
        self.assertNotIn("GGML_VK_Q6K_EXACT_DIV_POISON", trace)
        self.assertIn(capsule.TRACE_ENV, timing_removed)
        self.assertNotIn(capsule.TRACE_ENV, timing)
        self.assertEqual(timing["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(set(trace_removed) - {capsule.TRACE_ENV}, set(capsule.REMOVED_ENV) - {capsule.TRACE_ENV})

    def test_memory_floor_parser_is_explicit(self) -> None:
        self.assertEqual(capsule.available_memory_mib("MemTotal: 4 kB\nMemAvailable: 2097152 kB\n"), 2048)
        with self.assertRaises(capsule.CapsuleError):
            capsule.available_memory_mib("MemTotal: 4 kB\n")

    def test_artifact_receipt_binds_binary_bytes_and_hash(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama-server"
            binary.write_bytes(b"synthetic-binary")
            receipt = root / "artifacts.json"
            capsule.write_json(receipt, {
                "arm": "baseline", "status": "compiled_only",
                "artifacts": [{
                    "path": str(binary), "bytes": binary.stat().st_size,
                    "sha256": capsule.sha256_file(binary),
                }],
            })
            bound = capsule.verify_artifact(binary, receipt, "baseline")
            self.assertEqual(bound["sha256"], capsule.sha256_file(binary))
            bad = json.loads(receipt.read_text())
            bad["artifacts"][0]["sha256"] = "0" * 64
            bad_path = root / "bad.json"
            capsule.write_json(bad_path, bad)
            with self.assertRaises(capsule.CapsuleError):
                capsule.verify_artifact(binary, bad_path, "baseline")

    def test_stable_comparison_allows_only_timing_differences(self) -> None:
        base_record = {
            "row_id": "r", "phase": "cold", "rep": 1,
            "protocol_status": "accepted", "returned_token_ids": [42, 1],
            "raw_text": ">>>>>>> UPDATED", "raw_text_sha256": "x",
            "cap": 192, "context_size": 4096, "cap_status": "within_cap",
            "stop": True, "stop_type": "eos", "stopping_word": None,
            "truncated": False, "tokens_evaluated": 10, "tokens_predicted": 2,
            "tokens_cached": 0, "n_tokens_cached": 0, "n_prompt_tokens_cache": 0,
            "cache_prompt": False, "prompt_token_count": 9,
            "prompt_ids_sha256": "p", "prompt_sha256": "q",
            "prompt_id_status": "exact_stored_prompt",
            "parsed_output": {"status": "accepted"},
            "protocol_checks": {"stop": "eos"},
            "completion_wall_ms": 1.0, "timings": {"prompt_n": 10},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(json.dumps({"requests": [base_record]}))
            changed = dict(base_record)
            changed["completion_wall_ms"] = 99.0
            changed["timings"] = {"prompt_n": 10, "predicted_n": 2}
            right.write_text(json.dumps({"requests": [changed]}))
            self.assertEqual(capsule.compare_stable_fields(left, right)["status"], "pass")
            changed["stop_type"] = "length"
            right.write_text(json.dumps({"requests": [changed]}))
            self.assertEqual(capsule.compare_stable_fields(left, right)["status"], "fail")

    def test_import_does_not_start_a_process_or_need_framework(self) -> None:
        self.assertNotIn("torch", capsule.__dict__)
        self.assertNotIn("transformers", capsule.__dict__)
        self.assertIn("main", capsule.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
