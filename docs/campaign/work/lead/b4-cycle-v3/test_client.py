#!/usr/bin/env python3
"""Focused CPU-only tests for the b4 notebook baseline v2 preparation."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "client.py"
SPEC = importlib.util.spec_from_file_location("run03_b4_notebook_baseline_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class B4NotebookBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = MODULE.authority_contract()
        cls.renderer = MODULE.load_legacy_renderer(MODULE.DEFAULT_REPO_ROOT, cls.contract)
        cls.cases = MODULE.synthetic_cases()
        cls.prepared = [MODULE.prepare_fixture(case, cls.renderer) for case in cls.cases]

    def test_authority_receipts_bind_exact_protected_contract(self) -> None:
        accepted = self.contract["accepted"]
        self.assertEqual(accepted["model_bytes"], 2012011904)
        self.assertEqual(
            accepted["model_sha256"],
            "e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d",
        )
        self.assertEqual(
            accepted["runtime_sha256"],
            "123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804",
        )
        self.assertEqual(
            accepted["renderer_sha256"],
            "7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d",
        )
        self.assertEqual(accepted["parser_sha256"], "368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285")
        self.assertEqual(accepted["stops"], MODULE.EXTENSION_STOPS)
        self.assertEqual(accepted["max_tokens"], 320)

    def test_authority_and_local_model_runtime_metadata_stay_separate(self) -> None:
        expected = self.contract["accepted_artifacts"]
        self.assertEqual(expected["model"]["bytes"], 2012011904)
        self.assertIsNone(expected["runtime"]["bytes"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "protected.gguf"
            runtime = root / "llama-server"
            model.write_bytes(b"model-metadata-only")
            runtime.write_bytes(b"runtime")
            local = MODULE.local_artifact_metadata(model, runtime, self.contract)
            self.assertEqual(local["model"]["bytes"], len(b"model-metadata-only"))
            self.assertEqual(local["runtime"]["bytes"], len(b"runtime"))
            self.assertEqual(local["model"]["sha256"], "not_checked")
            self.assertEqual(local["runtime"]["sha256"], "not_checked")
            self.assertNotEqual(local["model"]["bytes"], local["runtime"]["bytes"])

    def test_request_phase_labels_only_first_server_request_cold(self) -> None:
        self.assertEqual(MODULE.request_phase(0, 0), "cold")
        self.assertEqual(MODULE.request_phase(0, 1), "repeat")
        self.assertEqual(MODULE.request_phase(1, 0), "different_prompt")
        self.assertEqual(MODULE.request_phase(1, 1), "repeat")
        self.assertNotIn("warm", {MODULE.request_phase(i, j) for i in range(2) for j in range(2)})

    def test_inline_geometry_matches_typed_partial_rule(self) -> None:
        case = dict(self.cases[0])
        case["region_old"] = ["sum(x)"]
        case["cursor_character"] = 6
        geometry = MODULE.inline_geometry(case, ["sum(x) + 1"])
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(geometry["range_start_character"], 0)
        self.assertEqual(geometry["range_end_character"], 6)
        self.assertEqual(MODULE.inline_geometry(case, []), None)

    def test_fixture_uses_pinned_renderer_and_no_prm03_prompt(self) -> None:
        self.assertEqual([case["id"] for case in self.cases], [
            "synthetic-b4-replace", "synthetic-b4-no-op"
        ])
        for item in self.prepared:
            self.assertIn("<[fim-suffix]>", item["prompt"])
            self.assertIn("<[fim-middle]>", item["prompt"])
            self.assertNotIn("[NO_EDIT]", item["prompt"])
            self.assertEqual(item["request_payload"]["n_predict"], 320)
            case = item["case"]
            visible = "\n".join(case["prefix"] + case["region_old"] + case["suffix"])
            self.assertGreaterEqual(len(visible.strip()), 30)

    def test_operation_and_family_labels_cannot_change_runtime_request(self) -> None:
        first = self.prepared[0]
        mutated_case = dict(first["case"])
        mutated_case.update({
            "family": "pretend_no_op",
            "operation": "replace",
            "target_lines": ["gold-label-change"],
        })
        mutated = MODULE.prepare_fixture(mutated_case, self.renderer)
        self.assertEqual(first["request_payload"], mutated["request_payload"])
        self.assertEqual(first["tokenize_payload"], mutated["tokenize_payload"])
        self.assertNotEqual(first["target_sha256"], mutated["target_sha256"])

    def test_fake_full_cycle_records_native_tokenizer_latency_and_gaps(self) -> None:
        calls = []

        def fake_request(
            base_url: str,
            endpoint: str,
            payload: dict[str, Any] | None,
            timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            calls.append((endpoint, payload, timeout_s))
            if endpoint == "/tokenize":
                return 200, {"tokens": [101, 102], "with_pieces": False}
            if endpoint == "/completion":
                return 200, {
                    "content": "value <- 2\n>>>>>>> UPDATED",
                    "tokens": [201, 202],
                    "stop": ">>>>>>> UPDATED",
                    "stop_type": "word",
                    "stopping_word": ">>>>>>> UPDATED",
                    "tokens_predicted": 2,
                    "tokens_evaluated": 2,
                    "tokens_cached": 0,
                    "truncated": False,
                    "has_new_line": True,
                    "timings_per_token": [1.0, 1.0],
                }
            raise AssertionError(endpoint)

        first = MODULE.run_cycle(
            "http://127.0.0.1:18099", self.prepared[0], "cold", 2.0,
            self.renderer, request_fn=fake_request,
        )
        repeat = MODULE.run_cycle(
            "http://127.0.0.1:18099", self.prepared[0], "repeat", 2.0,
            self.renderer, request_fn=fake_request,
        )
        self.assertEqual(first["response_ok"], 1)
        self.assertEqual(first["parsed_prediction"], ["value <- 2"])
        self.assertGreaterEqual(first["parsed_suggestion_latency_ms"], 0)
        self.assertEqual(first["application"]["writes"], 0)
        self.assertEqual(first["cancellation"]["status"], "not_exercised")
        self.assertTrue(first["cleanup"]["client_transport_context_closed"])
        self.assertEqual(repeat["phase"], "repeat")
        self.assertEqual(first["request_payload"], repeat["request_payload"])
        self.assertEqual([call[0] for call in calls], [
            "/tokenize", "/completion", "/tokenize", "/completion"
        ])
        for endpoint, payload, _timeout in calls:
            if endpoint == "/completion":
                self.assertEqual(payload["stop"], MODULE.EXTENSION_STOPS)
                self.assertEqual(payload["n_predict"], 320)
                self.assertFalse(payload["cache_prompt"])
        self.assertEqual(first["quality_claim"], "none")
        self.assertEqual(first["scope"], "endpoint_and_parser_diagnostic_only")
        self.assertIsNotNone(first["application"]["inline_geometry"])

    def test_noop_parser_and_application_are_separate_from_gold_scoring(self) -> None:
        def fake_request(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            _timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            if endpoint == "/tokenize":
                return 200, {"tokens": [1]}
            return 200, {
                "content": "<<<<<<< CURRENT\n=======\n<[fim-middle]>\n>>>>>>> UPDATED",
                "tokens_predicted": 0,
                "stop_type": "word",
            }

        record = MODULE.run_cycle(
            "http://127.0.0.1:18099", self.prepared[1], "cold", 2.0,
            self.renderer, request_fn=fake_request,
        )
        self.assertEqual(record["response_ok"], 1)
        self.assertEqual(record["parsed_prediction"], [])
        self.assertEqual(record["application"]["action"], "no_item")
        self.assertTrue(record["application"]["unchanged"])
        self.assertEqual(record["application"]["writes"], 0)
        self.assertEqual(record["quality_claim"], "none")
        self.assertNotIn("no_op_correct", record)

    def test_invalid_native_tokenizer_response_fails_closed(self) -> None:
        def bad_request(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            _timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            if endpoint == "/tokenize":
                return 200, {"tokens": ["not-an-id"]}
            raise AssertionError(endpoint)

        record = MODULE.run_cycle(
            "http://127.0.0.1:18099", self.prepared[0], "cold", 2.0,
            self.renderer, request_fn=bad_request,
        )
        self.assertEqual(record["response_ok"], 0)
        self.assertIn("integer token IDs", record["error"])
        self.assertEqual(record["cancellation"]["status"], "not_exercised")

    def test_preflight_rejects_wrong_model_or_context(self) -> None:
        def bad_model(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            _timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            if endpoint == "/health":
                return 200, {"status": "ok"}
            if endpoint == "/props":
                return 200, {
                    "model_path": "/wrong/protected.gguf",
                    "default_generation_settings": {"n_ctx": 8192},
                }
            return 200, {"data": []}

        with self.assertRaisesRegex(RuntimeError, "model identity"):
            MODULE.server_preflight(
                "http://127.0.0.1:18099", "/protected.gguf", 2.0,
                request_fn=bad_model,
            )

        def bad_context(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            _timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            if endpoint == "/health":
                return 200, {"status": "ok"}
            if endpoint == "/props":
                return 200, {
                    "model_path": "/protected.gguf",
                    "default_generation_settings": {"n_ctx": 4096},
                }
            return 200, {"data": []}

        with self.assertRaisesRegex(RuntimeError, "context"):
            MODULE.server_preflight(
                "http://127.0.0.1:18099", "/protected.gguf", 2.0,
                request_fn=bad_context,
            )

        def unhealthy(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            _timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            if endpoint == "/health":
                return 200, {"status": "loading"}
            raise AssertionError("preflight must stop before non-health endpoints")

        with self.assertRaisesRegex(RuntimeError, "health is not ok"):
            MODULE.server_preflight(
                "http://127.0.0.1:18099", "/protected.gguf", 2.0,
                request_fn=unhealthy,
            )

    def test_loopback_and_deadline_guards(self) -> None:
        with self.assertRaises(ValueError):
            MODULE._base_url("http://example.invalid:18099")
        self.assertEqual(
            MODULE.deadline_settings(10, 2),
            {"soft_s": 10.0, "reserve_s": 2.0, "cycle_s": 5.0, "diagnostic_s": 60.0},
        )
        with self.assertRaises(ValueError):
            MODULE.deadline_settings(0, 2)

    def test_remaining_overall_budget_decreases_between_requests(self) -> None:
        now = [100.0]
        calls = []
        def request(_url, endpoint, _payload, timeout):
            calls.append((endpoint, timeout))
            if endpoint == "/tokenize":
                now[0] += 1.5
                return 200, {"tokens": [101]}
            self.assertAlmostEqual(timeout, .5)
            now[0] += .6
            return 200, {"content": "value <- 1"}
        record = MODULE.run_cycle("http://127.0.0.1:18099", self.prepared[0], "cold", 2.0,
                                  self.renderer, request_fn=request, clock=lambda: now[0])
        self.assertEqual(record["response_ok"], 0)
        self.assertIn("absolute deadline", record["error"])
        self.assertEqual(len(calls), 2)

    def test_actual_splice_ignores_operation_label(self) -> None:
        case = dict(self.cases[1])
        result = MODULE.simulated_application(case, [" + FALSE"])
        case["operation"] = "replace"
        self.assertEqual(result, MODULE.simulated_application(case, [" + FALSE"]))
        self.assertFalse(result["unchanged"])
        self.assertTrue(MODULE.simulated_application(case, ["done <- TRUE"])["unchanged"])
        self.assertTrue(MODULE.simulated_application(case, [])["unchanged"])
        case.update(region_old=["a😀"], cursor_character=3)
        geometry = MODULE.inline_geometry(case, ["x"])
        self.assertEqual(geometry["range_start_character"], 3)
        self.assertFalse(MODULE.simulated_application(case, ["x"])["unchanged"])

    def test_slow_tokenize_leaves_only_residual_completion_deadline(self) -> None:
        now = [100.0]
        calls: list[tuple[str, float]] = []

        def fake_request(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            calls.append((endpoint, timeout_s))
            if endpoint == "/tokenize":
                now[0] += 4.0
                return 200, {"tokens": [101]}
            self.assertEqual(endpoint, "/completion")
            self.assertLessEqual(timeout_s, 1.0 + 1e-9)
            now[0] += 1.1
            return 200, {"content": "value <- 2", "tokens_predicted": 1}

        record = MODULE.run_cycle(
            "http://127.0.0.1:18099",
            self.prepared[0],
            "cold",
            20.0,
            self.renderer,
            request_fn=fake_request,
            cycle_deadline_s=5.0,
            clock=lambda: now[0],
        )
        self.assertEqual(record["response_ok"], 0)
        self.assertIn("absolute deadline", record["error"])
        self.assertEqual([endpoint for endpoint, _timeout in calls], ["/tokenize", "/completion"])
        self.assertGreater(calls[0][1], calls[1][1])

    def test_diagnostic_budget_is_explicit_and_separate(self) -> None:
        calls: list[float] = []

        def fake_request(
            _base_url: str,
            endpoint: str,
            _payload: dict[str, Any] | None,
            timeout_s: float,
        ) -> tuple[int, dict[str, Any]]:
            calls.append(timeout_s)
            if endpoint == "/tokenize":
                return 200, {"tokens": [101]}
            return 200, {"content": "value <- 2", "tokens_predicted": 1}

        record = MODULE.run_cycle(
            "http://127.0.0.1:18099",
            self.prepared[0],
            "cold",
            120.0,
            self.renderer,
            request_fn=fake_request,
            diagnostic=True,
            diagnostic_timeout_s=60.0,
        )
        self.assertEqual(record["response_ok"], 1)
        self.assertEqual(record["deadline"]["mode"], "diagnostic_60s")
        self.assertGreaterEqual(calls[0], 59.0)

    def test_atomic_artifact_paths_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.jsonl"
            path.write_text("kept\n")
            with self.assertRaises(RuntimeError):
                MODULE.fresh_path(path)
            self.assertEqual(path.read_text(), "kept\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
