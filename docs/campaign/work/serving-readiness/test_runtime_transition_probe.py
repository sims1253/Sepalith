#!/usr/bin/env python3
"""CPU-only fake-server tests for runtime_transition_probe.py."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("runtime_transition_probe_tested", HERE / "runtime_transition_probe.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeState:
    def __init__(self, row: dict[str, Any], *, delay_first_tokenize: bool = False,
                 delay_first_completion: bool = True) -> None:
        self.row = row
        self.expected_prompt = list(row["input_ids"][1:row["target_start"]])
        self.delay_first_tokenize = delay_first_tokenize
        self.delay_first_completion = delay_first_completion
        self.tokenize_calls = 0
        self.completion_calls = 0
        self.bodies: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def state(self) -> FakeState:
        return self.server.state  # type: ignore[attr-defined]

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        assert isinstance(value, dict)
        with self.state.lock:
            self.state.bodies.append((self.path, value))
        return value

    def _json(self, value: object) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._body()
        if self.path == "/tokenize":
            with self.state.lock:
                self.state.tokenize_calls += 1
                call = self.state.tokenize_calls
            if call == 1 and self.state.delay_first_tokenize:
                time.sleep(0.12)
            self._json({"tokens": self.state.expected_prompt})
            return
        if self.path != "/completion":
            self.send_error(404)
            return
        with self.state.lock:
            self.state.completion_calls += 1
            call = self.state.completion_calls
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if call == 1 and self.state.delay_first_completion:
            self.wfile.write(b'data: {"content":"","tokens":[]}\n\n')
            self.wfile.flush()
            time.sleep(0.12)
            try:
                self.wfile.write(b'data: {"content":"late","tokens":[42]}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return
        prompt = body.get("prompt")
        final = {
            "content": "",
            "tokens": [],
            "stop": True,
            "stop_type": "eos",
            "tokens_evaluated": len(prompt) if isinstance(prompt, list) else None,
            "timings": {"cache_n": 0, "prompt_n": len(prompt) if isinstance(prompt, list) else 0,
                        "predicted_n": 2, "queued_wait_ms": 3.5},
        }
        events = [
            {"content": "", "tokens": []},
            {"content": "replacement\n", "tokens": [42]},
            {"content": ">>>>>>> UPDATED", "tokens": [1]},
            final,
        ]
        for event in events:
            try:
                self.wfile.write(("data: " + json.dumps(event, separators=(",", ":")) + "\n\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return


class FakeServer:
    def __init__(self, state: FakeState) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.server.state = state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "FakeServer":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class TransitionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.manifest, cls.native_manifest = probe.load_inputs(
            HERE / "native-probe-train-fixture.jsonl",
            HERE / "native-probe-train-fixture.manifest.json",
            HERE / "runtime-transition-probe.manifest.json",
        )
        cls.row = cls.rows[0]

    def baseline(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "phase": "cold",
            "deadline_ms": 60000,
            "records": {
                self.row["id"]: {
                    "row_id": self.row["id"], "phase": "cold", "rep": 1,
                    "protocol_status": "accepted",
                    "raw_text": "replacement\n>>>>>>> UPDATED",
                    "raw_text_sha256": probe.native.sha256_text("replacement\n>>>>>>> UPDATED"),
                    "returned_token_ids": [42, 1],
                    "user_deadline_ms": 60000,
                }
            },
        }

    def run_one(self, state: FakeState, *, cancel_ms: int = 35) -> dict[str, Any]:
        with FakeServer(state) as server:
            return probe.run_transition(
                server.url, self.row, 64, cancel_ms, 192, 4096, 500,
                100, self.baseline(), time.monotonic_ns() + 5000 * 1_000_000,
            )

    def test_cancel_closes_socket_retry_is_exact_and_no_target_leaks(self) -> None:
        state = FakeState(self.row, delay_first_completion=True)
        with patch.object(probe.native, "diagnostic", side_effect=AssertionError("diagnostic must not delay retry")):
            result = self.run_one(state)
        self.assertEqual(result["protocol_status"], "completed")
        cold = result["cold"]
        retry = result["retry"]
        self.assertTrue(cold["cancelled"])
        self.assertEqual(cold["protocol_status"], "cancelled_after_deadline")
        self.assertIsNotNone(cold["completion"]["cancel_initiated_monotonic_ns"])
        self.assertGreaterEqual(cold["completion"]["socket_closed_monotonic_ns"],
                                cold["completion"]["cancel_initiated_monotonic_ns"])
        self.assertEqual(retry["protocol_status"], "accepted")
        self.assertEqual(result["identity"]["cold_retry_full_prompt_exact"], True)
        self.assertEqual(result["baseline_comparison"]["status"], "pass")
        self.assertEqual(retry["cache_n"], 0)
        self.assertEqual(retry["prompt_n"], len(self.row["input_ids"][1:self.row["target_start"]]) + 1)
        self.assertIsNone(retry["queued_wait_ms"])
        self.assertEqual(retry["queued_wait_status"], "unmeasured_no_log_correlation")
        self.assertEqual(retry["server_queue_hint_ms"], 3.5)
        forbidden = {"target_text", "target_body_tokens", "target_operation", "family",
                     "package_id", "renderer_id", "reward", "label", "completion"}
        for path, body in state.bodies:
            self.assertTrue(set(body).isdisjoint(forbidden), (path, body.keys()))
            if path == "/completion":
                self.assertEqual(body["prompt"], [0, *state.expected_prompt])
        self.assertEqual(state.completion_calls, 2)

    def test_whole_deadline_includes_tokenize_and_does_not_start_completion(self) -> None:
        state = FakeState(self.row, delay_first_tokenize=True, delay_first_completion=False)
        result = self.run_one(state, cancel_ms=35)
        cold = result["cold"]
        self.assertEqual(cold["protocol_status"], "timeout")
        self.assertTrue(cold["timed_out"])
        self.assertNotIn("completion", cold)
        self.assertEqual(state.completion_calls, 1)  # retry only
        self.assertEqual(result["retry"]["protocol_status"], "accepted")
        self.assertEqual(result["diagnostic"]["status"], "not_requested")

    def test_completion_before_cancel_deadline_is_valid_and_separate(self) -> None:
        result = self.run_one(FakeState(self.row, delay_first_completion=False), cancel_ms=500)
        self.assertEqual(result["protocol_status"], "completed")
        self.assertEqual(result["cold_outcome"], "completed_before_cancel_deadline")
        self.assertGreaterEqual(result["retry"]["combined_case_wall_ms"], result["retry"]["end_wall_ms"])

    def test_plan_requires_two_deadlines_and_caps_eight_sequences(self) -> None:
        self.assertEqual(probe.parse_deadlines("1000,5000"), (1000, 5000))
        with self.assertRaises(probe.TransitionProbeError):
            probe.parse_deadlines("1000")
        with self.assertRaises(probe.TransitionProbeError):
            probe.validate_plan(64, (1000, 5000), 60000, 192, 4096, 60000, 5)
        self.assertEqual([row["split"] for row in self.rows], ["train"] * 4)
        self.assertEqual(self.manifest["real_editor_transitions"]["status"], "pending")

    def test_rejected_baseline_is_never_called_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({"summary": {"user_deadline_ms": 5000}, "requests": []}))
            baseline = probe._load_baseline(path, "cold", self.rows)
        self.assertEqual(baseline["status"], "rejected_not_60s")
        comparison = probe.compare_retry({"protocol_status": "accepted"}, self.row["id"], baseline)
        self.assertEqual(comparison["status"], "baseline_unavailable")


if __name__ == "__main__":
    unittest.main()
