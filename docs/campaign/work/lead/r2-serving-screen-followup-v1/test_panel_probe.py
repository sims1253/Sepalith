#!/usr/bin/env python3
"""CPU-only synthetic HTTP test for the 4096-context panel client."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import paired_panel_probe as probe


PACKET = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    prompt_by_text: dict[str, list[int]] = {}
    completion_bodies: list[dict[str, object]] = []
    tokenize_bodies: list[dict[str, object]] = []

    def log_message(self, *_args: object) -> None:
        return

    def _json_body(self) -> dict[str, object]:
        size = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(size))
        assert isinstance(value, dict)
        return value

    def _send_json(self, value: object) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        body = self._json_body()
        if self.path == "/tokenize":
            Handler.tokenize_bodies.append(body)
            text = body.get("content")
            self._send_json({"tokens": Handler.prompt_by_text[str(text)]})
            return
        if self.path != "/completion":
            self.send_error(404)
            return
        Handler.completion_bodies.append(body)
        prompt = body["prompt"]
        assert isinstance(prompt, list) and prompt and prompt[0] == 0
        event = {
            "content": "[NO_EDIT]\n>>>>>>> UPDATED",
            "tokens": [42, 1],
            "stop": False,
        }
        final = {
            "content": "",
            "tokens": [],
            "stop": True,
            "stop_type": "eos",
            "truncated": False,
            "tokens_evaluated": len(prompt),
            "timings": {"prompt_n": len(prompt), "predicted_n": 2},
        }
        payload = (f"data: {json.dumps(event)}\n\n"
                   f"data: {json.dumps(final)}\n\n"
                   "data: [DONE]\n\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    panel = PACKET / "train-serving-panel.jsonl"
    manifest = PACKET / "train-serving-panel.manifest.json"
    rows, _ = probe.load_panel(panel, manifest)
    # Twenty short rows plus one natural long row prove that the client does
    # not inherit the old 2048 prompt admission.  Keep the test tiny in wire
    # work while exercising both cold and warm calls.
    rows = rows[:21]
    assert max(row["prompt_token_count"] for row in rows) > 2048
    Handler.prompt_by_text = {
        row["prompt_text"]: row["input_ids"][1:row["target_start"]] for row in rows
    }
    Handler.completion_bodies = []
    Handler.tokenize_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="r2-panel-probe-test-") as tmp:
            args = type("Args", (), {
                "url": f"http://127.0.0.1:{server.server_port}",
                "panel": str(panel), "manifest": str(manifest), "out": str(Path(tmp) / "result.json"),
                "cap": 192, "context": 4096, "reps": 1, "row_limit": 21, "deadline_ms": 5000,
            })()
            result = probe.run_probe(args)
            assert result["status"] == "completed"
            assert result["summary"]["requests"] == result["summary"]["accepted_requests"] == 42
            assert len(Handler.tokenize_bodies) == 42
            assert len(Handler.completion_bodies) == 42
            assert all(body["add_special"] is False and body["parse_special"] is False
                       and body["with_pieces"] is False for body in Handler.tokenize_bodies)
            assert all(body["n_predict"] == 192 and body["temperature"] == 0
                       and body["stream"] is True and body["return_tokens"] is True
                       for body in Handler.completion_bodies)
            assert [body["cache_prompt"] for body in Handler.completion_bodies[:2]] == [False, True]
            assert all(body["prompt"][0] == 0 for body in Handler.completion_bodies)
            assert probe.parse_wire_output("[NO_EDIT]\n>>>>>>> UPDATED\nextra")["status"] == "invalid"
            assert probe.parse_wire_output("[NO_EDIT]\n>>>>>>> UPDATED")["status"] == "accepted"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print("4096-context cold/warm synthetic panel probe PASS: 42/42")


if __name__ == "__main__":
    sys.exit(main())
