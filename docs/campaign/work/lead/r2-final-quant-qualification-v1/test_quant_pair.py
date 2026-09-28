#!/usr/bin/env python3
"""Synthetic HTTP test for selected-R2 F16/Q8 client and scorer."""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import quant_pair_probe as probe
import score_quant_pair as scorer


PACKET = Path(__file__).resolve().parent
PANEL = PACKET / "quant-panel.jsonl"
MANIFEST = PACKET / "quant-panel.manifest.json"


class Handler(BaseHTTPRequestHandler):
    by_text: dict[str, list[int]] = {}
    by_prompt: dict[tuple[int, ...], dict[str, object]] = {}
    completion_bodies: list[dict[str, object]] = []

    def log_message(self, *_args: object) -> None:
        return

    def body(self) -> dict[str, object]:
        size = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(size))
        assert isinstance(value, dict)
        return value

    def send_json(self, value: object) -> None:
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        body = self.body()
        if self.path == "/tokenize":
            self.send_json({"tokens": Handler.by_text[str(body["content"])]})
            return
        assert self.path == "/completion"
        Handler.completion_bodies.append(body)
        prompt = body["prompt"]
        assert isinstance(prompt, list) and prompt[0] == 0
        row = Handler.by_prompt[tuple(prompt[1:])]
        text = str(row["target_body_text"]) + "\n>>>>>>> UPDATED"
        event = {"content": text, "tokens": [42, 1], "stop": False}
        final = {"content": "", "tokens": [], "stop": True, "stop_type": "eos",
                 "truncated": False, "tokens_evaluated": len(prompt),
                 "timings": {"prompt_n": len(prompt), "predicted_n": 2}}
        payload = (f"data: {json.dumps(event)}\n\n"
                   f"data: {json.dumps(final)}\n\n"
                   "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_one(server: ThreadingHTTPServer, arm: str, path: str, sha: str, out: Path) -> None:
    args = SimpleNamespace(
        url=f"http://127.0.0.1:{server.server_port}", arm=arm, model_path=path,
        model_sha256=sha, panel=str(PANEL), manifest=str(MANIFEST), out=str(out),
        cap=192, context=4096, reps=1, row_limit=None, deadline_ms=5000,
    )
    result = probe.run_probe(args)
    out.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    rows, _ = probe.load_panel(PANEL, MANIFEST)
    Handler.by_text = {row["prompt_text"]: row["input_ids"][1:row["target_start"]] for row in rows}
    Handler.by_prompt = {tuple(row["input_ids"][1:row["target_start"]]): row for row in rows}
    Handler.completion_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    identity = json.loads(MANIFEST.read_text())["selected_r2_identity"]
    try:
        with tempfile.TemporaryDirectory(prefix="r2-quant-pair-test-") as tmp:
            root = Path(tmp)
            f16 = root / "f16.json"
            q8 = root / "q8.json"
            run_one(server, "f16", identity["f16_path"], identity["f16_sha256"], f16)
            run_one(server, "q8", identity["q8_path"], identity["q8_sha256"], q8)
            result = scorer.score_pair(PANEL, MANIFEST, f16, q8)
            assert result["status"] == "measured_requires_root_review"
            assert result["measurement_complete"] is True
            assert result["f16"]["protocol_valid_rows"] == 8
            assert result["q8"]["protocol_valid_rows"] == 8
            assert result["f16"]["edit_exact_rows"] == result["q8"]["edit_exact_rows"] == 4
            assert result["f16"]["strict_noop_correct_rows"] == result["q8"]["strict_noop_correct_rows"] == 4
            assert result["paired_comparison"]["exact_raw_text_and_token_id_rows"] == 16
            assert len(Handler.completion_bodies) == 32
            assert all(body["n_predict"] == 192 and body["temperature"] == 0
                       and body["cache_prompt"] in {False, True} for body in Handler.completion_bodies)
            incomplete = json.loads(f16.read_text())
            incomplete["requests"] = incomplete["requests"][1:]
            incomplete_path = root / "incomplete.json"
            incomplete_path.write_text(json.dumps(incomplete))
            partial = scorer.score_pair(PANEL, MANIFEST, incomplete_path, q8)
            assert partial["status"] == "partial"
            assert partial["measurement_complete"] is False
            assert partial["f16"]["missing_request_keys"]
            bad_panel = root / "bad-panel.jsonl"
            bad_panel.write_bytes(PANEL.read_bytes().replace(b"dataquieR", b"dataquieX", 1))
            try:
                scorer.score_pair(bad_panel, MANIFEST, f16, q8)
            except ValueError as exc:
                assert "byte hash mismatch" in str(exc)
            else:
                raise AssertionError("scorer accepted panel bytes not pinned by manifest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print("selected-R2 F16/Q8 synthetic pair and scorer PASS: 8/8 each, 16/16 exact parity")


if __name__ == "__main__":
    main()
