#!/usr/bin/env python3
"""Probe tokenization cost and parity: HF-side token IDs vs llama-server's
/tokenize on every frozen prompt.

The recommended flow sends token IDs (HF tokenizer, manual BOS) so serving
tokenization matches training exactly. This probe measures what the
alternative — sending text and letting the server tokenize — would add to
the ~64 ms TTFT floor, and verifies the two paths produce identical ids.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

import bench  # noqa: E402
from sepalith.campaign_protocol import PromptContext, render_prompt  # noqa: E402

import tokenizers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18087)
    args = parser.parse_args()

    tok = tokenizers.Tokenizer.from_file(str(bench.DEFAULT_TOKENIZER))
    bench.TOKENIZER = tok
    sessions = bench.load_sessions(
        bench.HERE / "sessions.jsonl", bench.HERE / "manifest.json", tok
    )

    config = bench.Config.load(bench.HERE / "configs" / "baseline.json")
    server = bench.Server(config, args.port, bench.ARTIFACTS / "probe-tokenize.log")
    try:
        server.wait_ready()
        conn = HTTPConnection("127.0.0.1", args.port, timeout=60)
        hf_ms: list[float] = []
        srv_ms: list[float] = []
        exact = 0
        exact_with_bos = 0
        n = 0
        for session in sessions:
            for event in session["events"]:
                text = render_prompt(PromptContext.from_mapping(event["context"]))
                t0 = time.monotonic()
                ids = tok.encode(text, add_special_tokens=False).ids
                hf = (time.monotonic() - t0) * 1000.0
                body = json.dumps({"content": text}).encode()
                t0 = time.monotonic()
                conn.request(
                    "POST", "/tokenize", body=body, headers={"Content-Type": "application/json"}
                )
                response = conn.getresponse()
                payload = json.loads(response.read())
                srv = (time.monotonic() - t0) * 1000.0
                server_ids = payload.get("tokens")
                n += 1
                hf_ms.append(hf)
                srv_ms.append(srv)
                if server_ids == ids:
                    exact += 1
                if server_ids == [0, *ids]:
                    exact_with_bos += 1
        conn.close()
        print(
            json.dumps(
                {
                    "prompts": n,
                    "hf_encode_ms_p50": round(statistics.median(hf_ms), 2),
                    "hf_encode_ms_max": round(max(hf_ms), 2),
                    "server_tokenize_ms_p50": round(statistics.median(srv_ms), 2),
                    "server_tokenize_ms_p95": round(sorted(srv_ms)[int(0.95 * len(srv_ms))], 2),
                    "server_tokenize_ms_max": round(max(srv_ms), 2),
                    "ids_exact_match": exact,
                    "ids_exact_match_with_bos": exact_with_bos,
                }
            )
        )
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
