#!/usr/bin/env python3
"""Probe llama-server behavior when the real request arrives during a warm-up.

Two scenarios, single slot (--parallel 1):
1. overlap: fire a long n_predict=0 warm-up, then immediately the real
   completion for the same prompt — does the real request queue behind the
   warm-up, and what TTFT results?
2. cancel: fire the warm-up, abort the client connection 500ms in, then send
   the real request — how much warm-up progress survives / how fast is the
   slot free?
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

import bench  # noqa: E402
import tokenizers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=bench.HERE / "configs" / "baseline.json")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--abort-after-s", type=float, default=5.0)
    parser.add_argument("--decode", type=int, default=8)
    args = parser.parse_args()

    bench.TOKENIZER = tokenizers.Tokenizer.from_file(str(bench.DEFAULT_TOKENIZER))
    sessions = bench.load_sessions(
        bench.HERE / "sessions.jsonl", bench.HERE / "manifest.json", bench.TOKENIZER
    )
    # a long prompt: last session, biggest event
    big = max(
        (e for s in sessions for e in s["events"]), key=lambda e: len(e["ids"])
    )
    print(json.dumps({"probe_prompt_tokens": len(big["ids"])}))

    config = bench.Config.load(args.config)
    server = bench.Server(config, args.port, bench.ARTIFACTS / "probe-interrupt.log")
    try:
        server.wait_ready()

        import http.client
        import threading

        def start_warmup() -> tuple[http.client.HTTPConnection, float]:
            body = json.dumps(
                bench.completion_request(big["ids"], 0, 3407, stream=False)
            ).encode()
            conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=120)
            began = time.monotonic()
            conn.request(
                "POST", "/completion", body=body, headers={"Content-Type": "application/json"}
            )
            return conn, began

        # --- abort probes FIRST (clean KV pool; earlier scenario used a
        # pool-populated state and could not distinguish partial-KV salvage
        # from whole-context restore)
        for label, post_abort_wait_s in (("abort_immediate", 0.0), ("abort_wait2s", 2.0)):
            conn, began = start_warmup()
            time.sleep(args.abort_after_s)
            if conn.sock is not None:
                conn.sock.close()
            conn.close()
            time.sleep(post_abort_wait_s)
            real = bench.stream_completion(args.port, big["ids"], args.decode, 3407)
            print(
                json.dumps(
                    {
                        "scenario": label,
                        "abort_after_s": args.abort_after_s,
                        "real_ttft_ms": real["ttft_ms"],
                        "real_cache_n": real["timings"].get("cache_n"),
                        "real_processed_n": real["timings"].get("prompt_n"),
                    }
                )
            )

        # --- cold control LAST (populates the pool deliberately)
        control = bench.stream_completion(args.port, big["ids"], args.decode, 3407)
        print(
            json.dumps(
                {
                    "scenario": "pool_restore_control",
                    "real_ttft_ms": control["ttft_ms"],
                    "real_cache_n": control["timings"].get("cache_n"),
                    "real_processed_n": control["timings"].get("prompt_n"),
                }
            )
        )

        # --- scenario: overlap (real request queues behind a long warm-up)
        warmup_result: dict[str, float] = {}

        def run_warmup() -> None:
            began = time.monotonic()
            result = bench.warmup_completion(args.port, big["ids"])
            warmup_result["wall_s"] = time.monotonic() - began
            warmup_result["processed"] = result["timings"].get("prompt_n")

        thread = threading.Thread(target=run_warmup)
        thread.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        real = bench.stream_completion(args.port, big["ids"], args.decode, 3407)
        real_wall = time.monotonic() - t0
        thread.join()
        print(
            json.dumps(
                {
                    "scenario": "overlap",
                    "real_ttft_ms": real["ttft_ms"],
                    "real_wall_s": round(real_wall, 2),
                    "real_processed_n": real["timings"].get("prompt_n"),
                    "real_cache_n": real["timings"].get("cache_n"),
                    "warmup_wall_s": round(warmup_result.get("wall_s", -1), 2),
                }
            )
        )
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
