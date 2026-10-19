#!/usr/bin/env python3
"""Probe the two-slot warm-up lane: --parallel 2, warm-ups pinned to slot 1,
real requests to slot 0 (or auto). Questions, on a fresh server with a cold
2438-token prompt:

A. cold control: real request on slot 0 (and auto) — the no-warm-up cost;
B. cross-slot reuse: warm-up completes on slot 1, then real request on
   slot 0 — does the unified KV pool restore the warmed prefix?
C. concurrency: warm-up mid-flight on slot 1, real request on slot 0 sent
   immediately — does it run without queuing behind the warm-up?
D. auto-selection: warm-up mid-flight on slot 1, real request with no
   id_slot — does slot selection avoid the busy slot?
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

import bench  # noqa: E402
import tokenizers  # noqa: E402


def real_request(port: int, ids: list[int], decode: int, slot: int | None) -> dict[str, object]:
    began = time.monotonic()
    result = bench.stream_completion(
        port, ids, decode, 3407, cache_prompt=True, id_slot=slot
    )
    return {
        "ttft_ms": result["ttft_ms"],
        "cache_n": result["timings"].get("cache_n"),
        "processed_n": result["timings"].get("prompt_n"),
        "wall_s": round(time.monotonic() - began, 2),
    }


def start_warmup(port: int, ids: list[int], slot: int | None) -> threading.Thread:
    def run() -> None:
        try:
            bench.warmup_completion(port, ids, id_slot=slot)
        except Exception as error:  # noqa: BLE001 — probe reports and continues
            print(json.dumps({"warmup_error": str(error)}))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--abort-after-s", type=float, default=5.0)
    parser.add_argument("--decode", type=int, default=8)
    args = parser.parse_args()

    bench.TOKENIZER = tokenizers.Tokenizer.from_file(str(bench.DEFAULT_TOKENIZER))
    sessions = bench.load_sessions(
        bench.HERE / "sessions.jsonl", bench.HERE / "manifest.json", bench.TOKENIZER
    )
    big = max((e for s in sessions for e in s["events"]), key=lambda e: len(e["ids"]))
    print(json.dumps({"probe_prompt_tokens": len(big["ids"])}))

    config = bench.Config.load(bench.HERE / "configs" / "baseline.json")
    # two slots: the warm-up lane candidate
    config = bench.Config(**{**config.__dict__, "server_args": (*config.server_args, "--parallel", "2")})
    server = bench.Server(config, args.port, bench.ARTIFACTS / "probe-parallel.log")
    try:
        server.wait_ready()

        # A: cold control on slot 0
        print(json.dumps({"scenario": "A_cold_slot0", **real_request(args.port, big["ids"], args.decode, 0)}))

        # B: warm-up completes on slot 1, then real request on slot 0
        warm = bench.warmup_completion(args.port, big["ids"], id_slot=1)
        print(json.dumps({"scenario": "B_warmup_done", "warmup_ms": warm["wall_ms"]}))
        print(json.dumps({"scenario": "B_cross_slot_reuse", **real_request(args.port, big["ids"], args.decode, 0)}))

        # C: warm-up mid-flight on slot 1, real request on slot 0 immediately
        start_warmup(args.port, big["ids"], 1)
        time.sleep(args.abort_after_s)
        print(json.dumps({"scenario": "C_concurrent_slot0", **real_request(args.port, big["ids"], args.decode, 0)}))

        # D: warm-up mid-flight on slot 1, real request with auto slot choice
        start_warmup(args.port, big["ids"], 1)
        time.sleep(1.0)
        print(json.dumps({"scenario": "D_auto_slot", **real_request(args.port, big["ids"], args.decode, None)}))
        time.sleep(2.0)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
