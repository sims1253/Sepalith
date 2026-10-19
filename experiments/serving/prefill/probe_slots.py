#!/usr/bin/env python3
"""Probe slot persistence: --slot-save-path save/restore across a restart.

Product scenario: after a VS Code / llama-server restart every cached
context is gone, so each file's first open pays the full cold prefill even
with the warm-up flow (the warm-up itself must recompute). This probe
measures whether restoring a saved slot from disk brings the first open
back near the floor:

1. server A with --slot-save-path: warm up a big prompt, save slot 0;
2. control: same prompt cold on a fresh server B (no restore) — the
   no-persistence cost;
3. server C (fresh): restore slot 0 from the file, send the request —
   the persistence win.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

import bench  # noqa: E402
import tokenizers  # noqa: E402


def slots_action(port: int, slot: int, action: str, filename: str) -> dict[str, object]:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    body = json.dumps({"filename": filename}).encode()
    conn.request(
        "POST",
        f"/slots/{slot}?action={action}",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    payload = response.read().decode("utf-8", "replace")
    conn.close()
    return {"status": response.status, "body": payload[:400]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-base", type=int, default=18086)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--save-dir", type=Path, default=bench.ARTIFACTS / "slot-save")
    args = parser.parse_args()

    bench.TOKENIZER = tokenizers.Tokenizer.from_file(str(bench.DEFAULT_TOKENIZER))
    sessions = bench.load_sessions(
        bench.HERE / "sessions.jsonl", bench.HERE / "manifest.json", bench.TOKENIZER
    )
    big = max((e for s in sessions for e in s["events"]), key=lambda e: len(e["ids"]))
    print(json.dumps({"probe_prompt_tokens": len(big["ids"])}))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    base = bench.Config.load(bench.HERE / "configs" / "baseline.json")
    save_args = (*base.server_args, "--slot-save-path", str(args.save_dir))

    # --- server A: warm + save
    server_a = bench.Server(
        bench.Config(**{**base.__dict__, "server_args": save_args}),
        args.port_base,
        bench.ARTIFACTS / "probe-slotsave-a.log",
    )
    try:
        server_a.wait_ready()
        warm = bench.warmup_completion(args.port_base, big["ids"])
        print(json.dumps({"phase": "warmup", "wall_ms": warm["wall_ms"], "cache_n": warm["timings"].get("cache_n")}))
        save = slots_action(args.port_base, 0, "save", "probe-ctx.bin")
        print(json.dumps({"phase": "save", **save}))
    finally:
        server_a.stop()

    saved = args.save_dir / "probe-ctx.bin"
    print(json.dumps({"phase": "saved_file", "exists": saved.exists(), "bytes": saved.stat().st_size if saved.exists() else 0}))

    # --- server B: control, cold open after restart
    server_b = bench.Server(base, args.port_base + 1, bench.ARTIFACTS / "probe-slotsave-b.log")
    try:
        server_b.wait_ready()
        began = time.monotonic()
        cold = bench.stream_completion(args.port_base + 1, big["ids"], args.decode, 3407)
        print(
            json.dumps(
                {
                    "phase": "control_cold_after_restart",
                    "ttft_ms": cold["ttft_ms"],
                    "processed_n": cold["timings"].get("prompt_n"),
                    "cache_n": cold["timings"].get("cache_n"),
                }
            )
        )
    finally:
        server_b.stop()

    # --- server C: restore, then open
    server_c = bench.Server(
        bench.Config(**{**base.__dict__, "server_args": save_args}),
        args.port_base + 2,
        bench.ARTIFACTS / "probe-slotsave-c.log",
    )
    try:
        server_c.wait_ready()
        began = time.monotonic()
        restore = slots_action(args.port_base + 2, 0, "restore", "probe-ctx.bin")
        restore_s = time.monotonic() - began
        print(json.dumps({"phase": "restore", "restore_s": round(restore_s, 2), **restore}))
        warm = bench.stream_completion(args.port_base + 2, big["ids"], args.decode, 3407)
        print(
            json.dumps(
                {
                    "phase": "open_after_restore",
                    "ttft_ms": warm["ttft_ms"],
                    "processed_n": warm["timings"].get("prompt_n"),
                    "cache_n": warm["timings"].get("cache_n"),
                }
            )
        )
    finally:
        server_c.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
