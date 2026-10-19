#!/usr/bin/env python3
"""Does a client abort cancel an in-flight prefill? Server-log evidence.

Complements ``probe_interrupt.py`` (which infers cancellation from the follow-up
request's TTFT/cache_n). Each scenario runs on a FRESH server (no KV-pool
state) and sends NO other HTTP traffic between the abort and the slot release:
polling ``GET /slots`` during the abort window starves llama-server's
disconnect check (the handler only checks ``should_stop`` when its 1 s result
wait times out, and every /slots result wakes the wait), which masks the
cancel entirely — measured on both builds, see the 2026-10-19 report.

Scenarios (per ``--batch`` value; empty = server default n_batch 2048):

- ``control``: full uninterrupted warm-up (n_predict=0): prefill wall time.
- ``abort_nonstream`` / ``abort_stream``: fire the warm-up, close the client
  socket after ``--abort-after-s``, wait for the warm-up task's
  ``stop processing: n_tokens = N`` line in the server log, then send the real
  request (n_predict=--decode). Records whether a ``cancel task`` line was
  logged, the tokens prefilled when the slot released (N < prompt = cancelled
  mid-prefill), the release delay after the abort, and the real request's
  TTFT / cache_n / prompt_n.

The server binary comes from ``SEPALITH_LLAMA_SERVER`` (bench.py convention).
"""
from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "sepalith" / "src"))

import bench
import tokenizers

TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
RELEASE = re.compile(r"task 0 \| stop processing: n_tokens = (\d+)")


def log_seconds(line: str) -> float | None:
    """llama.cpp log prefix m.ss.mmm.uuu (time since process start) -> seconds."""
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def wait_release(log: Path, timeout: float = 600.0) -> tuple[int | None, float | None, float | None]:
    """Poll the server log (no HTTP) for task 0's release line.
    Returns (n_tokens at release, release log-time, cancel log-time)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        text = log.read_text(errors="replace")
        cancel_t = None
        for line in text.splitlines():
            if "cancel task, id_task = 0" in line and cancel_t is None:
                cancel_t = log_seconds(line)
            m = RELEASE.search(line)
            if m:
                return int(m.group(1)), log_seconds(line), cancel_t
        time.sleep(0.1)
    return None, None, None


def start_warmup(port: int, ids: list[int], stream: bool) -> http.client.HTTPConnection:
    body = json.dumps(bench.completion_request(ids, 0, 3407, stream=stream)).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=bench.HERE / "configs" / "baseline.json")
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument("--abort-after-s", type=float, default=3.0)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--batch", nargs="*", default=["", "256"],
                        help="n_batch/n_ubatch values to probe ('' = server default)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bench.TOKENIZER = tokenizers.Tokenizer.from_file(str(bench.DEFAULT_TOKENIZER))
    sessions = bench.load_sessions(
        bench.HERE / "sessions.jsonl", bench.HERE / "manifest.json", bench.TOKENIZER
    )
    big = max((e for s in sessions for e in s["events"]), key=lambda e: len(e["ids"]))
    ids = list(big["ids"])
    base_config = bench.Config.load(args.config)
    results: dict = {"server_bin": str(bench.SERVER_BIN), "label": args.label,
                     "prompt_tokens": len(ids), "abort_after_s": args.abort_after_s,
                     "load_start": open("/proc/loadavg").read().split()[0], "runs": {}}

    for batch in args.batch:
        extra = ("-b", batch, "-ub", batch) if batch else ()
        config = dataclasses.replace(base_config, server_args=base_config.server_args + extra)
        tag = f"b{batch or 'default'}"
        results["runs"][tag] = {}
        for scenario in ("control", "abort_nonstream", "abort_stream"):
            log = bench.ARTIFACTS / f"probe-abort-{args.label}-{tag}-{scenario}.log"
            server = bench.Server(config, args.port, log)
            try:
                server.wait_ready()
                row: dict = {}
                if scenario == "control":
                    t0 = time.monotonic()
                    w = bench.warmup_completion(args.port, ids)
                    row["warmup_wall_s"] = round(time.monotonic() - t0, 2)
                    row["warmup_prompt_n"] = w["timings"].get("prompt_n")
                else:
                    conn = start_warmup(args.port, ids, stream=scenario.endswith("_stream"))
                    t_send = time.monotonic()
                    time.sleep(args.abort_after_s)
                    if conn.sock is not None:
                        conn.sock.close()
                    conn.close()
                    t_abort_rel = time.monotonic() - server.started
                    n_rel, t_rel, t_cancel = wait_release(log)
                    row["abort_at_s_since_spawn"] = round(t_abort_rel, 2)
                    row["cancel_logged"] = t_cancel is not None
                    row["cancel_after_abort_s"] = (
                        round(t_cancel - t_abort_rel, 2) if t_cancel is not None else None)
                    row["released_n_tokens"] = n_rel
                    row["release_after_abort_s"] = (
                        round(t_rel - t_abort_rel, 2) if t_rel is not None else None)
                    row["cancelled_mid_prefill"] = n_rel is not None and n_rel < len(ids)
                    real = bench.stream_completion(args.port, ids, args.decode, 3407)
                    row["real_ttft_ms"] = real["ttft_ms"]
                    row["real_cache_n"] = real["timings"].get("cache_n")
                    row["real_prompt_n"] = real["timings"].get("prompt_n")
                    row["real_done_after_send_s"] = round(time.monotonic() - t_send, 2)
                results["runs"][tag][scenario] = row
                print(json.dumps({tag: {scenario: row}}), flush=True)
            finally:
                server.stop()
    results["load_end"] = open("/proc/loadavg").read().split()[0]
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
