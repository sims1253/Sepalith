#!/usr/bin/env python3
"""Token-only parity probe: flip rate vs draft exposure (depth/n), load-immune.

Runs a baseline server and one spec server per config, cold requests only,
compares greedy token ids per trace. No timing — only divergence counts and
positions. Uses the same deterministic 60-trace subset as the sweep.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from bench_spec import DEFAULT_TRACES, Server, stream_completion

OUTROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/runs"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="name=flags (comma-joined) entries")
    ap.add_argument("--n-traces", type=int, default=60)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--server-bin", type=Path, default=None,
                    help="override llama-server binary (upgrade probes)")
    a = ap.parse_args()

    traces = [json.loads(line) for line in DEFAULT_TRACES.read_text().splitlines() if line.strip()]
    if a.n_traces < len(traces):
        step = len(traces) / a.n_traces
        traces = [traces[int(i * step)] for i in range(a.n_traces)]

    base = Server("pbase", 18611, [], OUTROOT / "probe-pbase.log", threads=a.threads,
                  server_bin=a.server_bin)
    base.start()
    base_tokens: dict[str, list[int]] = {}
    try:
        for tr in traces:
            r = stream_completion(base.port, tr["prompt_ids"], a.n_predict,
                                  cache_prompt=False)
            base_tokens[tr["trace_id"]] = r.tokens or []
    finally:
        base.stop()

    results = {}
    port = 18612
    for entry in a.configs:
        name, _, flags_s = entry.partition("=")
        flags = flags_s.split(",") if flags_s else []
        srv = Server(f"p-{name}", port, flags, OUTROOT / f"probe-{name}.log",
                     threads=a.threads, server_bin=a.server_bin)
        port += 1
        srv.start()
        flips = []
        div_pos = []
        try:
            for tr in traces:
                r = stream_completion(srv.port, tr["prompt_ids"], a.n_predict,
                                      cache_prompt=False)
                toks = r.tokens or []
                b = base_tokens[tr["trace_id"]]
                if toks != b:
                    i = 0
                    while i < min(len(toks), len(b)) and toks[i] == b[i]:
                        i += 1
                    div_pos.append(i)
                flips.append(toks == b)
        finally:
            srv.stop()
        rate = sum(flips) / len(flips)
        results[name] = {
            "parity": round(rate, 4), "n_flips": len(flips) - sum(flips),
            "n": len(flips),
            "flip_positions": sorted(div_pos),
            "median_div_pos": statistics.median(div_pos) if div_pos else None,
        }
        print(f"{name}: parity={rate:.3f} flips={results[name]['n_flips']}/{len(flips)}",
              flush=True)

    out = OUTROOT / time.strftime("parity_probe_%Y%m%dT%H%M%S.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
