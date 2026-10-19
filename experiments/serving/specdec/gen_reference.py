#!/usr/bin/env python3
"""Generate the frozen reference outputs (baseline greedy, product cap 1024).

Single baseline server, one request per trace, incremental writes to
``reference_outputs.json`` (so partial progress is usable). References are
load-independent: greedy token ids do not change under host contention.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench_spec import DEFAULT_TRACES, HERE, Server, stream_completion, N_PREDICT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--out", type=Path, default=HERE / "reference_outputs.json")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    traces = [json.loads(line) for line in a.traces.read_text().splitlines() if line.strip()]
    refs: dict = {}
    if a.out.exists():
        refs = json.loads(a.out.read_text())
        print(f"resuming: {len(refs)} references already present", flush=True)

    srv = Server("refgen", 18701, [], Path("/tmp/specdec-refgen.log"), threads=a.threads)
    srv.start()
    t0 = time.time()
    try:
        for i, tr in enumerate(traces):
            if tr["trace_id"] in refs:
                continue
            res = stream_completion(srv.port, tr["prompt_ids"], N_PREDICT,
                                    cache_prompt=False)
            if res.error or not res.tokens:
                print(f"[{i}] {tr['trace_id']} ERROR {res.error}", flush=True)
                continue
            refs[tr["trace_id"]] = {
                "tokens": res.tokens, "n_predicted": res.n_predicted,
                "stop_type": res.stop_type,
            }
            a.out.write_text(json.dumps(refs))
            done = len(refs)
            print(f"[{done}/{len(traces)}] {tr['trace_id'][:12]} {tr['family'][:14]:14s}"
                  f" out={len(res.tokens):4d} stop={res.stop_type}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
    finally:
        srv.stop()
    print(f"wrote {len(refs)} references -> {a.out}")


if __name__ == "__main__":
    main()
