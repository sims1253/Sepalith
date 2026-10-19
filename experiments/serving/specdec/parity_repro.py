#!/usr/bin/env python3
"""Minimal reproduction: speculative verification flips a near-tie argmax.

Baseline (no spec) vs ngram-simple n=1 m=1 (single drafted token per step,
verify batch of 2) on one trace; captures the top-2 token probabilities at
the divergence position from the baseline arm to show the near-tie.

Usage: python3 parity_repro.py [trace_index] [--server-bin BIN] [--out-root DIR]
       [--port-base PORT]   (default: scan the 60-trace subset for flip cases)
"""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path

from bench_spec import DEFAULT_TRACES, Server, stream_completion

OUTROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/runs"
)


def probs_at(port: int, ids: list[int], n_predict: int, pos: int) -> list[dict]:
    """Run with n_probs and return the top-2 probs at output position pos."""
    body = json.dumps({
        "prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": 3407, "cache_prompt": False, "stream": False, "n_probs": 5,
        "samplers": ["top_k"], "stop": [],
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/completion", body=body,
                 headers={"Content-Type": "application/json"})
    d = json.loads(conn.getresponse().read())
    conn.close()
    probs = d.get("completion_probabilities") or []
    if 0 <= pos < len(probs):
        top = probs[pos].get("top_logprobs") or []
        return [
            {"id": p.get("id"), "token": p.get("token"),
             "logprob": p.get("logprob")}
            for p in top[:3]
        ]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_index", nargs="?", type=int, default=None,
                    help="index into the 60-trace subset (default: scan all)")
    ap.add_argument("--server-bin", type=Path, default=None,
                    help="llama-server binary (default: pinned b10453)")
    ap.add_argument("--out-root", type=Path, default=OUTROOT)
    ap.add_argument("--port-base", type=int, default=18621)
    ap.add_argument("--max-cases", type=int, default=3)
    a = ap.parse_args()
    traces = [json.loads(line) for line in DEFAULT_TRACES.read_text().splitlines() if line.strip()]
    # probe flip positions: subset trace index -> subset list index; use a
    # trace known to flip: find by running both arms if index not given
    idx = a.trace_index
    step = len(traces) / 60
    subset = [traces[int(i * step)] for i in range(60)]

    spec_args = ["--spec-type", "ngram-simple",
                 "--spec-ngram-simple-size-n", "1",
                 "--spec-ngram-simple-size-m", "1"]
    a.out_root.mkdir(parents=True, exist_ok=True)
    base = Server("rbase", a.port_base, [], a.out_root / "repro-base.log",
                  server_bin=a.server_bin)
    spec = Server("rspec", a.port_base + 1, spec_args, a.out_root / "repro-spec.log",
                  server_bin=a.server_bin)
    base.start()
    spec.start()
    report = {"config": "ngram-simple n=1 m=1 (verify batch of 2)",
              "server_bin": str(base.server_bin), "n_scanned": 0, "cases": []}
    try:
        candidates = [subset[idx]] if idx is not None else subset
        for tr in candidates:
            rb = stream_completion(base.port, tr["prompt_ids"], 128, cache_prompt=False)
            rs = stream_completion(spec.port, tr["prompt_ids"], 128, cache_prompt=False)
            report["n_scanned"] += 1
            b, s = rb.tokens or [], rs.tokens or []
            if b == s:
                continue
            i = 0
            while i < min(len(b), len(s)) and b[i] == s[i]:
                i += 1
            top2 = probs_at(base.port, tr["prompt_ids"], 128, i)
            report["cases"].append({
                "trace_id": tr["trace_id"], "family": tr["family"],
                "diverge_pos": i,
                "baseline_token": b[i] if i < len(b) else None,
                "spec_token": s[i] if i < len(s) else None,
                "baseline_top3_logprobs_at_prev_pos": top2,
                "n_base": len(b), "n_spec": len(s),
            })
            print(json.dumps(report["cases"][-1], indent=1), flush=True)
            if len(report["cases"]) >= a.max_cases:
                break
    finally:
        spec.stop()
        base.stop()
    out = a.out_root / "parity_repro.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
