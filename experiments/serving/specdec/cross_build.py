#!/usr/bin/env python3
"""Cross-build paired bench: two llama.cpp builds x {no-spec, spec} in one session.

Four llama-server arms (old-base, old-spec, new-base, new-spec), one request
flow per trace and arm exactly as ``bench_spec.run_arm_trace`` (warm-up then
timed warm request; baseline arms also repeat the warm request as the
determinism control). Arm order rotates per trace so host-load drift hits
every arm. Reports, per build, the spec-vs-base parity and speedup, and
across builds the no-spec token identity (behaviour change vs pinned
receipts) and the no-spec / spec speed ratios.

Usage:
  python3 cross_build.py --old-bin <b10453 llama-server> --new-bin <master llama-server> \
      [--config configs/ngsimple-n2m4.json] [--n-traces 60] [--n-predict 256]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import bench_spec
from bench_spec import (
    DEFAULT_TRACES,
    Server,
    bootstrap_speedup_ratio,
    pooled_acceptance,
    run_arm_trace,
    summarize_arm,
    tokens_equal,
)

DEFAULT_OUTROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/llama-upgrade-20261019/runs"
)


def first_diff(a: list[int], b: list[int]) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def ref_match(tokens: list[int] | None, ref: list[int] | None, n_predict: int) -> bool:
    """Prefix-aware match against a reference generated at a larger cap."""
    if tokens is None or ref is None:
        return False
    if len(tokens) >= n_predict:  # capped here: compare the common prefix
        return ref[: len(tokens)] == tokens
    return tokens == ref


def parity(rows: list[dict], arm_a: str, arm_b: str) -> dict[str, Any]:
    ta = {r["trace_id"]: r.get("tokens") for r in rows if r["arm"] == arm_a and not r.get("error")}
    tb = {r["trace_id"]: r.get("tokens") for r in rows if r["arm"] == arm_b and not r.get("error")}
    common = sorted(set(ta) & set(tb))
    flips = [t for t in common if not tokens_equal(ta[t], tb[t])]
    return {
        "n": len(common),
        "identical": len(common) - len(flips),
        "parity": (len(common) - len(flips)) / len(common) if common else None,
        "flips": [{"trace_id": t, "diverge_pos": first_diff(ta[t] or [], tb[t] or [])} for t in flips],
    }


def speed(rows: list[dict], traces: list[dict], base: str, cand: str) -> dict[str, Any]:
    by = {(r["arm"], r["trace_id"]): r for r in rows}
    pairs = []
    dec_pairs = []
    for t in traces:
        b, c = by.get((base, t["trace_id"])), by.get((cand, t["trace_id"]))
        if not b or not c or b.get("error") or c.get("error"):
            continue
        pairs.append((b["wall_ms"], c["wall_ms"]))
        if b.get("predicted_ms") and c.get("predicted_ms"):
            dec_pairs.append((b["predicted_ms"], c["predicted_ms"]))
    bs, cs = summarize_arm(rows, base), summarize_arm(rows, cand)
    lo, hi = bootstrap_speedup_ratio(pairs)
    sb, sc = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    return {
        "base": base, "cand": cand, "n_pairs": len(pairs),
        "e2e_warm_ms": [bs["e2e_warm_ms"], cs["e2e_warm_ms"]],
        "speedup_warm": (bs["e2e_warm_ms"] / cs["e2e_warm_ms"])
        if bs["e2e_warm_ms"] and cs["e2e_warm_ms"] else None,
        "speedup_warm_ci95": [lo, hi],
        "speedup_warm_pooled": sb / sc if sc else None,
        "decode_ms": [bs["decode_ms"], cs["decode_ms"]],
        "speedup_decode": (bs["decode_ms"] / cs["decode_ms"])
        if bs["decode_ms"] and cs["decode_ms"] else None,
        "speedup_decode_ci95": list(bootstrap_speedup_ratio(dec_pairs)),
        "tok_s": [bs["tok_s"], cs["tok_s"]],
        "ttft_ms": [bs["ttft_ms"], cs["ttft_ms"]],
        "p95_warm_ms": [bs["p95_warm_ms"], cs["p95_warm_ms"]],
        "cold_e2e_ms": [bs["cold_e2e_ms"], cs["cold_e2e_ms"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-bin", type=Path, required=True)
    ap.add_argument("--new-bin", type=Path, required=True)
    ap.add_argument("--old-name", default="b10453")
    ap.add_argument("--new-name", default="master")
    ap.add_argument("--config", type=Path, default=bench_spec.HERE / "configs" / "ngsimple-n2m4.json")
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--n-traces", type=int, default=60)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port-base", type=int, default=18711)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--no-spec", action="store_true",
                    help="only the two no-spec arms (cross-build identity + speed)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--reference", type=Path, default=bench_spec.HERE / "reference_outputs.json")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUTROOT)
    a = ap.parse_args()

    if a.model is not None:
        bench_spec.DEFAULT_MODEL = a.model
    cfg = json.loads(a.config.read_text())
    spec_args = list(cfg.get("server_args", []))
    traces = [json.loads(line) for line in a.traces.read_text().splitlines() if line.strip()]
    if a.n_traces and a.n_traces < len(traces):
        step = len(traces) / a.n_traces
        traces = [traces[int(i * step)] for i in range(a.n_traces)]

    tag = "nospec" if a.no_spec else cfg.get("name", a.config.stem)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-xbuild-{tag}"
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    o, n = a.old_name, a.new_name
    arms = [
        Server(f"{o}-base", a.port_base, [], out_dir / f"server-{o}-base.log",
               threads=a.threads, server_bin=a.old_bin),
        Server(f"{o}-spec", a.port_base + 1, spec_args, out_dir / f"server-{o}-spec.log",
               threads=a.threads, server_bin=a.old_bin),
        Server(f"{n}-base", a.port_base + 2, [], out_dir / f"server-{n}-base.log",
               threads=a.threads, server_bin=a.new_bin),
        Server(f"{n}-spec", a.port_base + 3, spec_args, out_dir / f"server-{n}-spec.log",
               threads=a.threads, server_bin=a.new_bin),
    ]
    if a.no_spec:
        arms = [s for s in arms if s.name.endswith("-base")]
    load0 = open("/proc/loadavg").read().split()[0]
    rows: list[dict] = []
    loads: list[float] = []
    started: list[Server] = []
    try:
        for s in arms:
            s.start()
            started.append(s)
        with open(out_dir / "per_request.jsonl", "w") as fh:
            for i, tr in enumerate(traces):
                k = i % len(arms)
                order = arms[k:] + arms[:k]
                if (i // len(arms)) % 2:
                    order = order[::-1]
                for srv in order:
                    row = run_arm_trace(srv, tr, do_cold=a.cold, n_predict=a.n_predict,
                                        repeat=srv.name.endswith("-base"))
                    row["i"] = i
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                loads.append(float(open("/proc/loadavg").read().split()[0]))
                print(f"[{i + 1}/{len(traces)}] {tr['trace_id']} load={loads[-1]:.1f} " + " ".join(
                    f"{r['arm']}={r.get('wall_ms', 0):.0f}ms" for r in rows[-len(arms):]), flush=True)
    finally:
        for s in started:
            s.stop()

    ref = json.loads(a.reference.read_text()) if a.reference and a.reference.exists() else None
    summary: dict[str, Any] = {
        "run_id": run_id, "old_bin": str(a.old_bin), "new_bin": str(a.new_bin),
        "model": str(bench_spec.DEFAULT_MODEL), "spec_config": cfg,
        "n_traces": len(traces), "n_predict": a.n_predict, "threads": a.threads,
        "cold": a.cold, "load_start": load0,
        "load_min": min(loads) if loads else None, "load_max": max(loads) if loads else None,
        "load_mean": sum(loads) / len(loads) if loads else None,
        "arms": {s.name: summarize_arm(rows, s.name) for s in arms},
        "parity": {f"{n}-base_vs_{o}-base": parity(rows, f"{o}-base", f"{n}-base")},
        "speed": {"nospec_new_vs_old": speed(rows, traces, f"{o}-base", f"{n}-base")},
        "acceptance": {
            s.name: dict(zip(("tok_per_step", "accept_rate", "mean_draft_len"),
                             pooled_acceptance(rows, s.name), strict=True))
            for s in arms if s.name.endswith("-spec")
        },
    }
    if not a.no_spec:
        summary["parity"].update({
            f"{o}-spec_vs_{o}-base": parity(rows, f"{o}-base", f"{o}-spec"),
            f"{n}-spec_vs_{n}-base": parity(rows, f"{n}-base", f"{n}-spec"),
            f"{n}-spec_vs_{o}-spec": parity(rows, f"{o}-spec", f"{n}-spec"),
        })
        summary["speed"].update({
            f"{o}_spec_speedup": speed(rows, traces, f"{o}-base", f"{o}-spec"),
            f"{n}_spec_speedup": speed(rows, traces, f"{n}-base", f"{n}-spec"),
            "spec_new_vs_old": speed(rows, traces, f"{o}-spec", f"{n}-spec"),
            "champion_new_vs_shipped_old_base": speed(rows, traces, f"{o}-base", f"{n}-spec"),
        })
    for s in arms:
        if s.name.endswith("-base"):
            rep = [tokens_equal(r.get("tokens"), r.get("tokens_repeat"))
                   for r in rows if r["arm"] == s.name and r.get("tokens_repeat") is not None]
            summary[f"{s.name}_repeat_parity"] = sum(rep) / len(rep) if rep else None
        if ref is not None:
            m = [ref_match(r.get("tokens"), (ref.get(r["trace_id"]) or {}).get("tokens"), a.n_predict)
                 for r in rows if r["arm"] == s.name and not r.get("error")]
            summary[f"{s.name}_ref_parity"] = sum(m) / len(m) if m else None
        if a.cold:
            wc = [tokens_equal(r.get("tokens"), r.get("cold_tokens"))
                  for r in rows if r["arm"] == s.name and not r.get("error")]
            summary[f"{s.name}_warm_vs_cold_parity"] = sum(wc) / len(wc) if wc else None
    summary["n_err"] = sum(1 for r in rows if r.get("error"))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("arms",)}, indent=1))
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
