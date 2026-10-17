#!/usr/bin/env python3
"""Paired A/B speculative-decoding bench for Sepalith PRM03 traces (CPU).

Runs a baseline llama-server (no speculation) and a candidate llama-server
(config-driven spec flags) **in the same session**, interleaving requests
per trace so host-load drift hits both arms equally. Per trace and arm:

  1. (optional, --cold) timed cold request (cache_prompt=false): full
     prefill + decode — the single-shot end-to-end number;
  2. warm-up request (n_predict=0): prefills the prompt into the slot
     cache, mirroring the extension's warm-up flow (PR #66);
  3. timed warm request (n_predict=1024, cache_prompt=true, stream):
     ~1 fresh token + decode — the product's post-pause request flow.

Greedy token ids are captured via return_tokens; the candidate must be
token-identical to the baseline for 100% of traces (lossless gate).

Emits ``METRIC name=value`` lines (autoresearch-compatible) and writes
per-request JSONL + summary.json into the run directory.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SERVER_BIN = Path(
    "/home/m0hawk/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-server"
)
DEFAULT_MODEL = Path(
    "/mnt/e/sepalith/campaign-20260915/quant-work/arm-a/arm-a-Q8_0.gguf"
)
DEFAULT_TRACES = HERE / "traces.jsonl"
DEFAULT_OUTROOT = Path(
    "/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/runs"
)
N_PREDICT = 1024  # brief: generation cap (product); sweeps may lower it
PORT_BASE = 18601


# ---------------------------------------------------------------------------
# client


def completion_body(ids: list[int], n_predict: int, *, cache_prompt: bool) -> bytes:
    return json.dumps(
        {
            "prompt": ids,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "mirostat": 0,
            "ignore_eos": False,
            "stop": [],
            "seed": 3407,
            "cache_prompt": cache_prompt,
            "return_tokens": True,
            "stream": True,
            "n_probs": 0,
            "samplers": ["top_k"],
        }
    ).encode()


class Res:
    __slots__ = (
        "ttft_ms", "wall_ms", "timings", "tokens", "n_predicted",
        "stop_type", "error",
    )

    def __init__(self) -> None:
        self.ttft_ms: float | None = None
        self.wall_ms = 0.0
        self.timings: dict[str, Any] = {}
        self.tokens: list[int] = []
        self.n_predicted: int | None = None
        self.stop_type: str | None = None
        self.error: str | None = None


def stream_completion(port: int, ids: list[int], n_predict: int,
                      *, cache_prompt: bool, timeout: float = 900.0) -> Res:
    res = Res()
    body = completion_body(ids, n_predict, cache_prompt=cache_prompt)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/completion", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"/completion {resp.status}: {resp.read()[:300]!r}")
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("content") and res.ttft_ms is None:
                res.ttft_ms = (time.monotonic() - t0) * 1000.0
            # stream mode: each partial chunk carries its incremental token ids
            toks = chunk.get("tokens")
            if isinstance(toks, list):
                res.tokens.extend(int(t) for t in toks)
            if chunk.get("stop"):
                res.timings = chunk.get("timings", res.timings)
                res.n_predicted = chunk.get("tokens_predicted", res.n_predicted)
                res.stop_type = chunk.get("stop_type")
                break
    except Exception as e:  # noqa: BLE001 — record, never crash the leg
        res.error = repr(e)[:300]
    finally:
        conn.close()
    res.wall_ms = (time.monotonic() - t0) * 1000.0
    return res


def warmup(port: int, ids: list[int], timeout: float = 900.0) -> None:
    body = json.dumps(
        {
            "prompt": ids,
            "n_predict": 0,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 3407,
            "cache_prompt": True,
            "stream": False,
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", "/completion", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"warmup {resp.status}: {resp.read()[:300]!r}")
        resp.read()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# server


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Server:
    def __init__(self, name: str, port: int, server_args: list[str],
                 log_path: Path, threads: int = 8, ctx: int = 16384):
        self.name, self.port = name, port
        self.server_args = list(server_args)
        self.log_path, self.threads, self.ctx = log_path, threads, ctx
        self.proc: subprocess.Popen | None = None

    def cmd(self) -> list[str]:
        return [
            "nice", "-n", "10", str(SERVER_BIN),
            "-m", str(DEFAULT_MODEL),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--alias", f"specdec-{self.name}",
            "--temp", "0", "--parallel", "1", "--metrics",
            "-c", str(self.ctx), "-t", str(self.threads), "-ngl", "0",
        ] + self.server_args

    def start(self, ready_timeout: float = 420.0) -> None:
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} in use — refusing to touch it")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        with open(self.log_path, "ab") as log:
            log.write(f"\n==== {time.strftime('%F %T')} {' '.join(self.cmd())}\n".encode())
            log.flush()
            self.proc = subprocess.Popen(
                self.cmd(), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        t0 = time.monotonic()
        while time.monotonic() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"[{self.name}] llama-server rc={self.proc.returncode}; "
                    f"log tail: {self.log_tail()}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                conn.request("GET", "/health")
                r = conn.getresponse()
                r.read()
                conn.close()
                if r.status == 200:
                    return
            except OSError:
                pass
            time.sleep(2.0)
        raise RuntimeError(f"[{self.name}] not ready; log tail: {self.log_tail()}")

    def log_tail(self, n: int = 2500) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return "<no log>"

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.kill(self.proc.pid, signal.SIGTERM)
            for _ in range(100):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(self.proc.pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass
        finally:
            self.proc = None


# ---------------------------------------------------------------------------
# stats helpers


def med(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def bootstrap_speedup_ratio(pairs: list[tuple[float, float]], reps: int = 1000,
                            seed: int = 7) -> tuple[float, float]:
    """95% CI of median(base)/median(cand) over trace-level bootstrap."""
    if len(pairs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ratios = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        b = statistics.median([p[0] for p in sample])
        c = statistics.median([p[1] for p in sample])
        if c > 0:
            ratios.append(b / c)
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios)) - 1]
    return (lo, hi)


def tokens_equal(a: list[int] | None, b: list[int] | None) -> bool:
    return a is not None and b is not None and list(a) == list(b)


# ---------------------------------------------------------------------------
# bench


def run_arm_trace(server: Server, tr: dict, do_cold: bool, n_predict: int) -> dict:
    row: dict[str, Any] = {
        "arm": server.name, "trace_id": tr["trace_id"],
        "family": tr["family"], "band": tr["band"], "is_noop": tr["is_noop"],
        "prompt_tokens": tr["prompt_token_count"],
    }
    if do_cold:
        cold = stream_completion(server.port, tr["prompt_ids"], n_predict,
                                 cache_prompt=False)
        row.update(
            cold_wall_ms=cold.wall_ms, cold_ttft_ms=cold.ttft_ms,
            cold_prompt_ms=(cold.timings or {}).get("prompt_ms"),
            cold_tokens=cold.tokens, cold_stop=cold.stop_type,
            cold_error=cold.error,
        )
        if cold.error:
            row["error"] = cold.error
    warmup(server.port, tr["prompt_ids"])
    w = stream_completion(server.port, tr["prompt_ids"], n_predict,
                          cache_prompt=True)
    t = w.timings or {}
    row.update(
        wall_ms=w.wall_ms, ttft_ms=w.ttft_ms,
        prompt_ms=t.get("prompt_ms"), prompt_n=t.get("prompt_n"),
        cache_n=t.get("cache_n"),
        predicted_ms=t.get("predicted_ms"), predicted_n=t.get("predicted_n"),
        draft_n=t.get("draft_n"), draft_n_accepted=t.get("draft_n_accepted"),
        tokens=w.tokens, stop_type=w.stop_type, error=w.error,
    )
    if w.error and not row.get("error"):
        row["error"] = w.error
    return row


def summarize_arm(rows: list[dict], arm: str) -> dict:
    rs = [r for r in rows if r["arm"] == arm]
    ok = [r for r in rs if not r.get("error")]
    walls = [r["wall_ms"] for r in ok]
    return {
        "n": len(rs), "n_err": len(rs) - len(ok),
        "e2e_warm_ms": med(walls), "p95_warm_ms": p95(walls),
        "ttft_ms": med([r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]),
        "decode_ms": med([r["predicted_ms"] for r in ok if r.get("predicted_ms")]),
        "tok_s": med([
            r["predicted_n"] / (r["predicted_ms"] / 1000.0)
            for r in ok if r.get("predicted_ms") and r.get("predicted_n")
        ]),
        "cold_e2e_ms": med([r["cold_wall_ms"] for r in ok if r.get("cold_wall_ms") is not None]),
        "cold_prompt_ms": med([r["cold_prompt_ms"] for r in ok if r.get("cold_prompt_ms") is not None]),
        "cache_frac_med": med([
            r["cache_n"] / r["prompt_tokens"] if r.get("cache_n") is not None else None
            for r in ok
        ] if any(r.get("cache_n") is not None for r in ok) else []),
    }


def pooled_acceptance(rows: list[dict], arm: str) -> tuple[float | None, float | None, float | None]:
    """(tok_per_step, accept_rate, mean_draft_len) pooled over drafting rows."""
    drafted = sum(r["draft_n"] or 0 for r in rows if r["arm"] == arm)
    accepted = sum(r["draft_n_accepted"] or 0 for r in rows if r["arm"] == arm)
    predicted = sum(r["predicted_n"] or 0 for r in rows
                    if r["arm"] == arm and r.get("draft_n"))
    steps = predicted - accepted
    if drafted <= 0 or steps <= 0:
        return None, None, None
    return predicted / steps, accepted / drafted, drafted / steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="candidate arm JSON {name, server_args[]}")
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUTROOT)
    ap.add_argument("--n-traces", type=int, default=0,
                    help="deterministic subset (family-spread) for quick runs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port-base", type=int, default=PORT_BASE)
    ap.add_argument("--cold", action="store_true",
                    help="also time a cold (cache_prompt=false) pass per trace")
    ap.add_argument("--n-predict", type=int, default=N_PREDICT,
                    help="generation cap (product=1024; sweeps may lower)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="reference_outputs.json for cross-session parity")
    ap.add_argument("--save-reference", action="store_true",
                    help="write this session's baseline outputs as reference")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    cand_args: list[str] = list(cfg.get("server_args", []))
    cand_name = str(cfg.get("name", a.config.stem))

    traces = [json.loads(l) for l in a.traces.read_text().splitlines() if l.strip()]
    if a.n_traces and a.n_traces < len(traces):
        step = len(traces) / a.n_traces
        traces = [traces[int(i * step)] for i in range(a.n_traces)]

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{cand_name}"
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    load0 = open("/proc/loadavg").read().split()
    n_predict_global = [a.n_predict]
    base = Server("baseline", a.port_base, [], out_dir / "server-baseline.log",
                  threads=a.threads)
    cand = Server(cand_name, a.port_base + 1, cand_args,
                  out_dir / f"server-{cand_name}.log", threads=a.threads)
    rows: list[dict] = []
    out_fh = open(out_dir / "per_request.jsonl", "w")
    try:
        base.start()
        cand.start()
        for i, tr in enumerate(traces):
            arms = [base, cand] if i % 2 == 0 else [cand, base]
            for srv in arms:
                row = run_arm_trace(srv, tr, do_cold=a.cold, n_predict=a.n_predict)
                row["i"] = i
                rows.append(row)
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                err = " ERR" if row.get("error") else ""
                print(
                    f"[{i + 1}/{len(traces)}] {srv.name} {tr['trace_id']} "
                    f"wall={row.get('wall_ms', 0):.0f}ms "
                    f"dec={row.get('predicted_ms') or 0:.0f}ms "
                    f"draft={row.get('draft_n')} acc={row.get('draft_n_accepted')}"
                    f"{err}", flush=True)
    finally:
        cand.stop()
        base.stop()
        out_fh.close()
    load1 = open("/proc/loadavg").read().split()

    with open(out_dir / "summary_meta.json", "w") as fh:
        fh.write(json.dumps({"n_rows": len(rows)}) + "\n")

    # parity: candidate vs baseline, same trace, same session
    base_tok = {r["trace_id"]: r.get("tokens") for r in rows if r["arm"] == "baseline"}
    cand_rows = [r for r in rows if r["arm"] == cand_name]
    par = [
        tokens_equal(r.get("tokens"), base_tok.get(r["trace_id"]))
        for r in cand_rows if not r.get("error")
    ]
    parity = sum(par) / len(par) if par else 0.0
    # baseline internal consistency: warm vs cold (batch-shape drift floor)
    if a.cold:
        bpar = [
            tokens_equal(r.get("tokens"), r.get("cold_tokens"))
            for r in rows if r["arm"] == "baseline" and not r.get("error")
        ]
        parity_base_warmcold = sum(bpar) / len(bpar) if bpar else None
    else:
        parity_base_warmcold = None

    ref = None
    if a.reference and a.reference.exists():
        ref = json.loads(a.reference.read_text())
        rpar = [
            tokens_equal(r.get("tokens"), ref.get(r["trace_id"], {}).get("tokens"))
            for r in cand_rows if not r.get("error")
        ]
        parity_ref = sum(rpar) / len(rpar) if rpar else None
    else:
        parity_ref = None

    if a.save_reference:
        ref_out = {
            r["trace_id"]: {"tokens": r.get("tokens"),
                            "n_predicted": r.get("predicted_n")}
            for r in rows if r["arm"] == "baseline"
        }
        (HERE / "reference_outputs.json").write_text(json.dumps(ref_out))

    bs, cs = summarize_arm(rows, "baseline"), summarize_arm(rows, cand_name)
    pairs = [
        (b["wall_ms"], c["wall_ms"])
        for b, c in (
            (next(r for r in rows if r["arm"] == "baseline" and r["trace_id"] == t["trace_id"]),
             next(r for r in rows if r["arm"] == cand_name and r["trace_id"] == t["trace_id"]))
            for t in traces
        )
        if b.get("wall_ms") and c.get("wall_ms")
        and not b.get("error") and not c.get("error")
    ]
    speedup = (bs["e2e_warm_ms"] / cs["e2e_warm_ms"]) if (
        bs["e2e_warm_ms"] and cs["e2e_warm_ms"]) else float("nan")
    ci_lo, ci_hi = bootstrap_speedup_ratio(pairs)
    tok_step, acc_rate, draft_len = pooled_acceptance(rows, cand_name)
    sum_b = sum(p[0] for p in pairs)
    sum_c = sum(p[1] for p in pairs)
    speedup_pooled = (sum_b / sum_c) if sum_c > 0 else float("nan")

    summary = {
        "run_id": run_id, "candidate": cand_name,
        "server_args": cand_args, "n_traces": len(traces),
        "n_predict": n_predict_global[0],
        "threads": a.threads, "cold": a.cold,
        "model": str(DEFAULT_MODEL),
        "load_start": load0[0], "load_end": load1[0],
        "baseline": bs, "candidate": cs,
        "speedup_warm": speedup,
        "speedup_warm_pooled": speedup_pooled,
        "speedup_warm_ci95": [ci_lo, ci_hi],
        "parity": parity, "parity_ref": parity_ref,
        "parity_base_warmcold": parity_base_warmcold,
        "accept_tok_per_step": tok_step, "accept_rate": acc_rate,
        "mean_draft_len": draft_len,
        "n_parity_fail": len(par) - sum(par),
    }
    # per-band / edit-vs-noop breakdown
    for dim, keyf in (("band", lambda r: r["band"]),
                      ("edittype", lambda r: "noop" if r["is_noop"] else "edit"),
                      ("family", lambda r: r["family"])):
        for val in sorted({keyf(r) for r in rows}):
            sel = [r for r in rows if keyf(r) == val]
            summary[f"{dim}:{val}"] = {
                "baseline": summarize_arm(sel, "baseline"),
                "candidate": summarize_arm(sel, cand_name),
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    def fmt(x: float | None, nd: int = 1) -> str:
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "nan"

    print("\nMETRIC speedup_warm=" + fmt(speedup, 3))
    print("METRIC speedup_warm_pooled=" + fmt(speedup_pooled, 3))
    print("METRIC speedup_warm_ci_lo=" + fmt(ci_lo, 3))
    print("METRIC e2e_warm_ms=" + fmt(cs["e2e_warm_ms"]))
    print("METRIC e2e_warm_base_ms=" + fmt(bs["e2e_warm_ms"]))
    print("METRIC decode_ms=" + fmt(cs["decode_ms"]))
    print("METRIC decode_ms_base=" + fmt(bs["decode_ms"]))
    print("METRIC tok_s=" + fmt(cs["tok_s"], 2))
    print("METRIC tok_s_base=" + fmt(bs["tok_s"], 2))
    print("METRIC ttft_ms=" + fmt(cs["ttft_ms"]))
    print("METRIC ttft_ms_base=" + fmt(bs["ttft_ms"]))
    print("METRIC p95_warm_ms=" + fmt(cs["p95_warm_ms"]))
    print("METRIC p95_warm_ms_base=" + fmt(bs["p95_warm_ms"]))
    print("METRIC accept_tok_per_step=" + fmt(tok_step, 3))
    print("METRIC accept_rate=" + fmt(acc_rate, 3))
    print("METRIC mean_draft_len=" + fmt(draft_len, 2))
    print("METRIC parity=" + fmt(parity, 4))
    if parity_ref is not None:
        print("METRIC parity_ref=" + fmt(parity_ref, 4))
    if parity_base_warmcold is not None:
        print("METRIC parity_base_warmcold=" + fmt(parity_base_warmcold, 4))
    if a.cold:
        print("METRIC cold_e2e_ms=" + fmt(cs["cold_e2e_ms"]))
        print("METRIC cold_e2e_ms_base=" + fmt(bs["cold_e2e_ms"]))
    print("METRIC n_err=" + str(bs["n_err"] + cs["n_err"]))
    print("METRIC load1=" + load1[0])
    print(f"summary -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
