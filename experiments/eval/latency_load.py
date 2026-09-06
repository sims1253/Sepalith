#!/usr/bin/env python3
"""V1c — TTFT + concurrent-load bench (eval-strategy-v2 §3, the tail leg).

Instrument named by the design doc (`eval/latency_load.py`). What it measures
(the serving-axis columns the v2 battery was missing):

  Leg A (ttft):  serial streaming TTFT DISTRIBUTION per ctx class (2k/8k) on
                 the v7-class serving GGUF — send -> first CONTENT chunk
                 (includes prefill; what the editor feels under the 1500ms
                 debounce). Cold per request (cache_prompt false = fresh edit).
  Leg B (sweep): concurrent-load sweep on one --parallel-4 server: c active
                 streams in {1,2,4}, each replaying distinct frozen traces;
                 every request is SUPERSEDED 1500ms after its first token
                 (the extension aborts on cursor move) — client aborts the
                 socket, server stops decoding. Reports TTFT p50/p95 per load
                 level + abort-waste (tokens generated for superseded
                 requests) + starve counts (no token within the hard window).

Prompts: the S0 frozen trace set (/mnt/h/sepalith/datasets/spec_traces,
MANIFEST.md — real render_zeta2/PSM prompts; stop ">>>>>>> UPDATED",
max_tokens 48 cap, ctx >= 10240 honored per the MANIFEST rig notes; the v7
model is minicpm5-family so prompt_n is recounted server-side per request).

CPU-only (no CUDA context), tracked-PID servers (SpecServer from spec_bench),
battery lock respected. RUN IN A QUIET WINDOW (TTFT is latency-sensitive).

Usage (repo root, .venv):
  .venv/bin/python experiments/eval/latency_load.py --port 18431 \
      [--n-ttft 50] [--n-sweep 10] [--levels 1,2,4] [--classes 2k,8k]
Writes experiments/eval/results_v1c_ttft.jsonl + results_v1c_sweep.jsonl
(per-request raw rows) + prints the summary block (also JSON on stdout last).
"""
import argparse
import http.client
import json
import queue
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_bench  # noqa: E402 — SpecServer, load_traces, sample_traces, STOP

MODEL = HERE.parent / "models" / "sft_v7_minicpm5-Q8_0.gguf"  # v7-class serving GGUF
OUT_TTFT = HERE / "results_v1c_ttft.jsonl"
OUT_SWEEP = HERE / "results_v1c_sweep.jsonl"
N_PREDICT = 48            # ghost-text cap (S2's tg48 convention)
SUPERSEDE_MS = 1500.0     # design doc: the 1500ms debounce -> supersession window
HARD_S = 300.0            # starve threshold: no first token in this window
                          # (wide: under c=4 load a 9K-token prefill on 8 shared
                          # threads can take minutes — only TRUE starvation counts)
CTX_PER_SLOT = 12288      # MANIFEST: -c >= 10240 for the 8k class + margin


def stream_once(port, prompt, supersede_ms=None, hard_s=HARD_S,
                n_predict=N_PREDICT):
    """One streaming /completion. If supersede_ms is set: abort the socket
    supersede_ms after the FIRST content token (cursor-move supersession);
    else run to completion. Returns a per-request metrics dict."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=hard_s)
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.0, "stop": [spec_bench.STOP],
                       "stream": True, "cache_prompt": False}).encode()
    res = dict(ttft_ms=None, tokens=0, completed=False, aborted=False,
               starved=False, wall_ms=None, prompt_n=None, prompt_ms=None,
               predicted_ms=None, n_predicted=None, error=None)
    t0 = time.time()
    sup_dl = None          # supersession deadline (set at first token)
    resp = None
    sock = None
    try:
        conn.connect()     # hold a direct sock ref: conn.sock can be reset to
        sock = conn.sock   # None by getresponse() on EOF-terminated responses
        sock.settimeout(min(hard_s, 2.0))
        conn.request("POST", "/completion", body=body,
                     headers={"Content-Type": "application/json"})
        try:
            resp = conn.getresponse()
        except socket.timeout:
            # server never even sent headers within the hard window
            res["starved"] = res["ttft_ms"] is None
            res["aborted"] = res["ttft_ms"] is not None
            resp = None
        while resp is not None:
            now = time.time()
            # next deadline = supersession if armed, else the hard starve
            # window. The socket timeout is set to EXACTLY the remaining
            # time-to-deadline: a buffered reader is poisoned after any
            # mid-read timeout ("cannot read from timed out object"), so a
            # timeout may only ever fire when we intend to stop reading.
            # (Prefill silence is tens of seconds — a short fixed read
            # timeout would falsely abort cold 8k requests.)
            dl = sup_dl if sup_dl is not None else t0 + hard_s
            remain = dl - now
            if remain <= 0:
                if sup_dl is not None:
                    res["aborted"] = True
                else:
                    res["starved"] = res["ttft_ms"] is None
                    res["aborted"] = res["ttft_ms"] is not None
                break
            sock.settimeout(max(0.05, remain))
            try:
                raw = resp.readline()
            except socket.timeout:
                if sup_dl is not None:
                    res["aborted"] = True
                else:
                    res["starved"] = res["ttft_ms"] is None
                    res["aborted"] = res["ttft_ms"] is not None
                break
            if not raw:
                res["aborted"] = res["ttft_ms"] is not None
                res["starved"] = res["ttft_ms"] is None
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            content = chunk.get("content", "")
            if content:
                if res["ttft_ms"] is None:
                    res["ttft_ms"] = (time.time() - t0) * 1000.0
                    if supersede_ms is not None:
                        sup_dl = time.time() + supersede_ms / 1000.0
                res["tokens"] += 1
            if chunk.get("stop"):
                res["completed"] = True
                t = chunk.get("timings", {})
                res["prompt_n"] = t.get("prompt_n")
                res["prompt_ms"] = t.get("prompt_ms")
                res["predicted_ms"] = t.get("predicted_ms")
                res["n_predicted"] = chunk.get("tokens_predicted")
                break
    except Exception as e:  # noqa: BLE001 — record, never crash the leg
        res["error"] = repr(e)[:200]
    finally:
        res["wall_ms"] = round((time.time() - t0) * 1000.0, 1)
        try:
            if resp is not None:
                resp.close()  # closes fp (owns the fd when conn.sock was reset)
        except OSError:
            pass
        try:
            conn.close()   # client abort: server sees disconnect, stops decode
        except OSError:
            pass
    return res


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 1)


def summarize_ttft(rows):
    out = {}
    for cc in sorted({r["ctx_class"] for r in rows}):
        rs = [r for r in rows if r["ctx_class"] == cc and r.get("ttft_ms") is not None]
        xs = [r["ttft_ms"] for r in rs]
        out[cc] = dict(n=len(rs), n_err=sum(1 for r in rows
                                            if r["ctx_class"] == cc and r.get("error")),
                       ttft_p50=pct(xs, 50), ttft_p95=pct(xs, 95),
                       ttft_p99=pct(xs, 99), ttft_mean=round(sum(xs) / len(xs), 1)
                       if xs else None,
                       prompt_n_med=round(statistics.median(
                           [r["prompt_n"] for r in rs if r.get("prompt_n")]), 0)
                       if any(r.get("prompt_n") for r in rs) else None)
    return out


def summarize_sweep(rows):
    out = {}
    for key in sorted({(r["level"], r["ctx_class"]) for r in rows}):
        lvl, cc = key
        rs = [r for r in rows if r["level"] == lvl and r["ctx_class"] == cc]
        xs = [r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None]
        waste = [r["tokens"] for r in rs if r.get("aborted")]
        out[f"c{lvl}|{cc}"] = dict(
            n=len(rs), n_starved=sum(1 for r in rs if r.get("starved")),
            n_completed=sum(1 for r in rs if r.get("completed")),
            n_aborted=sum(1 for r in rs if r.get("aborted")),
            ttft_p50=pct(xs, 50), ttft_p95=pct(xs, 95), ttft_mean=round(
                sum(xs) / len(xs), 1) if xs else None,
            abort_waste_tokens_med=round(statistics.median(waste), 1) if waste else None,
            abort_waste_tokens_tot=sum(waste) if waste else 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-ttft", type=int, default=50, help="serial-leg requests per class")
    ap.add_argument("--n-sweep", type=int, default=10, help="requests per stream per level")
    ap.add_argument("--levels", default="1,2,4", help="active streams per level")
    ap.add_argument("--classes", default="2k,8k")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--server", type=Path, default=spec_bench.SERVER)
    ap.add_argument("--traces", type=Path, default=spec_bench.TRACES)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--foreground", action="store_true",
                    help="Keep servers inside the runner step process group")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    by_class = spec_bench.load_traces(a.traces)
    classes = [c for c in a.classes.split(",") if c in by_class]
    levels = [int(x) for x in a.levels.split(",")]
    max_lvl = max(levels)

    # ---- Leg A: serial TTFT distribution (--parallel 1) --------------------
    print(f"[legA] serial TTFT n={a.n_ttft}/class model={a.model}", flush=True)
    srvA = spec_bench.SpecServer(a.model, a.port, ["--parallel", "1"],
                                 threads=a.threads, ctx=CTX_PER_SLOT,
                                 log_path=a.out / "llama-server-v1c-legA.log",
                                 server=a.server, foreground=a.foreground)
    ttft_rows = []
    try:
        srvA.start()
        with open(a.out / "results_v1c_ttft.jsonl", "w") as fh:
            for cc in classes:
                sel = spec_bench.sample_traces(by_class[cc], a.n_ttft)
                for i, tr in enumerate(sel):
                    r = stream_once(a.port, tr["prompt"])
                    row = dict(leg="ttft", ctx_class=cc, i=i,
                               trace_id=tr["trace_id"], **r)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    ttft_rows.append(row)
                    print(json.dumps({k: row.get(k) for k in
                                      ("ctx_class", "i", "ttft_ms", "wall_ms",
                                       "tokens", "completed", "starved",
                                       "prompt_n", "error")}), flush=True)
    finally:
        srvA.stop()

    # ---- Leg B: concurrent-load sweep (--parallel max_lvl) -----------------
    print(f"[legB] sweep levels={levels} n={a.n_sweep}/stream "
          f"slots={max_lvl} supersede={SUPERSEDE_MS}ms", flush=True)
    srvB = spec_bench.SpecServer(a.model, a.port + 1,
                                 ["--parallel", str(max_lvl)],
                                 threads=a.threads,
                                 ctx=CTX_PER_SLOT * max_lvl,
                                 log_path=a.out / "llama-server-v1c-legB.log",
                                 server=a.server, foreground=a.foreground)
    sweep_rows = []
    try:
        srvB.start()
        with open(a.out / "results_v1c_sweep.jsonl", "w") as fh:
            for cc in classes:
                sel = spec_bench.sample_traces(by_class[cc], 16 * max_lvl)
                for lvl in levels:
                    q = queue.Queue()

                    def worker(sid):
                        for k in range(a.n_sweep):
                            tr = sel[(sid * a.n_sweep + k) % len(sel)]
                            r = stream_once(a.port + 1, tr["prompt"],
                                            supersede_ms=SUPERSEDE_MS)
                            q.put(dict(leg="sweep", ctx_class=cc, level=lvl,
                                       stream=sid, req=k,
                                       trace_id=tr["trace_id"], **r))

                    ths = [threading.Thread(target=worker, args=(s,))
                           for s in range(lvl)]
                    t0 = time.time()
                    for t in ths:
                        t.start()
                    for t in ths:
                        t.join()
                    rows = [q.get() for _ in range(lvl * a.n_sweep)]
                    wall = time.time() - t0
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    sweep_rows.extend(rows)
                    xs = [r["ttft_ms"] for r in rows if r.get("ttft_ms")]
                    print(json.dumps(dict(cc=cc, level=lvl, wall_s=round(wall, 1),
                                          ttft_p50=pct(xs, 50),
                                          starved=sum(1 for r in rows if r.get("starved")),
                                          aborted=sum(1 for r in rows if r.get("aborted")),
                                          completed=sum(1 for r in rows if r.get("completed")))),
                          flush=True)
    finally:
        srvB.stop()

    summary = dict(
        model=a.model, threads=a.threads, n_predict=N_PREDICT,
        supersede_ms=SUPERSEDE_MS, hard_s=HARD_S, ctx_per_slot=CTX_PER_SLOT,
        ttft=summarize_ttft(ttft_rows), sweep=summarize_sweep(sweep_rows),
        loadavg=open("/proc/loadavg").read().split()[0])
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
