#!/usr/bin/env python3
"""KT (A2 §4.4 / RT1-A9): 4-arm prefix-cache bench on vanilla llama.cpp, CPU.

Grounds the A2 serving claims + the PSM anchor design empirically: what does
the product's edit-event sequence actually cost in prefill on stock
llama.cpp's byte-prefix cache, per event class, in the extension's real
PSM render (suffix-first, run_intent_suite.render_prompt parity) vs a
prefix-first ordering of the identical session state?

Arms (n trials each; each trial = establish-request R0 then measured R1,
same slot/server, --cache-reuse 256 per the design's serving stance):
  a  keystroke/suffix-append (PSM): partial line grows at the cursor ->
     prompt changes only near its END.
  b  anchor-move small (PSM): cursor + ~512 tok (~55 lines); suffix block
     (prompt HEAD) changes wholesale, prefix tail grows.
  c  anchor-move large (PSM): cursor + ~4K tok (~430 lines).
  b2/c2 = same anchors as b/c, rendered PREFIX-FIRST (suffix last) — the
     cache-friendly ordering companion, isolated as its own arm (own R0).
  d  fresh context: unseen file, cold R0 measured directly.

Metrics per measured request (from llama-server response timings):
  prompt_n (total prompt tokens), n_prompt_tokens_processed (computed),
  cached = prompt_n - processed, cached_frac, prompt_ms, prefill tok/s.

Model: qwen3.5-2b-Q4_K_M.gguf (the 22.1 t/s anchor box's own class).
Corpus: real R files staged to /tmp (one drvfs read burst, then local IO).
Server: own child, readiness = POST /v1/completions 200, tracked-PID kill.

Usage:  nohup nice -n 5 .venv/bin/python3 -u cache_bench.py \
            > /tmp/a2gates/gate4_cachebench.log 2>&1 &
Writes: experiments/eval/results_cachebench_qwen35-2b.jsonl (+ summary row).
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models" / "qwen3.5-2b-Q4_K_M.gguf"
SERVER = HERE.parent / "bin" / "llama" / "llama-b10453" / "llama-server"
CORPUS_SRC = Path("/mnt/h/sepalith/normalized")
STAGE = Path("/tmp/a2gates/bench_corpus")
OUT = HERE / "results_cachebench_qwen35-2b.jsonl"

TOK_PER_LINE = 9.3          # measured house mean
N_TRIALS = 30
PREFIX_LINES = 260          # ~2.4K tok prefix tail
SUFFIX_LINES = 130          # ~1.2K tok suffix head
SMALL_MOVE_LINES = 55       # ~512 tok
LARGE_MOVE_LINES = 430      # ~4K tok
GEN = 16


def stage_corpus():
    """Copy real R files: 60 mid-size (>=150 lines) + 40 large (>=800)."""
    if STAGE.exists() and len(list(STAGE.glob("*.R"))) >= 80:
        out = []
        for p in sorted(STAGE.glob("*.R")):
            out.append((p, p.read_text(errors="replace").splitlines()))
        return out
    STAGE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    pkgs = rng.sample(sorted(p.name for p in CORPUS_SRC.iterdir() if p.is_dir()), 25)
    mids, bigs = [], []
    for pkg in pkgs:
        for f in (CORPUS_SRC / pkg).rglob("*.R"):
            try:
                if f.stat().st_size < 2_000:
                    continue
                lines = f.read_text(errors="replace").splitlines()
            except OSError:
                continue
            if len(lines) >= 800 and len(bigs) < 40:
                bigs.append((f, lines))
            elif 150 <= len(lines) < 800 and len(mids) < 60:
                mids.append((f, lines))
            if len(mids) >= 60 and len(bigs) >= 40:
                break
        if len(mids) >= 60 and len(bigs) >= 40:
            break
    out = []
    for i, (f, lines) in enumerate(mids + bigs):
        dst = STAGE / f"bench_{i:03d}.R"
        dst.write_text("\n".join(lines) + "\n")
        out.append((dst, lines))
    print(json.dumps(dict(staged_files=len(out),
                          big=len(bigs), mid=len(mids))), flush=True)
    return out


def render_psm(fname, lines, anchor, partial=""):
    """Extension parity (run_intent_suite.render_prompt): SUFFIX FIRST."""
    suffix = [l for l in lines[anchor:anchor + SUFFIX_LINES]]
    prefix = [l for l in lines[max(0, anchor - PREFIX_LINES):anchor]]
    parts = ["<[fim-suffix]>"] + suffix
    parts += [f"<[fim-prefix]><filename>{fname}"] + prefix
    region = [partial + "<|user_cursor|>"] if partial else ["<|user_cursor|>"]
    parts += ["<<<<<<< CURRENT"] + region + ["=======", "<[fim-middle]>"]
    return "\n".join(parts)


def render_prefix_first(fname, lines, anchor, partial=""):
    """Same session state, suffix LAST (cache-friendly ordering)."""
    suffix = [l for l in lines[anchor:anchor + SUFFIX_LINES]]
    prefix = [l for l in lines[max(0, anchor - PREFIX_LINES):anchor]]
    parts = [f"<[fim-prefix]><filename>{fname}"] + prefix
    region = [partial + "<|user_cursor|>"] if partial else ["<|user_cursor|>"]
    parts += ["<<<<<<< CURRENT"] + region + ["=======", "<[fim-suffix]>"] + suffix
    parts += ["<[fim-middle]>"]
    return "\n".join(parts)


def complete(port, prompt, gen=GEN):
    body = json.dumps({"prompt": prompt, "max_tokens": gen, "temperature": 0,
                       "cache_prompt": True, "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        data = json.loads(r.read())
    t = data.get("timings", {})
    # b10453 semantics (verified): timings.prompt_n == n_prompt_tokens_processed
    # == tokens COMPUTED this call; the TOTAL prompt size is usage.prompt_tokens.
    processed = t.get("n_prompt_tokens_processed", t.get("prompt_n"))
    total = data.get("usage", {}).get("prompt_tokens", processed)
    return dict(prompt_n=total, processed=processed,
                cached=(total - processed) if (total is not None
                                               and processed is not None) else None,
                prompt_ms=round(t.get("prompt_ms", (time.time() - t0) * 1000), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=16384)
    args = ap.parse_args()

    files = stage_corpus()
    bigs = [(p, l) for p, l in files if len(l) >= 800]
    mids = [(p, l) for p, l in files if 150 <= len(l) < 800]
    if not mids:
        sys.exit(f"staging found no mid-size files ({len(files)} total)")
    if not bigs:
        sys.exit(f"staging found no >=800-line files for the large-move arm "
                 f"({len(files)} total)")
    rng = random.Random(5)

    log = open("/tmp/a2gates/llama-server-cachebench.log", "ab")
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--port", str(args.port),
         "--host", "127.0.0.1", "-t", str(args.threads), "-ngl", "0",
         "--parallel", "1", "-c", str(args.ctx), "--cache-reuse", "256"],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True)
    print(f"llama-server pid {proc.pid} port {args.port}", flush=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                sys.exit("server exited during startup")
            try:
                body = json.dumps({"prompt": "ready", "max_tokens": 1,
                                   "temperature": 0}).encode()
                rq = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=60) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            sys.exit("server never ready")
        print("server ready", flush=True)

        # one-time raw response dump: verifies the field semantics this
        # script's cached/processed math rests on (best-effort: never fatal)
        try:
            body = json.dumps({"prompt": "x <- 1 + 2\n", "max_tokens": 1,
                               "temperature": 0, "cache_prompt": True}).encode()
            rq = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/v1/completions", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=300) as r:
                probe = json.loads(r.read())
            print("RAW-PROBE timings=" + json.dumps(probe.get("timings")) +
                  " usage=" + json.dumps(probe.get("usage")), flush=True)
        except Exception as e:
            print(f"RAW-PROBE failed ({e}); continuing", flush=True)

        out = open(OUT, "w")

        def trial(arm, render, lines_path, lines, a0, a1, p0, p1):
            f = lines_path.name
            r0 = complete(args.port, render(f, lines, a0, p0))
            r1 = complete(args.port, render(f, lines, a1, p1))
            rec = dict(arm=arm, file=f, anchor0=a0, anchor1=a1,
                       lines=len(lines), **{f"r1_{k}": v for k, v in r1.items()},
                       r0_prompt_n=r0["prompt_n"], r0_prompt_ms=r0["prompt_ms"])
            rec["cached_frac"] = (round(rec["r1_cached"] / rec["r1_prompt_n"], 4)
                                  if rec.get("r1_cached") is not None
                                  and rec["r1_prompt_n"] else None)
            rec["r1_prefill_tps"] = (round(rec["r1_prompt_n"] /
                                           (rec["r1_prompt_ms"] / 1000), 1)
                                     if rec["r1_prompt_ms"] else None)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps({k: rec[k] for k in
                              ("arm", "file", "r1_prompt_n", "r1_cached",
                               "cached_frac", "r1_prompt_ms")}), flush=True)
            return rec

        recs = []
        # arm a: keystroke suffix-append (PSM render, mid files)
        pool = [mids[i % len(mids)] for i in range(N_TRIALS)]
        for i, (p, lines) in enumerate(pool):
            anchor = max(PREFIX_LINES, len(lines) // 2)
            recs.append(trial("a_suffix_append_psm", render_psm, p, lines,
                              anchor, anchor, "", "x <- mean(v"))
        # arm b: anchor-move small (PSM)
        for i, (p, lines) in enumerate(pool):
            anchor = max(PREFIX_LINES, len(lines) // 3)
            recs.append(trial("b_anchor_small_psm", render_psm, p, lines,
                              anchor, anchor + SMALL_MOVE_LINES, "", ""))
        # arm c: anchor-move large (PSM; big files)
        poolc = [bigs[i % len(bigs)] for i in range(N_TRIALS)]
        for i, (p, lines) in enumerate(poolc):
            anchor = max(PREFIX_LINES, len(lines) // 4)
            recs.append(trial("c_anchor_large_psm", render_psm, p, lines,
                              anchor, anchor + LARGE_MOVE_LINES, "", ""))
        # arm b2/c2: prefix-first companions
        for i, (p, lines) in enumerate(pool):
            anchor = max(PREFIX_LINES, len(lines) // 3)
            recs.append(trial("b2_anchor_small_prefixfirst", render_prefix_first,
                              p, lines, anchor, anchor + SMALL_MOVE_LINES, "", ""))
        for i, (p, lines) in enumerate(poolc):
            anchor = max(PREFIX_LINES, len(lines) // 4)
            recs.append(trial("c2_anchor_large_prefixfirst", render_prefix_first,
                              p, lines, anchor, anchor + LARGE_MOVE_LINES, "", ""))
        # arm d: fresh context (cold R0 on rotating unseen files)
        for i, (p, lines) in enumerate(pool + poolc[:6]):
            f = p.name
            anchor = max(PREFIX_LINES, len(lines) // 2)
            # rotate to a different prompt each trial by shifting suffix window
            r = complete(args.port, render_psm(f, lines, anchor + 7 * i, ""))
            rec = dict(arm="d_fresh_cold", file=f, anchor0=anchor + 7 * i,
                       anchor1=None, lines=len(lines),
                       **{f"r1_{k}": v for k, v in r.items()},
                       r0_prompt_n=None, r0_prompt_ms=None)
            rec["cached_frac"] = (round(rec["r1_cached"] / rec["r1_prompt_n"], 4)
                                  if rec.get("r1_cached") is not None
                                  and rec["r1_prompt_n"] else None)
            rec["r1_prefill_tps"] = (round(rec["r1_prompt_n"] /
                                           (rec["r1_prompt_ms"] / 1000), 1)
                                     if rec["r1_prompt_ms"] else None)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps({k: rec[k] for k in
                              ("arm", "file", "r1_prompt_n", "r1_cached",
                               "cached_frac", "r1_prompt_ms")}), flush=True)
            recs.append(rec)

        # summary
        summary = dict(gate="cache_bench", model="qwen3.5-2b-Q4_K_M",
                       ctx=args.ctx, threads=args.threads,
                       cache_reuse=256, n_trials=N_TRIALS, arms={})
        for arm in sorted({r["arm"] for r in recs}):
            rs = [r for r in recs if r["arm"] == arm]
            def med(k):
                xs = [r[k] for r in rs if r.get(k) is not None]
                return round(statistics.median(xs), 2) if xs else None
            summary["arms"][arm] = dict(
                n=len(rs), prompt_n_med=med("r1_prompt_n"),
                cached_med=med("r1_cached"), cached_frac_med=med("cached_frac"),
                prompt_ms_med=med("r1_prompt_ms"),
                prompt_ms_p95=round(sorted(r["r1_prompt_ms"] for r in rs
                                           if r["r1_prompt_ms"])[
                                            max(0, int(0.95 * len(rs)) - 1)], 1),
                prefill_tps_med=med("r1_prefill_tps"))
        out.write(json.dumps({"summary": summary}) + "\n")
        out.close()
        print(json.dumps(summary, indent=1), flush=True)
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGTERM)
            for _ in range(50):
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
        print(f"server pid {proc.pid} stopped", flush=True)


if __name__ == "__main__":
    main()
