#!/usr/bin/env python3
"""Gate 4 supplement: PINNED-HEAD window renders.

The main bench (cache_bench.py) measured that ANY anchor move kills the
byte-prefix cache in both PSM suffix-first and prefix-first orderings —
because the bounded-context prefix window SLIDES (its head changes).
This supplement measures the obvious rescue: a PINNED head (prefix starts
at the file top and only GROWS; the old head is never cut). Anchor moves
then only append before the region -> large common prefix expected.

Arms (n=10 each, prefix-first pinned renders):
  s1 small anchor move (+55 lines) pinned head
  s2 large anchor move (+430 lines) pinned head
Writes experiments/eval/results_cachebench_pinned_noreuse.jsonl.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models" / "qwen3.5-2b-Q4_K_M.gguf"
SERVER = HERE.parent / "bin" / "llama" / "llama-b10453" / "llama-server"
STAGE = Path("/tmp/a2gates/bench_corpus")
OUT = HERE / "results_cachebench_pinned_noreuse.jsonl"
SUFFIX_LINES = 130
GEN = 16
PORT = 18099


def render_pinned(fname, lines, anchor):
    """Prefix-first, prefix pinned to file line 0 (grows monotonically)."""
    suffix = lines[anchor:anchor + SUFFIX_LINES]
    parts = [f"<[fim-prefix]><filename>{fname}"] + lines[:anchor]
    parts += ["<<<<<<< CURRENT", "<|user_cursor|>", "=======",
              "<[fim-suffix]>"] + suffix + ["<[fim-middle]>"]
    return "\n".join(parts)


def complete(prompt):
    body = json.dumps({"prompt": prompt, "max_tokens": GEN,
                       "temperature": 0, "cache_prompt": True,
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    t = d.get("timings", {})
    proc = t.get("n_prompt_tokens_processed", t.get("prompt_n"))
    total = d.get("usage", {}).get("prompt_tokens", proc)
    return dict(prompt_n=total, processed=proc, cached=total - proc,
                prompt_ms=round(t.get("prompt_ms", -1), 1))


def main():
    files = sorted(STAGE.glob("*.R"))
    bigs = [p for p in files if sum(1 for _ in open(p, errors="replace")) >= 800]
    mids = [p for p in files if 150 <= sum(1 for _ in open(p, errors="replace")) < 800]
    log = open("/tmp/a2gates/llama-server-cachebench-pinned-noreuse.log", "ab")
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--port", str(PORT), "--host",
         "127.0.0.1", "-t", "6", "-ngl", "0", "--parallel", "1",
         "-c", "16384"],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True)
    print(f"llama-server pid {proc.pid} port {PORT}", flush=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                sys.exit("server exited during startup")
            try:
                body = json.dumps({"prompt": "ready", "max_tokens": 1,
                                   "temperature": 0}).encode()
                rq = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=60) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            sys.exit("server never ready")
        print("server ready", flush=True)
        out = open(OUT, "w")
        n = 10
        for arm, pool, move in (("s1_small_move_pinned", mids, 55),
                                ("s2_large_move_pinned", bigs, 430)):
            for i in range(n):
                p = pool[i % len(pool)]
                lines = p.read_text(errors="replace").splitlines()
                a0 = max(150, len(lines) // 3)
                a1 = min(a0 + move, len(lines) - SUFFIX_LINES)
                if a1 <= a0:
                    continue
                r0 = complete(render_pinned(p.name, lines, a0))
                r1 = complete(render_pinned(p.name, lines, a1))
                rec = dict(arm=arm, file=p.name, anchor0=a0, anchor1=a1,
                           r1_prompt_n=r1["prompt_n"], r1_cached=r1["cached"],
                           cached_frac=round(r1["cached"] / r1["prompt_n"], 4)
                           if r1["prompt_n"] else None,
                           r1_prompt_ms=r1["prompt_ms"],
                           r0_prompt_n=r0["prompt_n"])
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps(rec), flush=True)
        out.close()
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
