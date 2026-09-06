#!/usr/bin/env python3
"""S1 spec-decode bench (queue §3 S-series): replay the S0 frozen trace set
(/mnt/h/sepalith/datasets/spec_traces, MANIFEST.md) through llama-server
arms and measure acceptance (tok/step) + wall tok/s + TTFT at both ctx
classes (2k, 8k). CPU llama-server only (ports 18xxx, no CUDA context —
b10453 CPU build per the repo serving convention).

Arms — SINGLE-MODE ONLY (A2 §4.5 rule: no stacked spec claims):
  baseline            no speculative flags (the reference + greedy oracle)
  ngram-simple        --spec-type ngram-simple; depth = drafted tokens/step
                      via --spec-draft-n-max (corrected 2026-09-06: the
                      binary's --spec-ngram-simple-size-m is the draft
                      m-gram LENGTH, not the step depth); sweep via
                      ngram-simple@n
  draft-mtp           --spec-type draft-mtp on the MTP-preserving export
                      (experiments/models/mtp-b4_qwen35_2b-Q8_0.gguf, nextn
                      tensors embedded; produced by
                      experiments/training/export_gguf_mtp.py); depth =
                      --spec-draft-n-max; sweep via draft-mtp@n
  model-draft         --spec-type draft-simple --spec-draft-model <0.8b>
                      (the Matryoshka S-tier stand-in: b2 qwen3.5-0.8b,
                      same tokenizer family; swap in the A2 8L-prefix tier
                      when it exists); depth = --spec-draft-n-max;
                      sweep via model-draft@n

RIG NOTES (binding, from the S0 MANIFEST "For S1's rig"):
  -c 10240 (8k p95 prompt is 9163 + generation; the 8192 default truncates),
  stop ">>>>>>> UPDATED", max_tokens 64, Qwen-family token counts (the
  trace token bands were counted with the Qwen3.5 tokenizer; prompt_n from
  the server is the authoritative per-request count).

Metrics per request (native /completion, stream:true, temperature 0):
  ttft_ms        request-send -> first content chunk (streamed), cold
                 (cache_prompt False — the fresh-edit scenario)
  prompt_ms      server-reported prefill time
  gen_tps        tokens_predicted / predicted_ms (decode wall tok/s)
  wall_ms        total request wall time
  accept_tps     acceptance tok/step = 1 + accepted/verify_steps, from
                 /metrics counter diffs around the request
                 (spec_decode_num_accepted_tokens_total /
                  spec_decode_num_drafts_total / ..._num_draft_tokens_total)
  accept_rate    accepted draft tokens / drafted draft tokens
  warm_prompt_ms / warm_ttft_ms  second identical request with
                 cache_prompt True (the keystroke/cache-hit case; H1 tie-in)
  matches_baseline  greedy output identical to the baseline arm's output
                 for the same trace (spec is distribution-lossless — a
                 mismatch means wiring trouble, not quality drift)

Usage (from repo root):
  smoke: .venv/bin/python experiments/eval/spec_bench.py --smoke
  full:  .venv/bin/python experiments/eval/spec_bench.py \
             --arms baseline,ngram-simple,draft-mtp,model-draft \
             --ctx-classes 2k,8k --n-traces 100 --reps 3 \
             --depths-ngram 2,3,4,8,16,48 --depths-draft 1,2,3,4,5
Writes experiments/eval/results_specbench/<run_id>/per_request.jsonl +
summary.json (run_id timestamps + arm set).
"""
import argparse
import fcntl
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/eval
EXP = HERE.parent                               # experiments
TRACES = Path("/mnt/h/sepalith/datasets/spec_traces/traces.jsonl")
SERVER = EXP / "bin" / "llama" / "llama-b10453" / "llama-server"
MODEL = EXP / "models" / "b4_qwen35_2b-Q8_0.gguf"          # target (b4)
MODEL_MTP = EXP / "models" / "mtp-b4_qwen35_2b-Q8_0.gguf"  # b4 + nextn
MODEL_DRAFT = EXP / "models" / "b2_qwen35_08b-Q8_0.gguf"   # 0.8b family draft
OUTROOT = HERE / "results_specbench"

STOP = ">>>>>>> UPDATED"
MAX_TOKENS = 64           # MANIFEST: targets p95=58, max=60 -> 64 cap
CTX = 10240               # MANIFEST: >= 10240 (8k p95 prompt 9163 + gen)
BASE_PORT = 18401         # S1 rig ports (H1 owns 1831x)
BATTERY_LOCK = Path("/tmp/b_battery.lock")

# ---------------------------------------------------------------------------
# trace loading


def load_traces(path=TRACES):
    """Read traces.jsonl -> {ctx_class: [row, ...]} preserving file order.

    Asserts the MANIFEST invariants the rig depends on (band + pairing
    fields present) so a silently-corrupt set fails loudly, not slowly.
    """
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            assert r["ctx_class"] in ("2k", "8k"), r["ctx_class"]
            assert isinstance(r["prompt"], str) and r["prompt"]
            assert isinstance(r["target"], str) and r["target"].endswith(STOP), \
                f"{r.get('trace_id')}: target lacks terminal {STOP!r}"
            rows.append(r)
    by_class = {"2k": [], "8k": []}
    for r in rows:
        by_class[r["ctx_class"]].append(r)
    return by_class


def sample_traces(rows, n, seed=20260905):
    """Deterministic per-class sample (stable across runs/reps)."""
    if n >= len(rows):
        return list(rows)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]


def paired_smoke_traces(by_class, seed=20260905):
    """One trace PAIR (same edit, both ctx classes) for the smoke run."""
    ids2 = {r["trace_id"].rsplit("-", 1)[0]: r for r in by_class["2k"]}
    for r in by_class["8k"]:
        base = r["trace_id"].rsplit("-", 1)[0]
        if base in ids2:
            return {"2k": [ids2[base]], "8k": [r]}
    raise RuntimeError("no paired traces found — set broken")


# ---------------------------------------------------------------------------
# acceptance math


def acceptance_from_delta(d_draft_tokens, d_accepted, d_verify_steps):
    """(tok_per_step, accept_rate) from /metrics counter deltas.

    tok_per_step = 1 + accepted/verify_steps: every verify step emits at
    least the sampled token (baseline pace = 1.0); accepted drafts ride
    along. accept_rate = accepted/drafted (the classic per-draft-token
    acceptance). Returns (None, None) when no speculative activity happened
    (baseline arm, or a spec arm that never drafted) — NOT 1.0, which would
    silently claim baseline==perfect-draft.
    """
    if d_verify_steps <= 0 or d_draft_tokens <= 0:
        return None, None
    tok_per_step = 1.0 + d_accepted / d_verify_steps
    accept_rate = d_accepted / d_draft_tokens
    return tok_per_step, accept_rate


def parse_prometheus(text):
    """'llamacpp:name{label="v"} 123' -> {name: value}; last write wins."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\S+?)(?:\{[^}]*\})?\s+(\S+)$", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


SPEC_COUNTERS = (
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
)


def spec_deltas(before, after):
    """Counter deltas for the three spec counters (missing -> 0)."""
    return tuple(after.get(c, 0.0) - before.get(c, 0.0) for c in SPEC_COUNTERS)


def acceptance_from_response(timings):
    """(tok_per_step, accept_rate) from the /completion final timings.

    b10453 exposes per-request spec stats in the final stream chunk:
    draft_n (drafted tokens), draft_n_accepted, predicted_n (emitted).
    Verify steps = predicted_n - draft_n_accepted (each step emits its
    sampled token plus any accepted drafts), so
        tok_per_step = predicted_n / (predicted_n - draft_n_accepted)
    (baseline pace = 1.0; the same formula the server logs as
    'mean len'). Returns (None, None) when nothing was drafted (baseline
    arm / spec arm that never drafted) — NOT 1.0.
    """
    if not timings:
        return None, None
    drafted = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted")
    predicted = timings.get("predicted_n") or 0
    if drafted <= 0 or accepted is None or predicted <= 0:
        return None, None
    steps = predicted - accepted
    if steps <= 0:
        return None, None
    return predicted / steps, accepted / drafted


# ---------------------------------------------------------------------------
# streaming client + TTFT


class StreamResult:
    __slots__ = ("text", "ttft_ms", "wall_ms", "timings", "n_predicted",
                 "stop_hit", "error")

    def __init__(self):
        self.text, self.ttft_ms, self.wall_ms = "", None, None
        self.timings, self.n_predicted, self.stop_hit, self.error = \
            {}, None, False, None


def stream_completion(port, prompt, max_tokens=MAX_TOKENS, stop=STOP,
                      cache_prompt=False, timeout=1800):
    """Native /completion with stream:true. TTFT = send -> first CONTENT
    chunk (not the open event, not role chunks): time-to-first-token as the
    editor extension would feel it. The final SSE message carries timings +
    tokens_predicted."""
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens,
                       "temperature": 0.0, "stop": [stop],
                       "stream": True, "cache_prompt": cache_prompt}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    res = StreamResult()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
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
                if content and res.ttft_ms is None:
                    res.ttft_ms = (time.time() - t0) * 1000.0
                if content:
                    res.text += content
                if chunk.get("stop"):
                    res.stop_hit = True
                    if "timings" in chunk:
                        res.timings = chunk["timings"]
                    res.n_predicted = chunk.get("tokens_predicted")
                    break
                if chunk.get("final") or "timings" in chunk and chunk.get("stop_type"):
                    # some builds put timings on a final non-stop chunk
                    res.timings = chunk.get("timings", res.timings)
                    res.n_predicted = chunk.get("tokens_predicted",
                                                res.n_predicted)
    except Exception as e:  # noqa: BLE001 — record, never crash the leg
        res.error = repr(e)[:300]
    res.wall_ms = (time.time() - t0) * 1000.0
    if not res.timings:
        res.timings = {}
    return res


def gen_tps(res):
    """Decode-phase wall tok/s from server timings (None if unusable)."""
    t = res.timings or {}
    n = res.n_predicted if res.n_predicted is not None \
        else t.get("tokens_predicted") or t.get("predicted_n")
    ms = t.get("predicted_ms")
    if n and ms and ms > 0:
        return n / (ms / 1000.0)
    if res.stop_hit and res.wall_ms and res.ttft_ms:
        # fallback: tokens (estimated) over decode wall — flagged by
        # timings_ok=False downstream; only used if server omits timings
        return None
    return None


# ---------------------------------------------------------------------------
# server management (tracked PID; ports 18xxx; battery lock respected)


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class SpecServer:
    """Own one CPU llama-server child with an arm's spec flags."""

    def __init__(self, model, port, extra_flags, threads=8, ctx=CTX,
                 log_path=None, server=None, foreground=False):
        self.server = Path(server) if server is not None else SERVER
        self.foreground = foreground
        self.model, self.port = str(model), port
        self.threads, self.ctx, self.extra = threads, ctx, list(extra_flags)
        self.log_path = Path(log_path or HERE / f"llama-server-spec-{port}.log")
        self.proc = None

    def cmd(self):
        c = [str(self.server), "-m", self.model, "--port", str(self.port),
             "--host", "127.0.0.1", "-t", str(self.threads),
             "--parallel", "1", "-c", str(self.ctx), "-ngl", "0"]
        return c + self.extra

    def start(self, ready_timeout=1800):
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} in use; tracked-PID-only "
                               f"policy — not touching it")
        cmd = self.cmd()
        # hold the battery lock across bind so battery servers can't
        # double-claim our port (protocol: flock /tmp/b_battery.lock)
        lock_f = os.open(str(BATTERY_LOCK), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            with open(self.log_path, "ab") as log:
                log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n"
                          .encode())
                log.flush()
                self.proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=not self.foreground)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            os.close(lock_f)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server rc={self.proc.returncode} "
                    f"log tail: {self.log_tail()}")
            try:
                body = json.dumps({"prompt": "ready", "max_tokens": 1,
                                   "temperature": 0}).encode()
                rq = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=120) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    json.JSONDecodeError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server not ready in time; "
                           f"log tail: {self.log_tail()}")

    def log_tail(self, n=2000):
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return "<no log>"

    def stop(self):
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(75):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass
        finally:
            self.proc = None


def get_metrics(port):
    """Best-effort: b10453 llama-server may 501 on /metrics unless enabled —
    the per-request response timings (draft_n/draft_n_accepted) are the
    primary acceptance source; this is only the cross-check path."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=60) as r:
            return parse_prometheus(r.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return {}


# ---------------------------------------------------------------------------
# arms


def arm_flags(arm, model=MODEL, model_mtp=MODEL_MTP, model_draft=MODEL_DRAFT):
    """(server_model, extra_flags, label) for one single-mode arm config.

    arm grammar: name | name@depth (depth = drafted tokens per step,
    configured with --spec-draft-n-max for every speculative arm).
    """
    name, _, depth = arm.partition("@")
    depth = int(depth) if depth else None
    if name == "baseline":
        assert depth is None, "baseline takes no depth"
        return model, [], arm
    if name == "ngram-simple":
        flags = ["--spec-type", "ngram-simple"]
        if depth is not None:
            # CORRECTED 2026-09-06 (flag semantics verified against the
            # b10453 binary --help + runtime evidence): depth = drafted
            # tokens/step is --spec-draft-n-max (applies to ALL spec types;
            # default 3). --spec-ngram-simple-size-m is the draft M-GRAM
            # LENGTH (default 48), NOT the step depth — setting it to 2
            # produced ZERO drafts (run-20260905T192522's mis-flagged arm,
            # 5 rows, kept as the no-draft-overhead datapoint).
            flags += ["--spec-draft-n-max", str(depth)]
        return model, flags, arm
    if name == "draft-mtp":
        if not Path(model_mtp).exists():
            raise FileNotFoundError(
                f"{model_mtp} missing — run "
                f"experiments/training/export_gguf_mtp.py first")
        flags = ["--spec-type", "draft-mtp"]
        if depth is not None:
            flags += ["--spec-draft-n-max", str(depth)]
        return model_mtp, flags, arm
    if name == "model-draft":
        if not Path(model_draft).exists():
            raise FileNotFoundError(f"{model_draft} missing")
        flags = ["--spec-type", "draft-simple",
                 "--spec-draft-model", str(model_draft)]
        if depth is not None:
            flags += ["--spec-draft-n-max", str(depth)]
        return model, flags, arm
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# bench


def run_leg(arm, traces, port, out_fh, baseline_texts=None, reps=1):
    """Replay traces against one arm on one server. Writes one JSON row per
    (trace, rep). Returns (rows, texts) — texts = greedy output per
    trace_id from rep 0 (the lossless-check oracle for spec arms)."""
    model, flags, label = arm_flags(arm)
    srv = SpecServer(model, port, flags,
                     log_path=OUTROOT / f"llama-server-{label.replace('/', '_')}.log")
    rows, texts = [], {}
    try:
        srv.start()
        for tr in traces:
            for rep in range(reps):
                m0 = get_metrics(port)
                res = stream_completion(port, tr["prompt"])
                m1 = get_metrics(port)
                d_tok, d_acc, d_ver = spec_deltas(m0, m1)
                # PRIMARY: per-request response stats (draft_n fields in the
                # final chunk); /metrics deltas as cross-check fallback
                tps, rate = acceptance_from_response(res.timings)
                if tps is None and d_tok > 0:
                    tps, rate = acceptance_from_delta(d_tok, d_acc, d_ver)
                # warm pass (keystroke/cache-hit case)
                warm = stream_completion(port, tr["prompt"], cache_prompt=True)
                row = dict(
                    arm=label, trace_id=tr["trace_id"], ctx_class=tr["ctx_class"],
                    rep=rep,
                    prompt_tokens=tr.get("prompt_tokens"),
                    target_tokens=tr.get("target_tokens"),
                    n_prompt_srv=(res.timings or {}).get("prompt_n"),
                    prompt_ms=(res.timings or {}).get("prompt_ms"),
                    n_predicted=res.n_predicted
                    or (res.timings or {}).get("predicted_n"),
                    draft_n=(res.timings or {}).get("draft_n"),
                    draft_n_accepted=(res.timings or {}).get("draft_n_accepted"),
                    gen_tps=gen_tps(res),
                    ttft_ms=res.ttft_ms, wall_ms=res.wall_ms,
                    stop_hit=res.stop_hit, error=res.error,
                    draft_tokens_delta=d_tok, accepted_delta=d_acc,
                    verify_steps_delta=d_ver,
                    accept_tps=tps, accept_rate=rate,
                    warm_prompt_ms=(warm.timings or {}).get("prompt_ms"),
                    warm_ttft_ms=warm.ttft_ms,
                )
                if baseline_texts is not None and tr["trace_id"] in baseline_texts:
                    row["matches_baseline"] = \
                        (res.text == baseline_texts[tr["trace_id"]]
                         if not res.error else False)
                if rep == 0 and not res.error:
                    texts[tr["trace_id"]] = res.text
                    row["gen_text"] = res.text  # persisted greedy oracle
                    # (additive 2026-09-06: rep-0 rows carry the greedy text
                    #  so the lossless check can span runs)
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                rows.append(row)
                ok = "ok" if not res.error else f"ERR {res.error[:60]}"
                print(json.dumps({k: row[k] for k in
                                  ("arm", "ctx_class", "trace_id", "rep",
                                   "prompt_ms", "gen_tps", "ttft_ms",
                                   "accept_tps", "accept_rate")} | {"st": ok}),
                      flush=True)
    finally:
        srv.stop()
    return rows, texts


def summarize(rows):
    """Per (arm, ctx_class) medians + speedup vs same-class baseline."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        for cc in sorted({r["ctx_class"] for r in rows}):
            rs = [r for r in rows if r["arm"] == arm and r["ctx_class"] == cc
                  and not r.get("error")]
            if not rs:
                continue

            def med(key, rnd=2):
                xs = [r[key] for r in rs
                      if r.get(key) is not None]
                return round(statistics.median(xs), rnd) if xs else None

            acc = [r for r in rs if r.get("accept_tps") is not None]
            key = f"{arm}|{cc}"
            out[key] = dict(
                n=len(rs), n_err=sum(1 for r in rs if r.get("error")),
                gen_tps_med=med("gen_tps"), ttft_ms_med=med("ttft_ms"),
                prompt_ms_med=med("prompt_ms"),
                warm_ttft_ms_med=med("warm_ttft_ms"),
                warm_prompt_ms_med=med("warm_prompt_ms"),
                accept_tps_med=(round(statistics.median(
                    [r["accept_tps"] for r in acc]), 3) if acc else None),
                accept_rate_med=(round(statistics.median(
                    [r["accept_rate"] for r in acc]), 3) if acc else None),
                n_with_accept=len(acc),
                stop_hit=sum(1 for r in rs if r.get("stop_hit")),
            )
            if any("matches_baseline" in r for r in rs):
                out[key]["matches_baseline"] = sum(
                    1 for r in rs if r.get("matches_baseline"))
    base = {cc: out.get(f"baseline|{cc}", {}).get("gen_tps_med")
            for cc in ("2k", "8k")}
    for key, v in out.items():
        cc = key.split("|", 1)[1]
        if base.get(cc) and v.get("gen_tps_med"):
            v["speedup_vs_baseline"] = round(v["gen_tps_med"] / base[cc], 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="baseline,ngram-simple",
                    help="comma list; name or name@depth "
                         "(baseline,ngram-simple[@m],draft-mtp[@n],"
                         "model-draft[@n])")
    ap.add_argument("--ctx-classes", default="2k,8k")
    ap.add_argument("--n-traces", type=int, default=100,
                    help="per ctx class (deterministic sample; full set = "
                         "pass a value >= 550)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="1 paired trace per class, arms baseline + "
                         "ngram-simple, 1 rep — end-to-end wiring proof")
    ap.add_argument("--port", type=int, default=BASE_PORT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    by_class = load_traces()
    if a.smoke:
        sel = paired_smoke_traces(by_class)
        arms, reps = ["baseline", "ngram-simple"], 1
        run_id = f"smoke-{time.strftime('%Y%m%dT%H%M%S')}"
    else:
        arms = [x.strip() for x in a.arms.split(",") if x.strip()]
        reps = a.reps
        sel = {cc: sample_traces(by_class[cc], a.n_traces)
               for cc in a.ctx_classes.split(",") if cc in by_class}
        run_id = f"run-{time.strftime('%Y%m%dT%H%M%S')}-n{a.n_traces}-r{a.reps}"
    out_dir = Path(a.out_dir) if a.out_dir else OUTROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id {run_id} arms {arms} classes {list(sel)} "
          f"traces/class {[len(v) for v in sel.values()]} reps {reps}",
          flush=True)

    with open(out_dir / "per_request.jsonl", "w") as fh:
        fh.write(json.dumps(dict(run_id=run_id, arms=arms, reps=reps,
                                 n_per_class={k: len(v)
                                              for k, v in sel.items()},
                                 stop=STOP, max_tokens=MAX_TOKENS, ctx=CTX,
                                 threads=a.threads, port=a.port,
                                 model=str(MODEL), model_mtp=str(MODEL_MTP),
                                 model_draft=str(MODEL_DRAFT))) + "\n")
        all_rows, baseline_texts = [], {}
        port = a.port
        # baseline must run FIRST when present: its greedy outputs are the
        # lossless-check oracle for every spec arm
        ordered = ([x for x in arms if x == "baseline"] +
                   [x for x in arms if x != "baseline"])
        for arm in ordered:
            traces = [t for cc in sorted(sel) for t in sel[cc]]
            bt = None if arm == "baseline" else baseline_texts
            rows, texts = run_leg(arm, traces, port, fh, baseline_texts=bt,
                                  reps=reps)
            if arm == "baseline":
                baseline_texts = texts
            all_rows.extend(rows)
            port += 1  # rotate ports so a killed server never blocks the next

    summary = dict(run_id=run_id, smoke=bool(a.smoke), arms=arms,
                   n_per_class={k: len(v) for k, v in sel.items()},
                   reps=reps, host_load=open("/proc/loadavg").read().split()[0],
                   arms_summary=summarize(all_rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["arms_summary"], indent=1), flush=True)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
