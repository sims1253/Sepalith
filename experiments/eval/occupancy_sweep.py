#!/usr/bin/env python3
"""A2 gate 5 — occupancy sweep: needle-COPY battery, hybrid vs dense (RT4 2.2).

Instrument: needle-copy (verbatim span reproduction), NOT needle-retrieval.
Prompt = real R code filler (deployment occupancy: identifier bindings from
the actual CRAN corpus staged by gate 4, lorem fill rejected per the gate
spec) with one marked span planted at a controlled position, followed by an
instruction to reproduce the span verbatim.

Battery: ctx targets {1K, 2K, 4K, 8K} (total prompt tokens, model's own
tokenizer) x needle lengths {40, 100, 300} tokens (nominal; actual measured
per row) x positions {start, middle, end} (fraction of the filler) x 10
seeds = 360 prompts per arm.

Scoring per row: exact line-normalized reproduction, char-level edit
similarity (difflib ratio), token-level first-divergence position,
near-miss taxonomy (exact / near_miss>=0.85 / partial>=0.5 / garbage).

Server: own child llama-server (tracked PID, SIGTERM teardown), readiness =
POST /v1/completions 200, temp 0 greedy, CPU threads 6, ctx 16384 —
identical settings for both arms.

Usage:
  .venv/bin/python3 -u occupancy_sweep.py \
      --model experiments/models/... --server <llama-server> \
      --label lfm25_hybrid --out experiments/eval/results_occupancy_lfm25.jsonl
Resumable: rows already present in --out (same cell+seed) are skipped.
"""
import argparse
import difflib
import json
import os
import random
import re
import signal
import string
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

CTXS = [1024, 2048, 4096, 8192]
NLENS = [40, 100, 300]
POSS = ["start", "middle", "end"]
POS_FRAC = {"start": 0.03, "middle": 0.50, "end": 0.97}
N_SEEDS = 10
MAX_TOKENS_FOR = {40: 140, 100: 260, 300: 680}          # raw mode
MAX_TOKENS_CHAT = {40: 600, 100: 1200, 300: 1500}      # chat: thinking budget
# 1K-class-100 cells MEASURED to converge at ~930 comp tokens (thinking +
# answer); 800 truncated them. 300-class gets headroom for span+thinking.
# NOTE: stop strings are RAW-mode only — in chat mode a thinking model
# quotes the delimiter inside its reasoning, the stop fires there, and the
# answer never arrives (measured on LFM2.5: 38 tokens then empty content).
STOP = ["# END_BLOCK"]
TAX_EXACT, TAX_NEAR, TAX_PARTIAL = 0.85, 0.85, 0.5

INSTR = ("The R script above contains exactly one marked block. It starts at "
         "the line '# SECRET_BLOCK_{bid}' and ends at the line '# END_BLOCK'. "
         "Reproduce that entire block VERBATIM, character for character, "
         "including both delimiter lines. Output only the block, nothing "
         "else.")


def http_post(port, path, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(port, text):
    """Token count via llama-server /tokenize (model's own tokenizer)."""
    r = http_post(port, "/tokenize", {"content": text}, timeout=300)
    return len(r.get("tokens", []))


# ---------------------------------------------------------------- needles
def hexs(rng, n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def b36(rng, n):
    a = string.ascii_lowercase + string.digits
    return "".join(rng.choice(a) for _ in range(n))


NEEDLE_LINE_TPL = [
    '  key_{h4} <- "{w6}-{d3}",  # pin {d2}',
    "  vec_{h4} <- c({f1}, {f2}, {f1} * {f2})",
    "  idx_{h4} <- {d2}:{d3}L",
    "  fn_{h4} <- function(x, k = {d2}) x * {f1} + k",
    '  tag_{h4} <- toupper("{w4}{d2}")',
    "  wt_{h4} <- round({f1} / {f2}, {d1})",
    "  map_{h4} <- list(a = {d2}, b = \"{w4}\", c = c({f1}, {f2}))",
]
# compact variants (~8-12 tok/line) for small-token-target needles where
# the templates above (~20-30 tok/line) are too coarse to hit +/-18%
NEEDLE_LINE_TPL_SHORT = [
    "  s_{h4} <- {d3}L",
    '  t_{h4} <- "{w4}"',
    "  u_{h4} <- {f1}",
    "  v_{h4} <- c({d2}, {d3})",
]


def needle_body(rng, target_chars, short=False):
    """R-plausible body lines with unique random tokens; ~target_chars long."""
    tpls = NEEDLE_LINE_TPL_SHORT if short else NEEDLE_LINE_TPL
    lines = []
    n = 0
    while n < target_chars:
        tpl = rng.choice(tpls)
        line = tpl.format(h4=hexs(rng, 4), w6=b36(rng, 6), w4=b36(rng, 4),
                          d3=str(rng.randint(100, 999)),
                          d2=str(rng.randint(10, 99)),
                          d1=str(rng.randint(1, 9)),
                          f1=round(rng.uniform(0.1, 9.9), 3),
                          f2=round(rng.uniform(0.1, 9.9), 3))
        lines.append(line)
        n += len(line)
    return lines


def make_needle_tok(port, rng, nlen_target):
    """Needle sized to the MODEL'S OWN tokenizer (dense hex content runs
    ~2 chars/token, so a chars heuristic misses by >2x). Iterate line-count
    rescaling; keep the BEST candidate seen (line granularity means exact
    convergence is not guaranteed for the 40-token class)."""
    bid = hexs(rng, 8)
    short = nlen_target <= 60
    per_line = 14 if short else 85          # first-guess chars per line
    body = needle_body(rng, max(1, nlen_target) * per_line, short=short)

    def build(b):
        return "\n".join([f"# SECRET_BLOCK_{bid}", ".sepalith_probe <- list("]
                         + b + [")", "# END_BLOCK"])

    best = None                             # (|err|, ntok, needle)
    for _ in range(8):
        needle = build(body)
        ntok = tokenize(port, needle)
        err = abs(ntok - nlen_target)
        if best is None or err < best[0]:
            best = (err, ntok, needle)
        if err <= 0.18 * nlen_target:
            break
        want = max(1, round(len(body) * nlen_target / max(1, ntok)))
        if want > len(body):
            body = body + needle_body(rng, (want - len(body)) * per_line,
                                      short=short)
        else:
            body = body[:want]
            if not body:
                break
    return bid, best[2], best[1]


# ---------------------------------------------------------------- filler
class Filler:
    """Real-R filler with per-block token counts measured ONCE against the
    serving model's own tokenizer (random-block density varies ~2x between
    blocks, so a global chars/token ratio oscillates — the probe runs
    showed 1879/2312-token '1K' prompts before this fix)."""

    def __init__(self, corpus_dir, port=None):
        self.blocks = []                     # (lines, tok)
        for p in sorted(Path(corpus_dir).glob("*.R")):
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError:
                continue
            # contiguous ~30-line blocks of real R code (local coherence,
            # real identifier-binding occupancy)
            for i in range(0, len(lines) - 30, 30):
                blk = [l.rstrip() for l in lines[i:i + 30]]
                blk = [l for l in blk if "SECRET_BLOCK" not in l]
                if sum(len(l) for l in blk) > 200:
                    self.blocks.append(blk)
        if len(self.blocks) < 50:
            sys.exit(f"corpus too small: {len(self.blocks)} blocks")
        self.tok = None
        if port is not None:
            self.measure(port)

    def measure(self, port):
        self.tok = []
        t0 = time.time()
        for blk in self.blocks:
            self.tok.append(tokenize(port, "\n".join(blk)))
        print(f"filler measured: {len(self.blocks)} blocks, "
              f"{sum(self.tok)} tok, {round(time.time() - t0, 1)}s",
              flush=True)

    def sample(self, rng, tok_budget):
        """Blocks to ~tok_budget MODEL tokens. Line-trims the last block."""
        idx = list(range(len(self.blocks)))
        rng.shuffle(idx)
        out, n = [], 0
        for i in idx:
            t = self.tok[i]
            if n + t > tok_budget and out:
                # trim last block by lines, proportional token cost
                per = t / max(1, len(self.blocks[i]))
                keep = max(0, round((tok_budget - n) / per))
                if keep > 2:
                    out.extend(self.blocks[i][:keep])
                    n += int(keep * per)
                break
            out.extend(self.blocks[i])
            out.append("")                   # blank separator between files
            n += t + 1
            if n >= tok_budget:
                break
        return out, n


# ---------------------------------------------------------------- scoring
def normalize(text):
    """Line-trailing-ws strip + drop ALL blank lines. The needle contains
    no interior blank lines by construction, so blank-line differences are
    always formatting noise (gemma inserts one before END_BLOCK)."""
    lines = [l.rstrip() for l in text.strip().splitlines()]
    return "\n".join(l for l in lines if l)


def extract_span(text, bid):
    """Pull the reproduced block out of the raw completion."""
    t = text.strip()
    m = re.search(r"#\s*SECRET_BLOCK", t)
    if m:
        t = t[m.start():]
    e = re.search(r"#\s*END_BLOCK", t)
    if e:
        t = t[:e.end()]
    return t


def score(ref, out, bid):
    ref_n, out_n = normalize(ref), normalize(out)
    exact = int(ref_n == out_n)
    sim = difflib.SequenceMatcher(None, ref_n, out_n).ratio()
    ref_toks, out_toks = ref_n.split(), out_n.split()
    div = None
    for i, (a, b) in enumerate(zip(ref_toks, out_toks)):
        if a != b:
            div = i
            break
    prefix_frac = (1.0 if div is None and len(ref_toks) == len(out_toks)
                   else (0 if div is None else div / max(1, len(ref_toks))))
    if exact:
        tax = "exact"
    elif sim >= TAX_NEAR:
        tax = "near_miss"
    elif sim >= TAX_PARTIAL:
        tax = "partial"
    else:
        tax = "garbage"
    return dict(exact=exact, edit_sim=round(sim, 4),
                first_div_tok=div, prefix_frac=round(prefix_frac, 4),
                tax=tax)


# ---------------------------------------------------------------- run
def build_prompt(port, filler, rng, ctx_target, nlen_target, pos,
                 overhead_tok=None):
    """Assemble filler+needle+instruction to ~ctx_target MODEL tokens.
    overhead_tok (instruction+delimiter tokens, measured once) is cached
    by the caller across the battery."""
    bid, needle, nlen_tok = make_needle_tok(port, rng, nlen_target)
    if overhead_tok is None:
        instr_probe = INSTR.format(bid=bid)
        overhead_tok = tokenize(port, instr_probe + "\n# verbatim block:\n"
                                + "# ---- occupancy filler (real R corpus) ----"
                                ) + 4
    budget = max(300, ctx_target - nlen_tok - overhead_tok)
    flines, ftok = filler.sample(rng, budget)
    # position cut on the TOKEN axis (fraction of filler tokens before needle)
    cut = 0
    acc = 0
    for j, l in enumerate(flines):
        if acc >= ftok * POS_FRAC[pos]:
            cut = j
            break
        acc += max(1, len(l)) // 4 + 1      # cheap per-line token estimate
        cut = j + 1
    pre, post = flines[:cut], flines[cut:]
    instr = INSTR.format(bid=bid)
    prompt = ("# ---- occupancy filler (real R corpus) ----\n"
              + "\n".join(pre) + "\n" + needle + "\n"
              + "\n".join(post) + "\n\n" + instr + "\n# verbatim block:\n")
    # final verification + spacer-line correction (spacers ~4 tok each)
    for _ in range(3):
        ntok = tokenize(port, prompt)
        err = ntok - ctx_target
        if abs(err) <= 0.04 * ctx_target:
            break
        spacer = "# - - -\n" * max(0, int(abs(err) / 4))
        if err < 0:
            post = post + [s for s in spacer.strip().split("\n") if s]
        else:
            post = post[: max(0, len(post) - int(err / 4))]
        prompt = ("# ---- occupancy filler (real R corpus) ----\n"
                  + "\n".join(pre) + "\n" + needle + "\n"
                  + "\n".join(post) + "\n\n" + instr + "\n# verbatim block:\n")
    ntok = tokenize(port, prompt)
    return prompt, bid, needle, ntok, nlen_tok, overhead_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/tmp/a2gates/bench_corpus")
    ap.add_argument("--port", type=int, default=18105)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="stop after N rows (0 = full battery; probe aid)")
    ap.add_argument("--mode", choices=["raw", "chat"], default="raw",
                    help="serving protocol: raw /v1/completions (the "
                         "vehicle's PSM-style format) or chat "
                         "/v1/chat/completions (native format for "
                         "thinking-style instruct models; chosen per arm "
                         "by the 1K calibration so each model runs in the "
                         "format it was trained for)")
    ap.add_argument("--attach", action="store_true",
                    help="attach to an already-running server on --port "
                         "(no spawn, no teardown) — used after the "
                         "coordinator's mid-run GPU swap")
    args = ap.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                if "error" in r:
                    continue            # failed rows are re-run on resume
                done.add((r["ctx_target"], r["nlen_target"], r["pos"],
                          r["seed"]))
            except Exception:
                pass
        fout = out.open("a")
    else:
        fout = out.open("w")
        fout.write(json.dumps(dict(
            meta=True, label=args.label, model=os.path.abspath(args.model),
            server=os.path.abspath(args.server), port=args.port,
            threads=args.threads, ctx=args.ctx, temp=0, mode=args.mode,
            cts=CTXS, nlens=NLENS, poss=POSS, seeds=N_SEEDS,
            corpus=args.corpus)) + "\n")
        fout.flush()
    print(f"resume: {len(done)} rows already present", flush=True)

    filler = Filler(args.corpus)
    Path("/tmp/a2gates").mkdir(parents=True, exist_ok=True)
    proc = None
    if not args.attach:
        slog = open(f"/tmp/a2gates/llama-server-occ-{args.label}.log", "ab")
        proc = subprocess.Popen(
            [args.server, "-m", args.model, "--port", str(args.port),
             "--host", "127.0.0.1", "-t", str(args.threads), "-ngl", "0",
             "--parallel", "1", "-c", str(args.ctx)],
            stdout=slog, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        print(f"llama-server pid {proc.pid} port {args.port} "
              f"label {args.label}", flush=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 1800:
            if proc is not None and proc.poll() is not None:
                sys.exit(f"server exited during startup — see /tmp/a2gates/"
                         f"llama-server-occ-{args.label}.log")
            try:
                http_post(args.port, "/v1/completions",
                          {"prompt": "ready", "max_tokens": 1,
                           "temperature": 0}, timeout=60)
                break
            except Exception:
                time.sleep(2)
        else:
            sys.exit("server never ready")
        print("server ready", flush=True)
        filler.measure(args.port)

        i = 0
        overhead = None
        t_start = time.time()
        for seed in range(N_SEEDS):
            for ctx in CTXS:
                for nlen in NLENS:
                    for pos in POSS:
                        key = (ctx, nlen, pos, seed)
                        if key in done:
                            continue
                        rng = random.Random(
                            zlib.crc32(repr((args.label, ctx, nlen, pos,
                                             seed)).encode()))
                        prompt, bid, needle, ntok, nlen_tok, overhead = \
                            build_prompt(args.port, filler, rng, ctx, nlen,
                                         pos, overhead)
                        try:
                            t0 = time.time()
                            if args.mode == "chat":
                                mt = MAX_TOKENS_CHAT[nlen]
                                r = http_post(
                                    args.port, "/v1/chat/completions",
                                    {"messages": [{"role": "user",
                                                   "content": prompt}],
                                     "max_tokens": mt,
                                     "temperature": 0, "stream": False},
                                    timeout=3600)
                            else:
                                mt = MAX_TOKENS_FOR[nlen]
                                r = http_post(
                                    args.port, "/v1/completions",
                                    {"prompt": prompt,
                                     "max_tokens": mt,
                                     "temperature": 0, "stream": False,
                                     "cache_prompt": False, "stop": STOP},
                                    timeout=3600)
                        except Exception as e:
                            rec = dict(model=args.label, ctx_target=ctx,
                                       nlen_target=nlen, pos=pos, seed=seed,
                                       error=str(e)[:200])
                            fout.write(json.dumps(rec) + "\n")
                            fout.flush()
                            continue
                        choice = r.get("choices", [{}])[0]
                        if args.mode == "chat":
                            msg = choice.get("message", {})
                            text = msg.get("content") or ""
                            # thinking-style models can burn the whole
                            # budget in reasoning_content with no answer
                            thought = msg.get("reasoning_content") or ""
                        else:
                            text = choice.get("text", "")
                            thought = ""
                        # stop string is stripped from the output by the
                        # server; re-append it ONLY when the delimiter stop
                        # fired on an on-task output (block started), so a
                        # model that never emitted the delimiter cannot
                        # score as if it had
                        if (choice.get("finish_reason") == "stop"
                                and "SECRET_BLOCK" in text):
                            text = text + "\n# END_BLOCK"
                        got = extract_span(text, bid)
                        from_reasoning = 0
                        # chat fallback: a thinking model that ruminates
                        # past the budget never emits content, but its
                        # REASONING often quotes the block — the state
                        # fidelity is measurable there even though answer
                        # delivery failed (flagged; delivery failure is
                        # reported separately as tax=no_answer)
                        if args.mode == "chat" and "SECRET_BLOCK" not in got \
                                and thought:
                            got_r = extract_span(thought, bid)
                            if "SECRET_BLOCK" in got_r:
                                got = got_r
                                from_reasoning = 1
                        sc = score(needle, got, bid)
                        if "SECRET_BLOCK" not in got:
                            sc["tax"] = "no_answer"
                        tim = r.get("timings", {})
                        rec = dict(
                            model=args.label, mode=args.mode,
                            ctx_target=ctx,
                            nlen_target=nlen, pos=pos, seed=seed,
                            bid=bid, prompt_tokens=ntok,
                            needle_tokens=nlen_tok,
                            usage_prompt=(r.get("usage", {})
                                          .get("prompt_tokens")),
                            completion_tokens=(r.get("usage", {})
                                               .get("completion_tokens")),
                            finish_reason=choice.get("finish_reason"),
                            max_tokens=mt,
                            thought_chars=len(thought),
                            from_reasoning=from_reasoning,
                            prompt_ms=tim.get("prompt_ms"),
                            predicted_ms=tim.get("predicted_ms"),
                            wall_s=round(time.time() - t0, 1),
                            out_len_chars=len(normalize(got)),
                            ref_len_chars=len(normalize(needle)), **sc)
                        fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        i += 1
                        if args.max_rows and i >= args.max_rows:
                            print(f"max-rows {args.max_rows} reached", flush=True)
                            raise SystemExit(0)
                        if i % 10 == 0 or i <= 5:
                            el = time.time() - t_start
                            print(json.dumps(dict(
                                rows=i, elapsed_min=round(el / 60, 1),
                                ctx=ctx, nlen=nlen, pos=pos, seed=seed,
                                exact=sc["exact"], sim=sc["edit_sim"])),
                                flush=True)
        print("battery complete", flush=True)
    finally:
        # tracked-PID teardown (own child only; never touch other procs —
        # in --attach mode there is no owned child to tear down)
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            print(f"server pid {proc.pid} stopped", flush=True)
        fout.close()


if __name__ == "__main__":
    main()
