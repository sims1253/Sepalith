#!/usr/bin/env python3
"""Zed-style intent-assertion eval: plain-English behavioral tests, LLM-judged.

Serves a GGUF locally (llama-server, CPU), completes every case in
intent_suite_v1.jsonl rendered with the extension's suffix-completion
convention (extensions/vscode-sepalith/src/extension.ts buildPrompt), then has
glm-5.3 score each completion against the case's plain-English assertion
(0/1/2). Three calibration anchors run first: a hand-written satisfying
completion and a ground-truth rename completion must score 2, a hand-written
violating completion must score 0 (corrupted-twin rule).

Usage:
  run_intent_suite.py --model ../models/sft_v3_minicpm5-Q8_0.gguf [--port 18091]

Ops rules honored here (docs/research/2026-08-19-night-results.md):
  - CPU serving only (-t 8 -ngl 0), readiness = POST /v1/completions -> 200
    (NEVER /health), kill ONLY the tracked child PID (port-scoped fuser as a
    last resort if the child died but the port still answers).
  - ZAI_API_KEY must be in the environment (source ~/.zshrc first).
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_SERVER = str(HERE.parent / "bin/llama/llama-b10453/llama-server")
DEFAULT_SUITE = HERE / "intent_suite_v1.jsonl"

# --- extension parity (extension.ts) ----------------------------------------

# generation stops: the UPDATED terminator plus every prompt marker the model
# has been seen echoing
STOPS = [">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>",
         "<[fim-suffix]>", "<[fim-prefix]>"]

# marker lines the model sometimes echoes from the prompt back into its completion
MARKER_LINE = re.compile(
    r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>)\s*$")


def render_prompt(case):
    """extension.ts renderPrompt/buildPrompt: suffix leads, prefix above the
    region, typed partial + <|user_cursor|> in the region; derived rows keep
    their edit history exactly as the zeta2/SFT render places it."""
    inp = case["input"]
    parts = ["<[fim-suffix]>"] + inp["suffix_lines"]
    eh = inp.get("edit_history_lines") or []
    if eh:
        parts += ["<[fim-prefix]><filename>edit_history"] + eh + [""]
    parts += [f"<[fim-prefix]><filename>{inp['filename']}"] + inp["prefix_lines"]
    partial = inp["cursor_partial"]
    region = [partial + "<|user_cursor|>"] if partial else ["<|user_cursor|>"]
    parts += ["<<<<<<< CURRENT"] + region + ["=======", "<[fim-middle]>"]
    return "\n".join(parts)


def parse_prediction(text):
    """extension.ts parsePrediction: cut at >>>>>>>, drop cursor marker and
    marker-only lines, strip blank ends, cut degenerate repetition."""
    if ">>>>>>>" in text:
        text = text.split(">>>>>>>")[0]
    text = text.replace("<|user_cursor|>", "")
    lines = [l.replace("\r", "") for l in text.split("\n")]
    lines = [l for l in lines if not MARKER_LINE.match(l)]
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    for i in range(2, len(lines)):
        if lines[i] == lines[i - 1] == lines[i - 2]:
            lines = lines[:i]
            break
    return lines


# --- local llama-server ------------------------------------------------------

def post_completion(port, prompt, max_tokens, stop, timeout=600):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": stop, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def wait_ready(port, proc, timeout=600):
    """Readiness = a real POST /v1/completions returning 200. NEVER /health."""
    body = json.dumps({"prompt": "ping", "max_tokens": 1,
                       "temperature": 0, "stream": False}).encode()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited early (code {proc.returncode}); see its log")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    sys.exit("llama-server never became ready on port %d" % port)


def kill_server(proc, port):
    """Kill ONLY the tracked child. Port-scoped fuser is the documented last
    resort when the child is gone but the port still answers."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    time.sleep(1)
    try:
        body = json.dumps({"prompt": "ping", "max_tokens": 1}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
        still = True
    except Exception:
        still = False
    if still:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)


# --- glm-5.3 judge (pattern from judge_drafting.py, verbatim) -----------------

ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"

JUDGE_PROMPT = """You are grading an inline code-completion suggestion for an R editing assistant.

The user is editing {path}. Their recent edit history (a diff; may be absent):
{history}

Code above the cursor (tail):
{prefix}

Typed so far on the cursor line (may be empty):
{partial}

Code below the cursor (head):
{suffix}

The model's completion (what it proposes to insert at the cursor; may be empty):
{completion}

ASSERTION — a plain-English test of intent:
{assertion}

Rate how the completion satisfies the assertion:
2 = the assertion is satisfied
1 = partially satisfied: right kind of content but missing a required element of the assertion
0 = the assertion is violated; OR the completion is empty while the assertion expects new content; OR the content is unrelated marker/format garbage
(If the assertion explicitly asks for an empty or no-op completion, an empty completion is a 2.)

Respond ONLY with JSON: {{"score": <0|1|2>, "reason": "<short>"}}"""


def judge(case, completion, retries=3):
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        sys.exit("ZAI_API_KEY missing from environment (source ~/.zshrc first)")
    inp = case["input"]
    body = json.dumps({
        "model": "glm-5.3", "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            path=inp["filename"],
            history="\n".join(inp.get("edit_history_lines") or []) or "(none)",
            prefix="\n".join(inp["prefix_lines"][-25:])[-1500:],
            partial=inp["cursor_partial"],
            suffix="\n".join(inp["suffix_lines"][:10])[:800],
            completion=("\n".join(completion))[:1200] if completion else "(empty)",
            assertion=case["assertion"])}],
        "response_format": {"type": "json_object"},
        "max_tokens": 800, "temperature": 0,
    }).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
            j = json.loads(content)
            if j.get("score") in (0, 1, 2):
                return j
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    return {"score": None, "reason": "transport"}


# --- calibration anchors ------------------------------------------------------

def anchors(suites_by_id):
    """(name, case, completion, expected) — run through the same judge.
    cal_gt_rename uses a rename row whose ground truth mentions ONLY the new
    name (some authors' GTs legitimately keep the old name on the RHS, which
    would make the anchor ambiguous)."""
    good_sd = ["rm = TRUE)", "  list(mean = mean_value, sd = sd_value)", "}"]
    bad_brace = ["", "  extra <- length(x)", "  if (extra == 0) stop(\"empty input\")", "}"]
    # pick first rename case whose GT text contains no bare old-name token at all
    ren = None
    for c in suites_by_id.values():
        if c.get("family") != "rename_propagation":
            continue
        m = re.search(r"old name \((\w+)\)", c["assertion"])
        gt = "\n".join(c.get("gt_completion") or [])
        if m and not re.search(rf"\b{re.escape(m.group(1))}\b", gt):
            ren = c
            break
    ren = ren or next(c for c in suites_by_id.values()
                      if c.get("family") == "rename_propagation")
    return [
        ("cal_good_live", suites_by_id["live-sd-line"], good_sd, 2),
        ("cal_gt_rename", ren, ren["gt_completion"], 2),
        ("cal_bad_postbrace", suites_by_id["live-post-brace"], bad_brace, 0),
    ]


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .gguf")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--resume", action="store_true",
                    help="reuse case results already in the output file")
    args = ap.parse_args()

    if not os.environ.get("ZAI_API_KEY"):
        sys.exit("ZAI_API_KEY missing from environment (source ~/.zshrc first)")

    cases = [json.loads(l) for l in open(args.suite)]
    by_id = {c["id"]: c for c in cases}
    stem = Path(args.model).stem
    out_path = HERE / f"results_intent_{stem}.jsonl"
    log_path = f"/tmp/llama-server-intent-{stem}.log"

    done = {}
    if out_path.exists() and args.resume:
        for l in open(out_path):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "id" in r and "score" in r:
                done[r["id"]] = r

    server_log = open(log_path, "w")
    cal_fail = []
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", str(Path(args.model).resolve()), "--port", str(args.port),
         "-t", str(args.threads), "-ngl", "0", "--parallel", "1", "-c", str(args.ctx),
         "--host", "127.0.0.1"],
        stdout=server_log, stderr=subprocess.STDOUT)
    print(f"llama-server pid {proc.pid} on port {args.port} (log {log_path})", flush=True)
    try:
        wait_ready(args.port, proc)
        print("server ready (POST /v1/completions 200)", flush=True)

        mode = "a" if (args.resume and done) else "w"
        with open(out_path, mode) as fh:
            def emit(rec):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(json.dumps({k: rec[k] for k in rec if k != "completion"}), flush=True)

            # 1. calibration anchors (judge only — completions are fixed)
            for name, case, completion, expected in anchors(by_id):
                j = judge(case, completion)
                rec = dict(id=case["id"], family="anchor", anchor=name,
                           expected=expected, score=j.get("score"),
                           reason=(j.get("reason") or "")[:120])
                emit(rec)
                if j.get("score") != expected:
                    cal_fail.append(name)
            if cal_fail:
                print(f"*** CALIBRATION FAILED: {cal_fail} — aggregate below is untrustworthy ***",
                      flush=True)

            # 2. every case: complete locally, parse like the extension, judge
            for c in cases:
                if c["id"] in done and done[c["id"]].get("score") is not None:
                    continue
                try:
                    raw, dt = post_completion(args.port, render_prompt(c),
                                              args.max_tokens, STOPS)
                    pred = parse_prediction(raw)
                    j = judge(c, pred)
                except Exception as e:
                    pred, dt, j = [], 0.0, {"score": None, "reason": f"error: {e}"[:120]}
                rec = dict(id=c["id"], family=c.get("family"), source=c["source"],
                           score=j.get("score"), reason=(j.get("reason") or "")[:120],
                           completion=("\n".join(pred))[:600], empty=int(not pred),
                           latency_s=round(dt, 2))
                emit(rec)
                done[c["id"]] = rec
    finally:
        kill_server(proc, args.port)
        server_log.close()
        print(f"llama-server (pid {proc.pid}) stopped", flush=True)

    rows = [r for r in done.values() if r.get("family") != "anchor"]
    scores = [r["score"] for r in rows if r["score"] is not None]
    by_fam = {}
    for fam in sorted({r["family"] for r in rows}):
        ss = [r["score"] for r in rows if r["family"] == fam and r["score"] is not None]
        by_fam[fam] = dict(n=len(ss),
                           mean=round(statistics.mean(ss), 3) if ss else None,
                           pct2=round(sum(1 for s in ss if s == 2) / len(ss), 3) if ss else None)
    agg = dict(model=stem, suite=Path(args.suite).name, n=len(scores),
               mean=round(statistics.mean(scores), 3) if scores else None,
               pct2=round(sum(1 for s in scores if s == 2) / len(scores), 3) if scores else None,
               pct0=round(sum(1 for s in scores if s == 0) / len(scores), 3) if scores else None,
               calibration="ok" if not cal_fail else f"FAIL:{cal_fail}",
               by_family=by_fam)
    with open(out_path, "a") as fh:
        fh.write(json.dumps({"aggregate": agg}) + "\n")
    print(json.dumps({"aggregate": agg}, indent=1), flush=True)


if __name__ == "__main__":
    main()
