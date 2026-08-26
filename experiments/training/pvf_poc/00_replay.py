#!/usr/bin/env python3
"""PVF POC step 0: replay GRPO-style groups from a GGUF checkpoint.

The GRPO runs (rl_smoke.py) never persisted per-rollout records — only
per-step means in rl_metrics.jsonl — so the critic POC regenerates faithful
groups: sample K=8 completions per prompt at temperature 1.0 (no stop
string, 192-token cap — exactly the rl_smoke generation regime; the SFT
model emits no EOS so completions run to the cap and parse_pred cuts at
the UPDATED marker) and score them with the VERBATIM rl_smoke reward
(exact + 0.2 * line_f1).

Prompts are the four RL families (rename/format/no_op/pipe) rendered from
scenarios_v1 with the exact assembler path (assemble_sft_v5.edit_row, the
suffix-convention render sft_v5+ mixtures use, incl. the no_op cursor
convention). Fidelity gate: a rendered row is kept only if its prompt
byte-matches a prompt in sft_v8/train.jsonl (RL-run-2's data) — the render
is then faithful by construction — and does not appear in the sft_v3
materialized holdout. Split is BY PACKAGE (80/20 train/val) so the critic
val packages are unseen (leakage control for the privileged arm).

Reference fields (region_old/region_new/note/kind) ride along per row:
the privileged-critic input in 01_train_value.py. For the propagation
families the reference region is NOT in the prompt (the event_diff in the
prompt is the triggering edit elsewhere — the observation, not the answer).

Output: /mnt/h/sepalith/datasets/pvf_poc_v1/replay_scenarios.jsonl, one
line per prompt: {pid, family, package, split, prompt, target, ref,
completions[K], rewards[K], exact[K]}. Reruns resume (pid-keyed).

Usage (.venv-sft; GPU must be free):
  python 00_replay.py --smoke            # 8 prompts, validates the pipeline
  python 00_replay.py                    # full ~1,500-prompt replay
"""
import argparse
import concurrent.futures as cf
import difflib
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "eval"))            # run_eval
sys.path.insert(0, str(HERE.parent.parent / "post-processing"))  # assembler
from run_eval import norm, parse_pred                             # noqa: E402
from assemble_sft_v5 import edit_row                              # noqa: E402

GGUF = Path("/home/m0hawk/Documents/Sepalith/experiments/models/"
            "sft_v7_minicpm5-Q8_0.gguf")
SERVER_BIN = Path("/home/m0hawk/Documents/Sepalith/experiments/bin/llama/"
                  "llama-b10453/llama-server")
SCEN_DIR = Path("/mnt/h/sepalith/datasets/scenarios_v1")
TRAIN_REF = Path("/mnt/h/sepalith/datasets/sft_v8/train.jsonl")
HOLDOUT_REF = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
OUT_DIR = Path("/mnt/h/sepalith/datasets/pvf_poc_v1")
PORT = 18097
K = 8                    # completions per prompt (PI grid tops at 16; 8 is
                         # the grid's midpoint and enough for EVAFUL groups)
TEMP = 1.0
MAX_TOKENS = 192         # rl_smoke cap; no stop string (see docstring)
QUOTAS = {"rename_propagation": 500, "format_propagation": 500,
          "no_op": 300, "pipe_rewrite": 200}
MAX_PROMPT_CHARS = 2400  # ~480-token proxy (build_dataset's cap); 01 re-checks
                         # with the real tokenizer
SEED = 3407


# --- verbatim rl_smoke reward path -------------------------------------------

def exact_reward(pred_lines, region_new_lines) -> float:
    """1.0 on exact match after rstrip-normalisation, else line-F1 (difflib).

    Verbatim copy of scenarios.exact_reward == rl_smoke.exact_reward."""
    p = [l.rstrip() for l in (pred_lines or [])]
    g = [l.rstrip() for l in (region_new_lines or [])]
    while p and p[-1] == "":
        p.pop()
    while g and g[-1] == "":
        g.pop()
    if p == g:
        return 1.0
    if not p or not g:
        return 0.0
    sm = difflib.SequenceMatcher(a=p, b=g, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    prec = matched / len(p)
    rec = matched / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


SHAPING = 0.2
UPDATED_MARK = ">>>>>>> UPDATED"


def gt_lines(target: str):
    """Normalized target region lines (rl_smoke.gt_lines, verbatim)."""
    body = target
    for suf in (f"\n{UPDATED_MARK}", UPDATED_MARK):
        if body.endswith(suf):
            body = body[: -len(suf)]
            break
    return norm(body.splitlines())


def reward_of(completion, target):
    pred = parse_pred("zeta2", completion)
    gt = gt_lines(target)
    ex = int(pred == gt)
    return ex, ex + SHAPING * exact_reward(pred, gt)


# --- prompt pool: render + fidelity-match against sft_v8 train ---------------

def build_pool():
    """Render all scenario rows of the 4 families; keep rows whose prompt
    is in the sft_v8 train split (byte-faithful to RL-run-2's data)."""
    train_prompts = {}
    for line in open(TRAIN_REF):
        r = json.loads(line)
        fam = r.get("family")
        if fam in QUOTAS:
            train_prompts.setdefault(fam, set()).add(r["prompt"])
    holdout = set()
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        if r.get("family") in QUOTAS:
            holdout.add(r["prompt"])

    pool, report = {}, {}
    for fam in QUOTAS:
        rows, seen = [], set()
        matched = rendered = 0
        for line in open(SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            if fam == "no_op":
                rr = edit_row(row, fam, row["package"], fd=0,
                              cursor_after=row.get("cursor_idx", 0))
            else:
                rr = edit_row(row, fam, row["package"])
            if rr is None:
                continue
            rendered += 1
            p = rr["prompt"]
            if p not in train_prompts.get(fam, ()):
                continue
            matched += 1
            if p in holdout or p in seen or len(p) > MAX_PROMPT_CHARS:
                continue
            seen.add(p)
            rows.append(dict(
                pid=hashlib.sha1(p.encode()).hexdigest()[:12], family=fam,
                package=row["package"], prompt=p, target=rr["target"],
                ref=dict(region_old=row.get("region_old") or [],
                         region_new=row.get("region_new") or [],
                         note=row.get("note", ""), kind=row.get("kind", ""))))
        pool[fam] = rows
        report[fam] = dict(rendered=rendered, matched_train=matched,
                           kept=len(rows))
    return pool, report


def draw_and_split(pool):
    """Seeded stratified draw per family, then an 80/20 train/val split BY
    PACKAGE (val prompts come from packages the critic never trains on)."""
    rng = random.Random(SEED)
    out = []
    for fam, rows in pool.items():
        pkgs = sorted({r["package"] for r in rows})
        rng.shuffle(pkgs)
        n_val = max(1, round(len(pkgs) * 0.2))
        val_pkgs = set(pkgs[:n_val])
        rng.shuffle(rows)
        for r in rows[: QUOTAS[fam]]:
            r["split"] = "val" if r["package"] in val_pkgs else "train"
            out.append(r)
    return out


# --- llama-server (tracked-PID lifecycle, GPU) --------------------------------

def port_open(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Server:
    """Same policy as eval_scenarios.Server: readiness = a real completion
    POST returning 200 (never /health); teardown signals ONLY the tracked
    child PID. GPU variant: -ngl 99, --parallel K so K slots decode the
    group concurrently."""

    def __init__(self, binary, model, port, log_path, parallel, ctx):
        self.binary = str(binary)
        self.model, self.port, self.parallel, self.ctx = \
            str(model), port, parallel, ctx
        self.log_path, self.proc = log_path, None

    def start(self, ready_timeout=1800):
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} in use; refusing to touch "
                               f"a server we did not start")
        cmd = [self.binary, "-m", self.model, "--port", str(self.port),
               "--host", "127.0.0.1", "-t", "8", "--parallel", str(self.parallel),
               "-c", str(self.ctx), "-ngl", "99"]
        log = open(self.log_path, "ab")
        log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server rc={self.proc.returncode}; "
                                   f"log tail:\n{self.log_tail()}")
            try:
                body = json.dumps({"prompt": "readiness", "max_tokens": 1,
                                   "temperature": 0}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError,
                    OSError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server not ready; "
                           f"log tail:\n{self.log_tail()}")

    def log_tail(self, n=1500):
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
        self.proc = None


def complete(port, prompt, seed):
    body = json.dumps({"prompt": prompt, "max_tokens": MAX_TOKENS,
                       "temperature": TEMP, "seed": seed,
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def replay(args, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "replay_scenarios.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line).get("pid"))
            except ValueError:
                pass
        print(f"resume: {len(done)} prompt(s) already replayed", flush=True)
    todo = [r for r in rows if r["pid"] not in done]

    server = Server(args.server_bin, args.model, PORT,
                    OUT_DIR / "llama-server-replay.log",
                    parallel=K, ctx=32768)
    server.start()
    n_groups = n_comp = 0
    t0 = time.time()
    try:
        with open(out_path, "a") as out, \
                cf.ThreadPoolExecutor(max_workers=K) as pool:
            for i, r in enumerate(todo):
                futures = [pool.submit(complete, PORT, r["prompt"],
                                       SEED + n_comp + j) for j in range(K)]
                comps = [f.result() for f in futures]
                scored = [reward_of(c, r["target"]) for c in comps]
                rec = {k: r[k] for k in
                       ("pid", "family", "package", "split", "prompt",
                        "target", "ref")}
                rec.update(completions=comps,
                           exact=[e for e, _ in scored],
                           rewards=[w for _, w in scored])
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_groups += 1
                n_comp += len(comps)
                if n_groups % 25 == 0 or n_groups == len(todo):
                    print(json.dumps(dict(
                        groups=n_groups, of=len(todo), completions=n_comp,
                        elapsed_s=round(time.time() - t0, 1),
                        last_reward_mean=round(
                            sum(rec["rewards"]) / len(rec["rewards"]), 3))),
                        flush=True)
    finally:
        server.stop()

    # replay report: zero-variance groups + degenerate-sampling check
    groups = [json.loads(l) for l in open(out_path)]
    zero_var = sum(1 for g in groups if len(set(g["rewards"])) == 1)
    identical = sum(1 for g in groups if len(set(g["completions"])) == 1)
    by_fam = {}
    for g in groups:
        d = by_fam.setdefault(g["family"], dict(n=0, zero_var=0,
                                                mean_reward=0.0))
        d["n"] += 1
        d["zero_var"] += int(len(set(g["rewards"])) == 1)
        d["mean_reward"] += sum(g["rewards"]) / len(g["rewards"])
    for d in by_fam.values():
        d["mean_reward"] = round(d["mean_reward"] / d["n"], 4)
    print(json.dumps(dict(total_groups=len(groups), zero_var_groups=zero_var,
                          all_identical_completions=identical,
                          by_family=by_fam), indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(GGUF))
    ap.add_argument("--server-bin",
                    default="/tmp/llama.cpp-cuda/build/bin/llama-server",
                    help="CUDA build preferred; the repo's b10453 binary is "
                         "CPU-only (-ngl 99 is silently ignored by it)")
    ap.add_argument("--smoke", action="store_true",
                    help="8 prompts (1/family x2), validates the pipeline")
    args = ap.parse_args()
    for p in (args.model, args.server_bin, TRAIN_REF, HOLDOUT_REF):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")

    pool, report = build_pool()
    print(json.dumps(dict(fidelity=report)), flush=True)
    for fam, r in report.items():
        if r["matched_train"] == 0:
            sys.exit(f"no rendered {fam} row matched sft_v8 train prompts — "
                     f"render-convention drift?")
    rows = draw_and_split(pool)
    if args.smoke:
        rows = (rows[:2] + rows[-2:] +
                [r for r in rows if r["family"] == "no_op"][:2] +
                [r for r in rows if r["family"] == "pipe_rewrite"][:2])
    print(json.dumps(dict(
        drawn=len(rows),
        splits={s: sum(1 for r in rows if r["split"] == s)
                for s in ("train", "val")},
        families={f: sum(1 for r in rows if r["family"] == f)
                  for f in QUOTAS})), flush=True)
    replay(args, rows)


if __name__ == "__main__":
    main()
