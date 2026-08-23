#!/usr/bin/env python3
"""Refinement prompt-set builder for RL-run-2 (the recursive validator-
feedback arm — user idea 2026-08-23: "the model gets feedback in the form
of ry/lsp/air/jarl on its proposal").

Source: the sft_v7 TRAIN split (never eval rows — battery contamination
control). The RL base IS v7, so v7 GGUF's failures are exactly on-policy
for the refinement family. Per sampled row:

  1. generate the completion with the v7 GGUF (CUDA llama-server),
  2. score it: exact vs the train target + the scenarios validator
     (eval_scenarios.validator_verdict — leak-free: validates the
     PREDICTED region structurally, never consults the target),
  3. on failure, emit a REFINEMENT row: the original zeta2 prompt with
     the CURRENT block holding the wrong attempt + one feedback comment:

       #! validator: <parse/shape/transform message>   (validator failed)
       #! feedback: dismissed                          (structurally OK
                                                        but not accepted —
                                                        the product's
       rejection signal)

     target = the ORIGINAL target (reward path unchanged in rl_smoke).

The prompt reuses only in-distribution vocabulary (existing zeta2 markers
+ a #! comment) — the model was never SFT-trained on refinement rows, so
RL is teaching a small format delta, not a new language.

Usage (uv env — tree_sitter_r available):
  uv run python experiments/training/build_refinement_set.py \
    --gguf experiments/models/sft_v7_minicpm5-Q8_0.gguf \
    --train /mnt/h/sepalith/datasets/sft_v7/train.jsonl \
    --out /mnt/h/sepalith/datasets/rl_refinement_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))           # run_eval helpers
sys.path.insert(0, str(HERE.parent / "synthetic-data"))  # scenarios validators
from run_eval import norm, parse_pred                    # noqa: E402
import eval_scenarios as ES                              # noqa: E402

CUDA_LLAMA = Path("/tmp/llamacpp-cuda-build/bin/llama-server")
UPDATED = ">>>>>>> UPDATED"
PORT = 18110

QUOTA = {          # sampled per family from the TRAIN split
    "rename_propagation": 450,
    "format_propagation": 450,
    "pipe_rewrite": 120,
    "no_op": 350,
}


def region_parses(pred_lines: list[str]) -> tuple[bool, str]:
    """Generic leak-free validator: does the proposed region parse as R?
    (train.jsonl rows carry prompts only — the family validators need the
    original scenario structure, so the wave-style parse gate is the
    honest generic diagnostic; target is never consulted.)"""
    import tree_sitter_r
    from tree_sitter import Language, Parser
    text = "\n".join(pred_lines)
    if not text.strip():
        return False, "empty region"
    lang = Language(tree_sitter_r.language())
    tree = Parser(lang).parse(text.encode())
    if tree.root_node.has_error:
        for n in ES.traverse(tree.root_node) if hasattr(ES, "traverse") else []:
            if n.type == "ERROR":
                return False, f"syntax error near line {n.start_point[0] + 1}"
        return False, "syntax error"
    return True, ""


def gt_lines(target: str):
    body = target
    for suf in (f"\n{UPDATED}", UPDATED):
        if body.endswith(suf):
            body = body[: -len(suf)]
            break
    return norm(body.splitlines())


def render_refinement(prompt: str, pred_lines: list[str],
                      feedback: str) -> str:
    """Insert the failed attempt + feedback into a zeta2 prompt. The
    original prompt ends with:
        ...<<<<<<< CURRENT\\n<region>\\n=======\\n<[fim-middle]>
    The refinement replaces the region block with the attempt and adds
    the feedback comment above the middle marker."""
    lines = prompt.split("\n")
    try:
        i_cur = lines.index("<<<<<<< CURRENT")
        i_mid = lines.index(">>>>>>> UPDATED") if ">>>>>>> UPDATED" in lines \
            else None
    except ValueError:
        return ""
    # find the ======= separator after CURRENT
    try:
        i_sep = lines.index("=======", i_cur)
    except ValueError:
        return ""
    fb = [f"#! {feedback}"]
    out = lines[:i_cur + 1] + pred_lines + lines[i_sep:i_sep + 1] + fb \
        + lines[i_sep + 1:]
    return "\n".join(out)


def complete(port: int, prompt: str, max_tokens: int = 192) -> str:
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(dict(prompt=prompt, max_tokens=max_tokens,
                             temperature=0.0)).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", default="experiments/models/sft_v7_minicpm5-Q8_0.gguf")
    ap.add_argument("--train", default="/mnt/h/sepalith/datasets/sft_v7/train.jsonl")
    ap.add_argument("--out", default="/mnt/h/sepalith/datasets/rl_refinement_v1.jsonl")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--max-prompt-chars", type=int, default=1500,
                    help="keep headroom for the attempt + feedback inside "
                         "rl_smoke's 480-token prompt cap")
    args = ap.parse_args()

    # 1. sample train rows per family (json parse: 'text' serializes first,
    # so a line-prefix scan would miss the family field entirely)
    pools = {f: [] for f in QUOTA}
    for line in open(args.train):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        f = r.get("family")
        if f in pools and len(r.get("prompt") or "") <= args.max_prompt_chars:
            pools[f].append(r)
    rng = random.Random(20260823)
    rows = []
    for f, pool in pools.items():
        rng.shuffle(pool)
        rows.extend(pool[: QUOTA[f]])
    print(f"sampled: { {f: len(pool[:QUOTA[f]]) for f, pool in pools.items()} }",
          flush=True)

    # 2. spawn the CUDA server on v7
    srv = subprocess.Popen(
        [str(CUDA_LLAMA), "-m", args.gguf, "--port", str(args.port),
         "--host", "127.0.0.1", "-c", "8192", "--parallel", "4",
         "-ub", "2048", "-t", "8", "-ngl", "99"],
        stdout=open("/tmp/llama-server-refine.log", "w"), stderr=subprocess.STDOUT)
    print(f"server pid {srv.pid} on {args.port}; waiting ready", flush=True)
    t0 = time.time()
    ready = False
    while time.time() - t0 < 600:
        try:
            complete(args.port, "x", 1)
            ready = True
            break
        except Exception:
            time.sleep(4)
    if not ready:
        srv.kill()
        sys.exit("server never became ready")
    print("server ready", flush=True)

    # 3. generate + score + build refinement rows
    out_rows, stats = [], {f: dict(n=0, fail=0, refined=0) for f in QUOTA}

    def work(r):
        f = r["family"]
        stats[f]["n"] += 1
        try:
            comp = complete(args.port, r["prompt"])
        except Exception:
            return None
        pred = parse_pred("zeta2", comp)
        gt = gt_lines(r["target"])
        if pred == gt:
            return None
        stats[f]["fail"] += 1
        if f == "no_op":
            feedback = "feedback: dismissed — no change was warranted here"
        else:
            ok, why = region_parses(pred)
            feedback = (f"validator: {why}" if not ok
                        else "feedback: dismissed — proposal not accepted")
        rp = render_refinement(r["prompt"], pred, feedback)
        if not rp:
            return None
        stats[f]["refined"] += 1
        return dict(prompt=rp, target=r["target"], family=f"refine_{f}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        for row in ex.map(work, rows):
            if row:
                out_rows.append(row)

    srv.terminate()
    with open(args.out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats=stats, wrote=len(out_rows), out=args.out)),
          flush=True)


if __name__ == "__main__":
    main()
