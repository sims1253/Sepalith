#!/usr/bin/env python3
"""Keystroke-to-suggestion latency simulation against a running llama-server.

Cold: first request at a given context size (empty KV cache).
Warm: repeated requests appending keystroke-sized deltas to the same prefix
      (prompt cache should reuse the shared prefix).

Usage: keystroke_sim.py --port 18081 --ctx 4096 [--keystrokes 10] [--gen 48]
Prints one JSON line per phase.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / ".cache" / "repos" / "r" / "data.table"


def post(port, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, (time.time() - t0) * 1000.0


def ntokens(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize",
                                 data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def build_r_context(target_tokens, port):
    files = sorted((REPO / "R").glob("*.R"))
    chunks, total = [], ""
    i = 0
    while ntokens(port, total) < target_tokens and files:
        f = files[i % len(files)]
        total += f.read_text(errors="ignore") + "\n\n"
        i += 1
        if i > 200:
            break
    # trim from the middle to keep natural head/tail
    while ntokens(port, total) > target_tokens * 1.05:
        cut = len(total) // 10
        mid = len(total) // 2
        total = total[: mid - cut] + total[mid + cut:]
    return total


KEYSTROKE_FRAGMENTS = [
    "  x = x[order(group)]\n",
    "  if (anyNA(x)) x = safena(x)\n",
    "  out = list(a = x, b = group)\n",
    "  attr(out, \"sorted\") = TRUE\n",
    "  class(out) = c(\"dt_result\", \"list\")\n",
    "  for (i in seq_along(x)) out[[i]] = x[[i]] * 2\n",
    "  on.exit(setDTthreads(threads))\n",
    "  ans = forderv(x, by, retGrp = TRUE)\n",
]

PROMPT_TAIL = "\n# the function below needs fixing\nnext_edit <- function(df, col) {\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--keystrokes", type=int, default=10)
    ap.add_argument("--gen", type=int, default=48)
    args = ap.parse_args()

    base = build_r_context(args.ctx, args.port)
    ctx_tokens = ntokens(port := args.port, base)

    # --- cold: full prompt, empty cache (server freshly started or cache evicted) ---
    prompt = base + PROMPT_TAIL
    data, wall_ms = post(port, {"prompt": prompt, "n_predict": args.gen,
                                "cache_prompt": True, "temperature": 0})
    cold = dict(ctx_tokens=ctx_tokens, gen_tokens=data.get("tokens_predicted", args.gen),
                prompt_eval_ms=round(data["timings"]["prompt_ms"]),
                gen_ms=round(data["timings"]["predicted_ms"]),
                wall_ms=round(wall_ms))

    # --- warm: keystroke-sized appends; prefix cache should serve all but the tail ---
    warm, cur = [], prompt
    for k in range(args.keystrokes):
        cur += KEYSTROKE_FRAGMENTS[k % len(KEYSTROKE_FRAGMENTS)]
        data, wall_ms = post(port, {"prompt": cur, "n_predict": args.gen,
                                    "cache_prompt": True, "temperature": 0})
        warm.append(dict(prompt_eval_ms=round(data["timings"]["prompt_ms"]),
                         gen_ms=round(data["timings"]["predicted_ms"]),
                         wall_ms=round(wall_ms)))
    ws = sorted(w["wall_ms"] for w in warm)
    warm_summary = dict(
        n=len(warm), mean_wall_ms=round(sum(ws) / len(ws)),
        p50_wall_ms=ws[len(ws) // 2], p95_wall_ms=ws[-1],
        mean_prompt_eval_ms=round(sum(w["prompt_eval_ms"] for w in warm) / len(warm)),
        mean_gen_ms=round(sum(w["gen_ms"] for w in warm) / len(warm)),
        prompt_eval_tokens_per_req=[w["prompt_eval_ms"] for w in warm][:3],
    )
    print(json.dumps({"ctx": args.ctx, "cold": cold, "warm": warm_summary}))


if __name__ == "__main__":
    main()
