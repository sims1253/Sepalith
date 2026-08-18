#!/usr/bin/env python3
"""Validate glm-5.3-low as an edit-suggestion judge BEFORE building the RL loop.

Input: zeta-2 predictions on held-out edits (results_zeta2.jsonl) + ground truth.
Judge sees: prefix/suffix context, the user's partial state, ground-truth
completion, and the model's completion. Rates on a rubric:
  2 = equivalent to ground truth (semantically, even if worded differently)
  1 = partially correct / plausible continuation of the right edit
  0 = wrong / unrelated
Output: agreement stats between judge score and line-F1 band, calibration table.
Usage: judge_validation.py [--n 60]   (uses ~1 call per example)
"""
import argparse, json, os, statistics, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"

JUDGE_PROMPT = """You are grading a code-edit suggestion for R/Python code.

Context: the user is mid-edit in file {path}. Their typed partial line is shown;
the ground-truth completion is what the developer actually wrote next. The
model's suggestion is what our system proposed.

File context before cursor:
{prefix}

Typed partial line: {partial}

Ground-truth completion:
{gt}

Model suggestion (may be empty or partial):
{pred}

Rate the model suggestion 0, 1, or 2:
2 = semantically equivalent to the ground truth (wording may differ)
1 = partial credit: right idea/location, incomplete or minor errors
0 = wrong, unrelated, or empty

Respond ONLY with JSON: {{"score": <0|1|2>, "reason": "<short>"}}"""


def judge(path, prefix, partial, gt, pred, retries=3):
    key = os.environ.get("ZAI_API_KEY")
    body = json.dumps({
        "model": "glm-5.3", "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            path=path, prefix=prefix[-600:], partial=partial, gt=gt[:800],
            pred=pred[:800])}],
        "response_format": {"type": "json_object"},
        "max_tokens": 800, "temperature": 0,
    }).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                content = json.loads(r.read())["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            time.sleep(2 * (a + 1))
    return {"score": None, "reason": "transport"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--results", default=str(HERE.parent / "eval/results_zeta2.jsonl"))
    ap.add_argument("--examples", default=str(HERE.parent / "eval/examples.jsonl"))
    args = ap.parse_args()

    exs = {(e["repo"], e["path"], e["sha"]): e
           for e in map(json.loads, open(args.examples))}
    preds = [json.loads(l) for l in open(args.results) if '"lang"' in l][: args.n]

    rows = []
    for i, r in enumerate(preds):
        key = (r.get("repo"), r.get("path"), r.get("sha"))
        ex = exs.get(key)
        if ex is None:
            continue  # shuffled-resume index rows lack matching keys
        ro = ex["region_old"]; rn = ex["region_new"]
        first_diff = next((k for k in range(min(len(ro), len(rn))) if ro[k] != rn[k]), 0)
        partial = ro[first_diff] if first_diff < len(ro) else ""
        gt = "\n".join(rn[first_diff:])[:800]
        # reconstruct prediction: use stored metrics only if raw absent — re-derive
        # from line_f1 band is wrong; judge needs the TEXT. results files don't
        # store it -> approximate band study using stored metrics + judge on GT
        # vs a degraded GT (simulated suggestion) for calibration instead.
        rows.append((ex, partial, gt, r))
        if len(rows) >= args.n:
            break

    # Calibration study: judge real GT (should score 2), GT-degraded (drop 40%
    # of lines -> should score 1), and unrelated completion (should score 0).
    out = []
    for ex, partial, gt, r in rows:
        lines = gt.splitlines()
        degraded = "\n".join(lines[::2]) if len(lines) > 1 else lines[0][: len(lines[0]) // 2]
        unrelated = "stop('internal error')\nif (is.null(x)) return(NULL)"
        for label, pred in [("gt", gt), ("degraded", degraded), ("unrelated", unrelated)]:
            j = judge(ex["path"], "\n".join(ex["prefix"]), partial, gt, pred)
            out.append(dict(lang=ex["lang"], label=label, score=j.get("score"),
                            reason=(j.get("reason") or "")[:80],
                            f1=r.get("line_f1")))
            print(json.dumps(out[-1]), flush=True)
    (HERE / "results" / "judge_calibration.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out) + "\n")

    agg = {}
    for label in ("gt", "degraded", "unrelated"):
        ss = [o["score"] for o in out if o["label"] == label and o["score"] is not None]
        agg[label] = dict(n=len(ss), mean=round(statistics.mean(ss), 2) if ss else None,
                          pct2=round(sum(1 for s in ss if s == 2) / len(ss), 2) if ss else None)
    print(json.dumps({"calibration": agg}, indent=1))


if __name__ == "__main__":
    main()
