#!/usr/bin/env python3
"""Judge comment-drafting predictions with glm-5.3-low.

Line-F1 cannot grade free-text comments (the author's wording never matches).
This judge sees the code being commented, the author's actual comment (ground
truth), and the model's draft, and scores 0/1/2. Calibration anchors run on a
subsample first: the ground-truth comment itself must score 2, an unrelated
comment must score 0 (the corrupted-twin rule from the playbook).

Usage:
  judge_drafting.py --results results_drafting_v3.jsonl [--n 229] [--cal 20]
"""
import argparse, json, os, statistics, time, urllib.request
from pathlib import Path

ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
HERE = Path(__file__).resolve().parent
DRAFT_EVAL = Path("/tmp/drafting_eval.jsonl")

JUDGE_PROMPT = """You are grading a draft code comment for R code.

The code block being commented (from the file {path}):
{code}

The comment the developer actually wrote (ground truth):
{gt}

The model's draft comment (may be empty):
{pred}

Rate the model's draft 0, 1, or 2:
2 = describes the same thing the developer's comment describes (wording may differ)
1 = a plausible comment for this code but a different emphasis/meaning than the developer's
0 = wrong, describes something else, or empty

Respond ONLY with JSON: {{"score": <0|1|2>, "reason": "<short>"}}"""


def judge(path, code, gt, pred, retries=3):
    key = os.environ.get("ZAI_API_KEY")
    body = json.dumps({
        "model": "glm-5.3", "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            path=path, code=code[-1200:], gt=gt[:600], pred=(pred or "")[:600])}],
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
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all comment rows")
    ap.add_argument("--cal", type=int, default=20)
    args = ap.parse_args()

    ev = [json.loads(l) for l in open(DRAFT_EVAL)]
    targets = {i: e for i, e in enumerate(ev) if e["family"] == "comment_drafting"}
    preds = {}
    for l in open(HERE / args.results):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "i" in r and "exact" in r and not r.get("error"):
            preds[r["i"]] = r

    pairs = [(i, targets[i], preds[i]) for i in sorted(targets) if i in preds]
    if args.n:
        pairs = pairs[: args.n]
    print(f"judging {len(pairs)} comment rows + {args.cal} calibration anchors", flush=True)

    out_path = HERE / (Path(args.results).stem + "_judged.jsonl")
    with open(out_path, "w") as fh:
        # calibration anchors: GT comment (expect 2), unrelated (expect 0)
        for i, e, p in pairs[: args.cal]:
            gt = e["target"]
            code = "\n".join(e["prompt"].splitlines()[-12:])
            for label, pred in [("cal_gt", gt),
                                ("cal_unrelated", "# load required libraries")]:
                j = judge("cal.R", code, gt, pred)
                rec = dict(i=i, label=label, score=j.get("score"),
                           reason=(j.get("reason") or "")[:80])
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(json.dumps(rec), flush=True)
        for i, e, p in pairs:
            gt = e["target"]
            code = "\n".join(e["prompt"].splitlines()[-12:])
            j = judge("row.R", code, gt, p.get("pred") or "")
            rec = dict(i=i, label="pred", score=j.get("score"),
                       reason=(j.get("reason") or "")[:80], f1=p.get("line_f1"))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(json.dumps(rec), flush=True)

    rows = [json.loads(l) for l in open(out_path)]
    agg = {}
    for label in ("cal_gt", "cal_unrelated", "pred"):
        ss = [r["score"] for r in rows if r["label"] == label and r["score"] is not None]
        agg[label] = dict(n=len(ss), mean=round(statistics.mean(ss), 2) if ss else None,
                          pct2=round(sum(1 for s in ss if s == 2) / len(ss), 2) if ss else None,
                          pct0=round(sum(1 for s in ss if s == 0) / len(ss), 2) if ss else None)
    print(json.dumps({"judge": agg}, indent=1))


if __name__ == "__main__":
    main()
