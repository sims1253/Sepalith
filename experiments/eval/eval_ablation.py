#!/usr/bin/env python3
"""Whole-region eval for the conditioning ablation arms.

Input: prompt/target JSONL rows as produced by post-processing/format_sft_types.py
(types/plain/dropout variants are index-matched: row i is the same underlying
record in every file). Sends each prompt to llama-server, stops at the UPDATED
marker, and scores the predicted region against the target the same way
run_eval.py scores zeta2 outputs.

Usage:
  eval_ablation.py --port 18085 --limit 500 /mnt/.../sft_ablation/types/eval.jsonl
  (add --resume <out> to skip already-scored row indices)

Writes one JSON row per example; the final line is an aggregate block split by
has_types and kind — the dropout arm's "evaluated both ways" comes from the
has_types split of a single run.
"""
import argparse, difflib, json, time, urllib.request

CURSORS = ("<|user_cursor|>", "<|editable_region_start|>", "<|editable_region_end|>")


def complete(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": [">>>>>>> UPDATED"],
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def norm(lines):
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def parse_pred(text):
    if ">>>>>>>" in text:
        text = text.split(">>>>>>> UPDATED", 1)[0].split(">>>>>>>")[0]
    if text.startswith("```"):
        text = text[3:]
    for c in CURSORS:
        text = text.replace(c, "")
    lines = norm(text.splitlines())
    while lines and not lines[0]:
        lines.pop(0)
    return lines


def gt_lines(target):
    t = target.rsplit(">>>>>>> UPDATED", 1)[0]
    for c in CURSORS:
        t = t.replace(c, "")
    lines = norm(t.splitlines())
    while lines and not lines[0]:
        lines.pop(0)
    return lines


def score(pred, gt):
    if pred is None:
        return dict(exact=0, first_line=0, line_f1=0.0, empty=1)
    exact = int(pred == gt)
    first = int(bool(pred) and bool(gt) and pred[0] == gt[0])
    sm = difflib.SequenceMatcher(a=pred, b=gt, autojunk=False)
    matched = sum(m.size for m in sm.get_matching_blocks())
    f1 = (2 * matched / (len(pred) + len(gt))) if (pred or gt) else 1.0
    return dict(exact=exact, first_line=first, line_f1=round(f1, 4), empty=int(not pred))


def agg(rows):
    n = len(rows)
    if not n:
        return None
    return dict(n=n,
                exact=sum(r["exact"] for r in rows) / n,
                first_line=sum(r["first_line"] for r in rows) / n,
                line_f1=round(sum(r["line_f1"] for r in rows) / n, 4),
                format_fail=sum(1 for r in rows if r.get("error") or r.get("empty")) / n,
                p50_latency_s=sorted(r["latency_s"] for r in rows)[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    exs = [json.loads(l) for l in open(args.examples)]
    if args.limit:
        exs = exs[: args.limit]
    done = set()
    if args.resume:
        for l in open(args.resume):
            try:
                done.add(json.loads(l).get("i"))
            except Exception:
                pass
        print(f"resume: skipping {len(done)} already-scored", flush=True)

    results = []
    for i, ex in enumerate(exs):
        if i in done:
            continue
        pred, sc = None, dict(exact=0, first_line=0, line_f1=0.0, empty=1)
        try:
            out, dt = complete(args.port, ex["prompt"], args.max_tokens)
            pred = parse_pred(out)
            sc = score(pred, gt_lines(ex["target"]))
        except Exception as e:
            dt, sc = 0.0, dict(exact=0, first_line=0, line_f1=0.0, empty=1)
            sc["error"] = str(e)[:100]
        rec = dict(i=i, package=ex.get("package") or ex.get("package_or_repo"),
                   kind=ex.get("kind"),
                   has_types=bool(ex.get("has_types")), latency_s=round(dt, 2),
                   pred=("\n".join(pred)[:600] if pred is not None else None), **sc)
        results.append(rec)
        print(json.dumps(rec), flush=True)

    out = dict(
        variant="ablation-whole-region",
        n_scored=len(results),
        overall=agg(results),
        by_has_types={str(k): agg([r for r in results if r["has_types"] == bool(int(k))])
                      for k in ("1", "0")},
        by_kind={k: agg([r for r in results if r["kind"] == k])
                 for k in sorted({r["kind"] for r in results})},
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
