#!/usr/bin/env python3
"""Score FIM midtraining checkpoints on the astfim package-held-out rows.

Each astfim row is {"text": prompt + target} in PSM markers; the prompt ends
at the cursor zone and the target is the missing span + <|end|>. We send the
prompt, stop at <|end|>, and score the predicted span (exact / line-F1).
"""
import argparse, difflib, json, time, urllib.request


def split_row(text):
    # format: <|context|>...\n<|history|>\n\n<|cursor|><|suffix|>\n<suffix lines>
    #         \n<|end|>\n<target span>\n<|end|>
    # first <|end|> terminates the prompt; the target lives between the two.
    mark = "\n<|end|>\n"
    i = text.find(mark)
    if i == -1:
        return None, None
    prompt = text[: i + 1]  # everything before the terminator line
    rest = text[i + len(mark):]
    j = rest.rfind("\n<|end|>")
    target = rest[: j] if j != -1 else rest
    return prompt, target.strip("\n")


def complete(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": ["<|end|>"], "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def norm(lines):
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def score(pred, gt):
    exact = int(pred == gt)
    sm = difflib.SequenceMatcher(a=pred, b=gt, autojunk=False)
    matched = sum(m.size for m in sm.get_matching_blocks())
    f1 = (2 * matched / (len(pred) + len(gt))) if (pred or gt) else 1.0
    return exact, round(f1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--examples", default="/mnt/h/sepalith/datasets/astfim_v1/eval.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=640)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.examples)][: args.limit]
    results = []
    for i, row in enumerate(rows):
        prompt, target = split_row(row["text"])
        if prompt is None:
            continue
        try:
            out, dt = complete(args.port, prompt, args.max_tokens)
            pred = norm(out.splitlines())
            while pred and not pred[0].strip():
                pred.pop(0)
            e, f1 = score(pred, norm(target.splitlines()))
        except Exception as ex:
            dt, e, f1, pred = 0.0, 0, 0.0, None
            results.append(dict(i=i, error=str(ex)[:100]))
            continue
        rec = dict(i=i, exact=e, line_f1=f1, latency_s=round(dt, 2),
                   pred=("\n".join(pred)[:400] if pred is not None else None))
        results.append(rec)
        print(json.dumps(rec), flush=True)

    n = len([r for r in results if "exact" in r])
    print(json.dumps(dict(n=n,
                          exact=sum(r["exact"] for r in results if "exact" in r) / max(n, 1),
                          line_f1=round(sum(r["line_f1"] for r in results if "line_f1" in r) / max(n, 1), 4))))


if __name__ == "__main__":
    main()
