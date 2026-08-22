"""
Served span-completion eval for the FIM dose ladder (probe2-style, POC scale).

Scoring approach reused from experiments/eval/eval_fim.py: send the PSM
prompt, stop at <|end|>, line-normalize, exact match + line-F1
(difflib matched-block F1). Rows are the held-out astfim eval rows selected
in data_prep_ladder.py (prompt<=640 tok so prompt+generation stays inside
the 1024-token trained position range).

4 concurrent client threads against llama-server --parallel 4.

Usage: eval_fim_served.py --port 18107 --arm ladder_fim35 --rows .../fim_eval_rows.json \
                          --out .../fim_eval_{arm}.jsonl --summary .../fim_eval_summary.json
"""
import argparse, difflib, json, sys, threading, time, urllib.request

ROWS = None
OUT = None
RESULTS = None
LOCK = threading.Lock()
DONE = [0]


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


def complete(port, prompt, max_tokens, retries=3):
    # fixed-corpus prompt fields end with the terminator line "<|end|>\n",
    # but the TRAINING text is suffix + "\n" + span + "\n<|end|>" — strip the
    # terminator so the served context matches the trained layout (else the
    # model sees an end-of-row signal and drifts into next-document mode)
    if prompt.endswith("<|end|>\n"):
        prompt = prompt[:-len("<|end|>\n")]
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": ["<|end|>"],
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            return data["choices"][0]["text"], time.time() - t0, \
                data.get("usage", {}).get("completion_tokens")
        except Exception as ex:
            last = ex
            time.sleep(5 * (a + 1))
    raise last


def worker(wid, port, n_rows):
    i = wid
    while i < n_rows:
        row = ROWS[i]
        rec = dict(i=row["i"], arm=ARGS.arm, prompt_tok=row["prompt_tok"],
                   target_tok=row["target_tok"])
        try:
            out, dt, ctoks = complete(port, row["prompt"], row["max_tokens"])
            pred = norm(out.splitlines())
            while pred and not pred[0].strip():
                pred.pop(0)
            e, f1 = score(pred, norm(row["target"].splitlines()))
            rec.update(exact=e, line_f1=f1, latency_s=round(dt, 2),
                       completion_tokens=ctoks,
                       stopped_early=(ctoks is not None and ctoks < row["max_tokens"]))
        except Exception as ex:
            rec.update(error=str(ex)[:120])
        with LOCK:
            RESULTS.append(rec)
            DONE[0] += 1
            if DONE[0] % 25 == 0:
                print(f"  {DONE[0]}/{n_rows} rows", flush=True)
                with open(OUT, "w") as f:
                    for r in RESULTS:
                        f.write(json.dumps(r) + "\n")
        i += 4


def main():
    global ROWS, OUT, RESULTS, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18107)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rows", default="/tmp/poc_twin/ladder/fim_eval_rows.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ARGS = ap.parse_args()

    ROWS = json.load(open(ARGS.rows))
    OUT = ARGS.out
    RESULTS = []
    ths = [threading.Thread(target=worker, args=(w, ARGS.port, len(ROWS)))
           for w in range(ARGS.threads)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    with open(OUT, "w") as f:
        for r in RESULTS:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in RESULTS if "exact" in r]
    err = len(RESULTS) - len(ok)
    summ = dict(arm=ARGS.arm, n=len(ok), n_err=err,
                exact=round(sum(r["exact"] for r in ok) / max(1, len(ok)), 4),
                line_f1=round(sum(r["line_f1"] for r in ok) / max(1, len(ok)), 4),
                mean_latency_s=round(sum(r["latency_s"] for r in ok) / max(1, len(ok)), 2),
                wall_s=round(time.time() - t0, 1))
    print(json.dumps(summ), flush=True)
    if ARGS.summary:
        s = {}
        if os.path.exists(ARGS.summary):
            s = json.load(open(ARGS.summary))
        s[ARGS.arm] = summ
        with open(ARGS.summary, "w") as f:
            json.dump(s, f, indent=1)


if __name__ == "__main__":
    import os
    main()
