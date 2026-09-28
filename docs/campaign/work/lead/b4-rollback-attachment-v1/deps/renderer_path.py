#!/usr/bin/env python3
"""Run next-edit-suggestion eval against a llama-server instance.

Usage:
  run_eval.py --port 18080 --model zeta1 examples.jsonl  -> results JSON on stdout
  run_eval.py --port 18080 --model zeta2 --official      -> sanity check vs official samples

Renders prompts in Zeta1 (alpaca) or Zeta2 (SPM merge-marker) format exactly as
published by Zed Industries, sends temperature-0 completions, and scores
predicted editable regions against ground truth.
"""
import argparse
import difflib
import json
import time
import urllib.request

ZETA1_TMPL = (
    "### Instruction:\n"
    "You are a code completion assistant and your task is to analyze user edits and then "
    "rewrite an excerpt that the user provides, suggesting the appropriate edits within the "
    "excerpt, taking into account the cursor location.\n\n"
    "### User Edits:\n\n{}\n\n"
    "### User Excerpt:\n\n{}\n\n"
    "### Response:\n\n"
)

START1, END1, CURSOR1 = "<|editable_region_start|>", "<|editable_region_end|>", "<|user_cursor_is_here|>"
CURSOR2 = "<|user_cursor|>"


def with_cursor(lines, idx, marker):
    out = list(lines)
    if 0 <= idx < len(out):
        out[idx] = out[idx] + marker
    return out


def norm(lines):
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def render_zeta1(ex):
    inp = "\n".join(
        ["```" + ex["path"]]
        + ex["prefix"]
        + [START1]
        + with_cursor(ex["region_old"], ex["cursor_idx"], CURSOR1)
        + [END1]
        + ex["suffix"]
        + ["```"]
    )
    events = ex.get("event_diff") or ""
    return ZETA1_TMPL.format(events, inp)


def render_zeta2(ex):
    # edit_history: raw git-diff body, no fences (strip the zeta1-style wrapper)
    ev = ex.get("event_diff") or ""
    for tag in ("```diff\n", "```"):
        ev = ev.replace(tag, "")
    ev_lines = [l for l in ev.splitlines()]
    if ev_lines and ev_lines[0].startswith('User edited'):
        ev_lines = ev_lines[1:]
    while ev_lines and not ev_lines[0].strip():
        ev_lines.pop(0)
    parts = ["<[fim-suffix]>"] + ex["suffix"] + ["<[fim-prefix]><filename>edit_history"]
    if ev_lines:
        parts += ev_lines + [""]
    parts += [f"<filename>{ex['path']}"] + ex["prefix"] + ["<<<<<<< CURRENT"]
    parts += with_cursor(ex["region_old"], ex["cursor_idx"], CURSOR2)
    parts += ["=======", "<[fim-middle]>"]
    return "\n".join(parts)


def render_zeta2_1(ex):
    """Zeta 2.1: multi-region numbered markers instead of merge markers."""
    # same as zeta2 but region wrapped in <|marker_1|>/<|marker_2|>
    ev = ex.get("event_diff") or ""
    for tag in ("```diff\n", "```"):
        ev = ev.replace(tag, "")
    ev_lines = [l for l in ev.splitlines()]
    if ev_lines and ev_lines[0].startswith("User edited"):
        ev_lines = ev_lines[1:]
    while ev_lines and not ev_lines[0].strip():
        ev_lines.pop(0)
    parts = ["<[fim-suffix]>"] + ex["suffix"] + ["<[fim-prefix]><filename>edit_history"]
    if ev_lines:
        parts += ev_lines + [""]
    parts += [f"<filename>{ex['path']}"] + ex["prefix"] + ["<|marker_1|>"]
    parts += with_cursor(ex["region_old"], ex["cursor_idx"], CURSOR2)
    parts += ["<|marker_2|>", "<[fim-middle]>"]
    return "\n".join(parts)


RENDER = {"zeta1": render_zeta1, "zeta2": render_zeta2, "zeta2_1": render_zeta2_1}


def complete(port, prompt, max_tokens, stop):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": stop or None,
                       "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


def parse_pred(model, text):
    """Extract predicted region lines from raw model output."""
    if model == "zeta2_1":
        if "<|marker_1|>" in text:
            text = text.split("<|marker_1|>", 1)[1]
        if "<|marker_2|>" in text:
            text = text.split("<|marker_2|>", 1)[0]
        text = text.replace(CURSOR2, "").replace(CURSOR1, "")
        lines = norm(text.splitlines())
        while lines and not lines[0]:
            lines.pop(0)
        return lines
    if model == "zeta1":
        if START1 in text:
            text = text.split(START1, 1)[1]
        else:
            return None  # format failure
        if END1 in text:
            text = text.split(END1, 1)[0]
    else:
        if ">>>>>>>" in text:
            text = text.split(">>>>>>> UPDATED", 1)[0].split(">>>>>>>")[0]
        if text.startswith("```"):
            text = text[3:]
    text = text.replace(CURSOR1, "").replace(CURSOR2, "")
    return norm(text.splitlines())


def make_midtyping(ex, seed=1):
    """Eval-v1: cursor mid-line on a partial prefix of the first changed line.

    Returns a modified example whose region_old ends with the typed partial
    (cursor after it) and whose region_new is the COMPLETION SUFFIX (rest of
    that line + lines through the last changed line) — the copy-from-context
    baseline scores ~0 on this construction by design.
    """
    import random
    rng = random.Random(seed * 1000 + ex["sha"].__hash__() % 1000)
    ro, rn = ex["region_old"], ex["region_new"]
    i = 0
    while i < min(len(ro), len(rn)) and ro[i] == rn[i]:
        i += 1
    if i >= len(rn):
        i = max(0, len(rn) - 1)
    first = rn[i] if i < len(rn) else ""
    j = i
    last_changed = i
    while j < len(rn):
        if rn[j] not in ro:
            last_changed = j
        j += 1
    frac = rng.uniform(0.3, 0.6)
    cut = max(1, int(len(first) * frac)) if first else 0
    partial = first[:cut]
    new = dict(ex)
    new["region_old"] = ro[:i] + ([partial] if True else [])
    new["cursor_idx"] = len(new["region_old"]) - 1
    new["region_new"] = ([first[cut:]] if cut < len(first) or True else []) + rn[i + 1: last_changed + 1]
    # first completion line is the rest of the partial line (may be empty)
    if not new["region_new"]:
        new["region_new"] = [first[cut:]]
    return new


def score(pred, gt):
    if pred is None:
        return dict(exact=0, first_line=0, line_f1=0.0, empty=0)
    gt_n = norm(gt)
    exact = int(pred == gt_n)
    first = int(bool(pred) and bool(gt_n) and pred[0] == gt_n[0])
    sm = difflib.SequenceMatcher(a=pred, b=gt_n, autojunk=False)
    matched = sum(m.size for m in sm.get_matching_blocks())
    f1 = (2 * matched / (len(pred) + len(gt_n))) if (pred or gt_n) else 1.0
    return dict(exact=exact, first_line=first, line_f1=round(f1, 4), empty=int(not pred))


def official_check(port, model):
    """Sanity: reproduce official published input/output pairs."""
    checks = []
    if model == "zeta2":
        import urllib.request as u
        prompt = u.urlopen("https://huggingface.co/zed-industries/zeta-2/resolve/main/sample.prompt", timeout=60).read().decode()
        target = u.urlopen("https://huggingface.co/zed-industries/zeta-2/resolve/main/sample.output", timeout=60).read().decode()
        out, dt = complete(port, prompt, 400, None)
        pred = parse_pred(model, out)
        gt = norm(target.splitlines())
        checks.append(("zeta2-official", score(pred, gt), dt, out[:400]))
    else:
        row = json.load(open(__file__.rsplit("/", 1)[0] + "/reference/zeta1_row5.json"))
        prompt = ZETA1_TMPL.format(row["events"], row["input"])
        out, dt = complete(port, prompt, 500, None)
        gt_raw = row["output"]
        gt = norm(gt_raw[gt_raw.find(START1) + len(START1):].split(END1)[0].splitlines())
        checks.append(("zeta1-official", score(parse_pred(model, out), gt), dt, out[:400]))
    return checks


def load_results(path):
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "lang" in r and "exact" in r:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", choices=["zeta1", "zeta2", "zeta2_1"], required=True)
    ap.add_argument("--examples", default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--resume", default=None, help="prior results file: skip examples already scored")
    ap.add_argument("--variant", default="commit", choices=["commit", "midtyping"])
    ap.add_argument("--align", default="raw", choices=["raw", "suffix"],
                    help="suffix: realign whole-region outputs to the completion "
                         "after the typed partial (fair to zeta-style models)")
    args = ap.parse_args()

    if args.official:
        if args.model == "zeta2_1":
            print(json.dumps({"check": "zeta2_1-official", "note": "no official sample on HF card; skipping"}))
            return
        for name, sc, dt, raw in official_check(args.port, args.model):
            print(json.dumps({"check": name, **sc, "latency_s": round(dt, 1),
                              "raw_head": raw}, indent=1))
        return

    exs = [json.loads(l) for l in open(args.examples)]
    if args.variant == "midtyping":
        exs = [make_midtyping(e, seed=i + 1) for i, e in enumerate(exs)]
    if args.limit:
        exs = exs[: args.limit]
    if args.resume:
        done = {(r.get("repo"), r.get("path"), r.get("sha"))
                for r in load_results(args.resume)}
        before = len(exs)
        exs = [e for e in exs if (e["repo"], e["path"], e["sha"]) not in done]
        import random
        random.Random(42).shuffle(exs)  # interleave langs so truncation stays balanced
        print(f"resume: skipping {before - len(exs)} already-scored, {len(exs)} to go", flush=True)
    # everything after these markers is unscored (suffix / format tail) —
    # stopping there also prevents post-fence rambling from burning CPU-hours
    stops = {"zeta1": ["<|editable_region_end|>"],
             "zeta2": [">>>>>>> UPDATED"],
             "zeta2_1": ["<|marker_2|>"]}[args.model]
    results = []
    for i, ex in enumerate(exs):
        prompt = RENDER[args.model](ex)
        pred = None
        try:
            out, dt = complete(args.port, prompt, args.max_tokens, stops)
            pred = parse_pred(args.model, out)
            if args.align == "suffix" and pred is not None:
                partial = (ex.get("region_old") or [""])[-1].strip()
                if partial:
                    for k in range(len(pred) - 1, -1, -1):
                        line = pred[k].strip()
                        if line == partial or (partial in line and len(partial) > 2):
                            tail = ([line[len(partial):]] if line.startswith(partial) and len(line) > len(partial) else []) + pred[k + 1:]
                            while tail and not tail[0].strip():
                                tail.pop(0)
                            pred = tail
                            break
            sc = score(pred, ex["region_new"])
        except Exception as e:
            out, dt, sc = "", 0.0, dict(exact=0, first_line=0, line_f1=0.0, empty=1)
            sc["error"] = str(e)[:100]
        rec = dict(i=i, lang=ex["lang"], repo=ex["repo"], path=ex["path"],
                   sha=ex["sha"], is_test=ex["is_test"], latency_s=round(dt, 2),
                   pred=("\n".join(pred)[:600] if pred is not None else None), **sc)
        results.append(rec)
        print(json.dumps(rec), flush=True)

    agg = {}
    for lang in sorted({r["lang"] for r in results}):
        rs = [r for r in results if r["lang"] == lang]
        n = len(rs)
        agg[lang] = dict(
            n=n,
            exact=sum(r["exact"] for r in rs) / n,
            first_line=sum(r["first_line"] for r in rs) / n,
            line_f1=sum(r["line_f1"] for r in rs) / n,
            format_fail=sum(1 for r in rs if r.get("error") or r.get("empty")) / n,
            p50_latency_s=sorted(r["latency_s"] for r in rs)[n // 2],
        )
    print(json.dumps({"model": args.model, "variant": args.variant, "agg": agg}, indent=1))


if __name__ == "__main__":
    main()
