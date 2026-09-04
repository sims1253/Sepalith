"""verdict_battery.py — the HELD-OUT verdict runner (plan §1.2).

Never D_harness: eval_scenarios' held-out selection (cap 150/family, the
materialized sft_v3 split authority) + the FULL eval_noop_fp case set
(build_cases(corpus_n=30, seed=7), all classes), one rollout per row at
temp 0 (the banked eval convention), scored under a candidate harness
config. Optional leg: the intent suite (44 rows, glm-judged) under the
config's gen knobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE.parent / "synthetic-data"))

from eval_scenarios import load_heldout, validator_verdict      # noqa: E402
from harness_config import (DEFAULT_CONFIG, DEFAULT_TEXTS,      # noqa: E402
                            validate_config, stops_for)
from render import render_scenario, build_scoped_prompt          # noqa: E402
from scorer import predict, parse_prediction, _p95               # noqa: E402
import methods                                                     # noqa: E402
import server as S                                                 # noqa: E402


def _shard_single(clients, items, render_fn):
    """Parallel singles across servers: items split by index parity, each
    server serves its shard; returns a list of rollouts aligned to items."""
    out = [None] * len(items)

    def run(ci):
        client = clients[ci]
        for j, item in enumerate(items):
            if j % len(clients) != ci:
                continue
            prompt, cfgstops, mt = render_fn(item)
            out[j] = client.single(prompt, cfgstops, mt)

    if len(clients) == 1:
        run(0)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(clients)) as ex:
            list(ex.map(run, range(len(clients))))
    return out


def verdict_scenarios(cfg, texts, clients) -> dict:
    examples, report = load_heldout(150)
    rows = []
    for fam, fam_rows in examples.items():
        for row in fam_rows:
            r = dict(row)
            r["id"] = hashlib.sha1(r["_prompt"].encode()).hexdigest()[:12]
            rows.append(r)
    stops, mt = stops_for(cfg), cfg["max_tokens"]

    def render(row):
        prompt, _ = render_scenario(row, cfg, texts)
        return prompt, stops, mt

    rolls = _shard_single(clients, rows, render)
    per_fam = {}
    lats = []
    for row, roll in zip(rows, rolls):
        pred = predict(roll["text"], row, cfg)
        ok, kind, reason = validator_verdict(row, pred)
        gt = [l.rstrip() for l in row["region_new"]]
        while gt and not gt[-1]:
            gt.pop()
        d = per_fam.setdefault(row["family"], dict(n=0, valid=0, exact=0, lats=[]))
        d["n"] += 1
        d["valid"] += int(ok)
        d["exact"] += int(pred == gt)
        d["lats"].append(roll["latency"])
        lats.append(roll["latency"])
    fams = {}
    for f, d in per_fam.items():
        fams[f] = dict(n=d["n"], valid_pass=round(d["valid"] / d["n"], 4),
                       exact=round(d["exact"] / d["n"], 4),
                       p95_latency_s=round(_p95(d["lats"]), 3))
    n_all = sum(d["n"] for d in per_fam.values())
    return dict(
        n_rows=n_all,
        valid_pass=round(sum(d["valid"] for d in per_fam.values()) / n_all, 4),
        exact_str=round(sum(d["exact"] for d in per_fam.values()) / n_all, 4),
        p95_latency_s=round(_p95(lats), 3),
        families=fams, selection=report)


def verdict_noop(cfg, texts, clients) -> dict:
    cases = methods.load_noop_cases(subset_n=None)  # FULL set
    stops, mt = stops_for(cfg), cfg["max_tokens"]

    def render(c):
        prompt, _ = build_scoped_prompt(c.lines, c.cursor_line, c.cursor_char,
                                        c.rel_path, cfg, texts)
        return prompt, stops, mt

    rolls = _shard_single(clients, cases, render)
    per_cls, scored, props, lats = {}, 0, 0, []
    for c, roll in zip(cases, rolls):
        proposed = len(parse_prediction(roll["text"], cfg)) > 0
        per_cls.setdefault(c.cls, [0, 0])
        per_cls[c.cls][1] += 1
        per_cls[c.cls][0] += int(proposed)
        if c.cls[0] in "acd":
            scored += 1
            props += int(proposed)
        lats.append(rolls[c]["latency"])
    return dict(scored_n=scored, proposal_rate=round(props / max(1, scored), 4),
                per_class={k: [v[0], v[1]] for k, v in sorted(per_cls.items())},
                p95_latency_s=round(_p95(lats), 3), n_cases=len(cases))


def verdict_intent(cfg, texts, clients) -> dict:
    """Intent suite leg (44 rows, glm-5.3 judged): run_intent_suite's render
    + judge with the candidate's cap/outline/text-slot render knobs and
    gen/parse knobs applied. Anchors run once per invocation for judge
    sanity; per-case scores are the arm's own."""
    from run_intent_suite import judge, anchors
    import proposer as PR
    PR._load_key_from_zshrc()
    cases = [json.loads(l) for l in
             open(HERE.parent / "eval" / "intent_suite_v1.jsonl")]
    by_id = {c["id"]: c for c in cases}
    cal = [(n, judge(c, comp), exp) for n, c, comp, exp in anchors(by_id)]
    cal_ok = all(j.get("score") == exp for _, j, exp in cal)
    stops, mt = stops_for(cfg), cfg["max_tokens"]
    scores = []
    for k, case in enumerate(cases):
        inp = case["input"]
        suffix = list(inp["suffix_lines"])
        prefix = list(inp["prefix_lines"])
        suffix_chars = sum(len(l) + 1 for l in suffix)
        kept, _ = R._cut_prefix_keep_tail(
            prefix, max(0, cfg["prefix_suffix_cap"] - suffix_chars))
        _pin, outline = R._scope_sections(prefix, len(prefix), cfg, texts)
        parts = ["<[fim-suffix]>"] + suffix
        eh = inp.get("edit_history_lines") or []
        if eh:
            parts += ["<[fim-prefix]><filename>edit_history"] + eh + [""]
        parts += [f"<[fim-prefix]><filename>{inp['filename']}"] + kept + outline
        if texts.get("checklist_line"):
            parts.append(texts["checklist_line"])
        if texts.get("instruction_line"):
            parts.append(texts["instruction_line"])
        partial = inp["cursor_partial"]
        region = [partial + "<|user_cursor|>"] if partial else ["<|user_cursor|>"]
        parts += ["<<<<<<< CURRENT"] + region + ["=======", "<[fim-middle]>"]
        prompt = "\n".join(parts)
        roll = clients[k % len(clients)].single(prompt, stops, mt)
        from scorer import parse_prediction
        pred = parse_prediction(roll["text"], cfg)
        j = judge(case, pred)
        scores.append(j.get("score"))
    valid = [s for s in scores if s is not None]
    return dict(n=len(scores), n_judged=len(valid),
                mean_score=round(sum(valid) / max(1, len(valid)), 4),
                frac_2=round(sum(1 for s in valid if s == 2) / max(1, len(valid)), 4),
                frac_0=round(sum(1 for s in valid if s == 0) / max(1, len(valid)), 4),
                anchors_ok=cal_ok)


def run_verdict(name, cfg, texts, clients, with_intent=False) -> dict:
    cfg = validate_config(cfg)
    t0 = time.time()
    scen = verdict_scenarios(cfg, texts, clients)
    noop = verdict_noop(cfg, texts, clients)
    out = dict(name=name, fp=None, cfg=cfg, texts=texts,
               scenarios=scen, noop=noop,
               wall_s=round(time.time() - t0))
    if with_intent:
        t1 = time.time()
        out["intent"] = verdict_intent(cfg, texts, clients)
        out["wall_s"] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=S.H1_PORT)
    ap.add_argument("--ports", default="",
                    help="comma-separated ports (overrides --port)")
    ap.add_argument("--model", default=str(
        HERE.parent / "models" / "sft_v7_minicpm5-Q8_0.gguf"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default="verdict.json")
    ap.add_argument("--with-intent", action="store_true",
                    help="include the glm-judged intent-suite leg")
    ap.add_argument("--intent-only", action="store_true",
                    help="append intent legs to an existing verdict file")
    args = ap.parse_args()

    results = Path(args.results)
    model_tag = Path(args.model).stem.replace("-Q8_0", "")
    ports = [int(p) for p in args.ports.split(",") if p.strip()] or [args.port]
    clients = [S.PairClient(p) for p in ports]
    for c in clients:
        c.load(results / f"cache-{model_tag}.jsonl")
    started = []
    for p in ports:
        if not S.port_open(p):
            s = S.Server(args.model, port=p,
                         log_path=results / f"llama-server-h1-{p}.log")
            s.start()
            started.append(s)
    path = results / args.out
    try:
        if args.intent_only:
            out = json.loads(path.read_text())
            for arm in out["arms"]:
                if arm.get("intent"):
                    continue
                t0 = time.time()
                arm["intent"] = verdict_intent(arm["cfg"], arm["texts"], clients)
                print(json.dumps(dict(arm=arm["name"],
                                      intent=arm["intent"],
                                      wall_s=round(time.time() - t0))), flush=True)
            path.write_text(json.dumps(out, indent=1))
        else:
            out = dict(model=model_tag, generated=time.strftime("%FT%T"), arms=[])
            out["arms"].append(run_verdict("default", DEFAULT_CONFIG, DEFAULT_TEXTS,
                                           clients, with_intent=args.with_intent))
            for arm in ("hill", "population", "gepa"):
                sp = results / arm / "state.json"
                if not sp.exists():
                    print(f"[skip] {arm}: no state", flush=True)
                    continue
                state = json.loads(sp.read_text())
                best = state["best_entry"]
                out["arms"].append(run_verdict(arm, best["cfg"], best["texts"],
                                               clients, with_intent=args.with_intent))
                print(json.dumps(dict(arm=arm, fp=best["fp"],
                                      valid=out["arms"][-1]["scenarios"]["valid_pass"])),
                      flush=True)
            path.write_text(json.dumps(out, indent=1))
    finally:
        S.dump_all(clients, results / f"cache-{model_tag}.jsonl")
        for s in started:
            s.stop()
    print(f"verdict -> {path}")


if __name__ == "__main__":
    main()
