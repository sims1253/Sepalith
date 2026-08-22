#!/usr/bin/env python3
"""Companion to landscape_intent_judged.py: adds the gemma-4 e2b (small-model
peer) arm to the intent judged comparison.

Same 30 deterministically-picked cases (pick_cases from
landscape_intent_judged), same blind agy gemini-3.7 judge and JUDGE_PROMPT,
same 600-char completion cap. gemma completions come from the externally-
owned server on 18106 (client-only; see landscape_gemma_scenarios.py) with
glm-parity treatment: the run_intent_suite render_prompt + FMT_INSTRUCTION
as one user message, chat completions, temperature 0, no stop, parsed with
the same parse_prediction. Writes results_intent_judged_gemma4e2b.jsonl;
the v7/glm53 arms are NOT re-run (they live in
results_intent_judged_v7_vs_glm53.jsonl).
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic-data" / "cases"))

from backends import AgyBackend                               # noqa: E402
from landscape_gemma_scenarios import (CUDA_BIN, FALLBACK_PORT,     # noqa: E402
                                       GEMMA_GGUF, PORT, _alive, chat)
from landscape_glm_scenarios import FMT_INSTRUCTION            # noqa: E402
from landscape_intent_judged import SEED, judge_with_agy, pick_cases  # noqa: E402
from run_intent_suite import parse_prediction, render_prompt   # noqa: E402

SUITE = HERE / "intent_suite_v1.jsonl"
OUT_PATH = HERE / "results_intent_judged_gemma4e2b.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    # resolve the gemma server: 18106 if its owner has it up, else our own
    port, owned = PORT, None
    if not _alive(port):
        from landscape_zeta_scenarios import Server
        port = FALLBACK_PORT
        owned = Server(CUDA_BIN, GEMMA_GGUF, port,
                       HERE / "llama-server-gemma4e2b.log")
        owned.start()

    cases = [json.loads(l) for l in open(SUITE)]
    picked = pick_cases(cases, args.n)

    done = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH):
            try:
                done.add(json.loads(l).get("id"))
            except ValueError:
                pass

    # blind order for the gemma judgments alone (same fixed seed family)
    todo = [c for c in picked if c["id"] not in done]
    random.Random(SEED + 1).shuffle(todo)

    agy = AgyBackend()
    rows = []
    try:
        # -- pass 1: collect completions (4-way concurrent; server is
        #    --parallel 4). Rows are written with score=null and judged in
        #    pass 2, so an agy quota outage never wastes completions.
        import concurrent.futures as cf

        def collect(c):
            try:
                raw, _ = chat(port, render_prompt(c) + "\n\n"
                              + FMT_INSTRUCTION)
                pred = parse_prediction(raw)
            except Exception as e:
                raw, pred = f"ERROR: {e}", []
            return c, pred, raw

        pending = []
        with cf.ThreadPoolExecutor(max_workers=4) as pool, \
                open(OUT_PATH, "a") as out:
            for c, pred, raw in pool.map(collect, todo):
                rec = dict(arm="gemma4e2b", id=c["id"],
                           family=c.get("family"), score=None,
                           reason="", n_lines=len(pred),
                           completion=("\n".join(pred))[:600],
                           raw=raw[:400], empty=int(not pred))
                rows.append(rec)
                pending.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps({k: rec[k] for k in
                                  ("arm", "id", "empty")}), flush=True)

        # -- pass 2: judge every collected row that still lacks a score ---
        for rec in pending:
            c = next(x for x in picked if x["id"] == rec["id"])
            lines = rec["completion"].split("\n") if rec["completion"] else []
            try:
                score, reason = judge_with_agy(agy, c,
                                               lines[:600] if lines else [])
            except Exception as e:   # quota/transport: leave for resume
                print(json.dumps(dict(judge_deferred=rec["id"],
                                      err=str(e)[:120])), flush=True)
                break
            rec["score"], rec["reason"] = score, reason
            print(json.dumps(dict(judged=rec["id"], score=score)),
                  flush=True)
        # rewrite the file with judged scores merged (append-style, resume
        # by id keeps the judged versions authoritative below)
        judged = {r["id"]: r for r in rows if r.get("score") is not None}
        if judged:
            with open(OUT_PATH, "a") as out:
                for r in judged.values():
                    out.write(json.dumps(r) + "\n")
    finally:
        if owned is not None:
            owned.stop()
            print("owned gemma server stopped", flush=True)

    # last occurrence per id wins (judged rows are re-appended after their
    # score-null collection rows)
    last = {}
    for l in open(OUT_PATH):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "id" in r:
            last[r["id"]] = r
    allrows = list(last.values())
    ss = [r["score"] for r in allrows if r.get("score") is not None]
    by_fam = {}
    for fam in sorted({r["family"] for r in allrows}):
        fs = [r["score"] for r in allrows
              if r["family"] == fam and r.get("score") is not None]
        by_fam[fam] = dict(n=len(fs), mean=round(statistics.mean(fs), 3)
                           if fs else None)
    agg = dict(judge="gemini-3.7-flash-low via agy (same blind protocol as "
               "the v7/glm53 run; gemma-only arm, shuffled seed SEED+1)",
               n=len(ss), mean=round(statistics.mean(ss), 3) if ss else None,
               pct2=round(sum(1 for s in ss if s == 2) / len(ss), 3) if ss else None,
               pct0=round(sum(1 for s in ss if s == 0) / len(ss), 3) if ss else None,
               empty=sum(1 for r in allrows if r.get("empty")),
               by_family=by_fam, agy_stats=agy.stats_summary())
    with open(OUT_PATH, "a") as fh:
        fh.write(json.dumps({"aggregate": agg}) + "\n")
    print(json.dumps({"aggregate": agg}, indent=1))


if __name__ == "__main__":
    main()
