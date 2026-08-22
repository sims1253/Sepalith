#!/usr/bin/env python3
"""Landscape leg: intent-style judged comparison, v7 vs glm-5.3, BLIND.

30 mixed-family cases from intent_suite_v1.jsonl (deterministic round-robin
over families, file order within each family). Each model's completion for
the SAME case is scored 0/1/2 against the case's plain-English assertion by
gemini-3.7-flash-low via the agy CLI (AgyBackend contract: --print
--new-project, prompt through the --prompt flag ONLY) — the calibrated
unbiased judge (docs/research/judge-calibration-gemini-opus.md: gemini-3.7
passed glm-5.3's three-gate protocol at/above glm's own baseline, agrees
78% exactly / 100% within-one-band with NO directional bias; glm-5.3 itself
is a contestant here, so it cannot judge).

  v7 arm   completions come FREE from the battery's
           results_intent_sft_v7_minicpm5*.jsonl per-case rows (stored
           parsed completions, capped at 600 chars — glm's completions are
           capped identically for parity). Never re-served.
  glm arm  same render_prompt (extension.ts convention, imported from
           run_intent_suite) + the SAME short format instruction used in
           the scenario leg, one zai chat call per case (temperature 0,
           ZaiBackend contract, no response_format), parsed with the SAME
           parse_prediction.

Blindness: all judgments (3 calibration anchors + 30 v7 + 30 glm) are
shuffled with a fixed seed and judged one completion at a time — the judge
prompt (run_intent_suite.JUDGE_PROMPT, verbatim) never names a model.
Anchors: the suite's own (satisfying completion -> 2, GT rename -> 2,
violating completion -> 0); a mismatch marks the aggregate untrustworthy.

Writes results_intent_judged_v7_vs_glm53.jsonl next to this script: one
judgment row per (blind job), then per-case join rows, aggregate LAST.
"""
import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic-data" / "cases"))

from backends import AgyBackend, extract_json_object            # noqa: E402
from landscape_glm_scenarios import FMT_INSTRUCTION, GlmScenarioBackend  # noqa: E402
from run_intent_suite import (anchors, parse_prediction,         # noqa: E402
                              render_prompt)

SUITE = HERE / "intent_suite_v1.jsonl"
OUT_PATH = HERE / "results_intent_judged_v7_vs_glm53.jsonl"
N_CASES = 30
SEED = 13


def pick_cases(cases, n=N_CASES):
    """Round-robin over sorted families, file order within each family,
    until n cases (deterministic)."""
    by_fam = {}
    for c in cases:
        by_fam.setdefault(c.get("family", "?"), []).append(c)
    fams = sorted(by_fam)
    picked, cycle = [], 0
    while len(picked) < n:
        advanced = False
        for f in fams:
            if len(picked) >= n:
                break
            pool = by_fam[f]
            if cycle < len(pool):
                picked.append(pool[cycle])
                advanced = True
        if not advanced:
            break
        cycle += 1
    return picked


def judge_with_agy(agy, case, completion_lines):
    prompt = build_judge_prompt(case, completion_lines)
    text = agy.complete(prompt)          # --prompt flag, --new-project
    obj = extract_json_object(text)
    if obj and obj.get("score") in (0, 1, 2):
        return obj["score"], str(obj.get("reason", ""))[:160]
    return None, (text or "")[:160]


def build_judge_prompt(case, completion_lines):
    """run_intent_suite.JUDGE_PROMPT verbatim — same rubric, same
    truncations, no model names (blind)."""
    from run_intent_suite import JUDGE_PROMPT
    inp = case["input"]
    return JUDGE_PROMPT.format(
        path=inp["filename"],
        history="\n".join(inp.get("edit_history_lines") or []) or "(none)",
        prefix="\n".join(inp["prefix_lines"][-25:])[-1500:],
        partial=inp["cursor_partial"],
        suffix="\n".join(inp["suffix_lines"][:10])[:800],
        completion=("\n".join(completion_lines))[:1200]
        if completion_lines else "(empty)",
        assertion=case["assertion"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_CASES)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: fewer cases")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(SUITE)]
    by_id = {c["id"]: c for c in cases}
    picked = pick_cases(cases, args.limit or args.n)
    print(f"picked {len(picked)} cases: "
          f"{ {f: sum(1 for c in picked if c.get('family')==f) for f in sorted({c.get('family') for c in picked})} }",
          flush=True)

    # -- v7 completions: battery rows if the chain got that far, else serve
    #    the 30 cases ourselves on an OWN GPU server (port 18106, CUDA
    #    build; the chain's CPU server on 18103 is never touched). Same
    #    render/parse/temperature as run_intent_suite. ----------------------
    v7_files = sorted(glob.glob(str(HERE / "results_intent_sft_v7_minicpm5*.jsonl")))
    v7_rows = {}
    for p in v7_files:
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if "id" in r and "completion" in r:
                v7_rows[r["id"]] = r
    missing = [c["id"] for c in picked if c["id"] not in v7_rows]
    v7_source = "battery"
    if missing:
        v7_source = f"served-here({len(missing)} cases; battery file absent " \
                    "or incomplete — chain phase 5 pending)"
        print(f"v7 arm: serving {len(missing)} cases on own GPU server "
              f"(battery intent rows unavailable)", flush=True)
        from landscape_zeta_scenarios import Server
        from run_intent_suite import STOPS
        srv = Server(
            "/tmp/llama.cpp-b10453/build-cuda/bin/llama-server",
            str(HERE.parent / "models" / "sft_v7_minicpm5-Q8_0.gguf"),
            18110, HERE / "llama-server-v7-intent30.log")
        srv.start()
        try:
            import urllib.request as _u
            for cid in missing:
                c = by_id[cid]
                body = json.dumps({"prompt": render_prompt(c),
                                   "max_tokens": 320, "temperature": 0,
                                   "stop": STOPS,
                                   "stream": False}).encode()
                req = _u.Request(
                    "http://127.0.0.1:18110/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                try:
                    with _u.urlopen(req, timeout=300) as r:
                        raw = json.loads(r.read())["choices"][0]["text"]
                    pred = parse_prediction(raw)
                except Exception as e:
                    raw, pred = f"ERROR: {e}", []
                v7_rows[cid] = dict(id=cid, completion=("\n".join(pred))[:600],
                                    empty=int(not pred), raw=raw[:400])
                print(json.dumps(dict(v7_case=cid, empty=int(not pred))),
                      flush=True)
        finally:
            srv.stop()
            print("v7 intent server stopped", flush=True)

    # -- glm completions (one zai call per case) -----------------------------
    glm = GlmScenarioBackend()
    glm_out = {}
    for c in picked:
        prompt = render_prompt(c) + "\n\n" + FMT_INSTRUCTION
        try:
            raw = glm.complete(prompt)
            pred = parse_prediction(raw)   # extension parse, same as suite
        except Exception as e:
            raw, pred = f"ERROR: {e}", []
        glm_out[c["id"]] = dict(raw=raw[:600],
                                completion=("\n".join(pred))[:600],
                                empty=int(not pred))
        print(json.dumps(dict(glm_case=c["id"], empty=int(not pred),
                              n_lines=len(pred))), flush=True)

    # -- blind judgment set ---------------------------------------------------
    jobs = []   # (arm, case, completion_lines)
    for name, case, completion, expected in anchors(by_id):
        jobs.append((f"anchor:{name}", case, completion, expected))
    for c in picked:
        v7c = (v7_rows[c["id"]].get("completion") or "").split("\n") \
            if v7_rows[c["id"]].get("completion") else []
        jobs.append(("v7", c, v7c, None))
        gc = glm_out[c["id"]]["completion"].split("\n") \
            if glm_out[c["id"]]["completion"] else []
        jobs.append(("glm53", c, gc, None))
    random.Random(SEED).shuffle(jobs)     # blind order, deterministic

    agy = AgyBackend()
    results = []
    with open(OUT_PATH, "w") as out:
        for arm, case, lines, expected in jobs:
            # parity: judge sees at most 600 chars of completion for both arms
            score, reason = judge_with_agy(agy, case, lines[:600] if lines else [])
            rec = dict(arm=arm, id=case["id"], family=case.get("family"),
                       score=score, reason=reason,
                       expected=expected, n_lines=len(lines),
                       completion=("\n".join(lines))[:600])
            results.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps({k: rec[k] for k in
                              ("arm", "id", "score", "expected")}), flush=True)

        # -- per-case join rows ----------------------------------------------
        cal = {r["arm"]: r["score"] for r in results
               if r["arm"].startswith("anchor:")}
        cal_ok = all(
            cal.get(f"anchor:{n}") == e
            for n, _, _, e in anchors(by_id))
        per_case = []
        for c in picked:
            v7r = next(r for r in results if r["arm"] == "v7" and r["id"] == c["id"])
            gr = next(r for r in results if r["arm"] == "glm53" and r["id"] == c["id"])
            rec = dict(id=c["id"], family=c.get("family"),
                       v7_score=v7r["score"], glm53_score=gr["score"],
                       v7_completion=v7r["completion"][:300],
                       glm53_completion=gr["completion"][:300])
            per_case.append(rec)
            out.write(json.dumps(rec) + "\n")

        def arm_stats(arm):
            ss = [r["score"] for r in results
                  if r["arm"] == arm and r["score"] is not None]
            return dict(n=len(ss),
                        mean=round(statistics.mean(ss), 3) if ss else None,
                        pct2=round(sum(1 for s in ss if s == 2) / len(ss), 3)
                        if ss else None,
                        pct0=round(sum(1 for s in ss if s == 0) / len(ss), 3)
                        if ss else None)

        by_fam = {}
        for fam in sorted({c.get("family") for c in picked}):
            entry = {}
            for arm in ("v7", "glm53"):
                ss = [r["score"] for r in results
                      if r["arm"] == arm and r["family"] == fam
                      and r["score"] is not None]
                entry[arm] = dict(n=len(ss),
                                  mean=round(statistics.mean(ss), 3)
                                  if ss else None)
            by_fam[fam] = entry
        agg = dict(judge="gemini-3.7-flash-low via agy (blind, shuffled "
                   f"seed {SEED})", calibration="ok" if cal_ok else
                   f"FAIL:{cal}", anchors=cal, v7_source=v7_source,
                   v7=arm_stats("v7"), glm53=arm_stats("glm53"),
                   by_family=by_fam,
                   glm_stats=glm.stats_summary(), agy_stats=agy.stats_summary())
        out.write(json.dumps({"aggregate": agg}) + "\n")
    print(json.dumps({"aggregate": agg}, indent=1), flush=True)


if __name__ == "__main__":
    main()
