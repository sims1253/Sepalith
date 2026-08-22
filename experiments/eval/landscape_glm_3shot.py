#!/usr/bin/env python3
"""Few-shot CONTROL leg: glm-5.3 with a 3-SHOT prefix on the SAME 158 rows.

Prices the format handicap in the landscape benchmark
(docs/research/landscape-v7-vs-glm53.md). The zero-shot leg
(landscape_glm_scenarios.py -> results_scenarios_glm53_zeroshot.jsonl,
0.424/0.335 overall) gave glm-5.3 the zeta2 render + ONE instruction; ~15%
of its rows bled points on pure format handling. Hypothesis under test:
part of the gap is format/task unfamiliarity, fixable by examples. If
few-shot closes most of the gap, v7's edge was format; if not, it is task
training.

Everything is IDENTICAL to the zero-shot leg except the prompt suffix:
  - SAME rows: load_heldout(150), first n<=60 per family — asserted to
    join 1:1 on id against results_scenarios_glm53_zeroshot.jsonl before
    any API call;
  - SAME render (assemble_sft_v2.edit_row zeta2 prompt, untouched), SAME
    FMT_INSTRUCTION string, SAME parse (run_eval.parse_pred "zeta2"),
    SAME validators (eval_scenarios.validator_verdict -> the EXACT
    scenarios.validate_example), same exact/reward scoring;
  - SAME API path: zai chat endpoint, ZaiBackend contract (endpoint/auth/
    UA/pacing/retries), temperature 0, max_tokens 2048, thinking enabled
    effort low, no response_format.

The ONLY delta: a 3-shot block appended after the instruction — three
SHORT worked examples (full input render -> correct output with the
>>>>>>> UPDATED terminator): one rename_propagation, one pipe_rewrite,
one format_propagation row. All three are drawn from the TRAIN side of
the same family files (rows whose zeta2 render is NOT in the materialized
/mnt/h/sepalith/datasets/sft_v3/eval.jsonl split; the script re-derives
and asserts this at runtime, and asserts their ids are not among the 158
eval ids). Pinned by (family, package, path, region_old) so dataset drift
fails loudly. The block is kept under ~600 tokens (see the aggregate's
fewshot_chars / fewshot_tokens_est).

Usage (ZAI_API_KEY must be in env):
  python3 experiments/eval/landscape_glm_3shot.py [--n 60] [--limit N]
      [--families fam1,fam2] [--out PATH]

Writes results_scenarios_glm53_3shot.jsonl next to this script (resume by
id) and prints the per-family aggregate LAST.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                        # eval_scenarios
sys.path.insert(0, str(HERE.parent / "synthetic-data" / "cases"))    # backends
sys.path.insert(0, str(HERE.parent / "post-processing"))             # assemble_sft_v2
sys.path.insert(0, str(HERE.parent / "synthetic-data"))              # scenarios

from backends import ZaiBackend                                      # noqa: E402
from assemble_sft_v2 import edit_row                                 # noqa: E402
from eval_scenarios import (FAMILIES, HOLDOUT_REF, SCEN_DIR,         # noqa: E402
                            load_heldout, validator_verdict)
from landscape_glm_scenarios import FMT_INSTRUCTION                  # noqa: E402
from run_eval import parse_pred                                      # noqa: E402
import scenarios                                                     # noqa: E402

OUT_PATH = HERE / "results_scenarios_glm53_3shot.jsonl"
ZEROSHOT_PATH = HERE / "results_scenarios_glm53_zeroshot.jsonl"

# the three worked examples, pinned to TRAIN-side rows (held-out check is
# re-asserted at runtime): family -> (package, path, region_old)
FEWSHOT_PINS = {
    "rename_propagation": ("downlit", "R/packages.R",
                           ["    packages <- lapply(expr, extract_package_attach)"]),
    "pipe_rewrite": ("highcharter", "R/data-helpers.R",
                     ["    as.data.frame() %>%"]),
    "format_propagation": ("marquee", "R/aaa.R",
                           ["  72/25.4, # mm", "  72/72.27, # points",
                            "  12 * 72/72.27, # picas"]),
}
FEWSHOT_LABELS = {
    "rename_propagation":
        "the rename from the edit history propagates to the matching "
        "symbol in the CURRENT block",
    "pipe_rewrite":
        "the pipe change from the edit history propagates to the next "
        "occurrence",
    "format_propagation":
        "the formatting change from the edit history propagates to every "
        "line of the CURRENT block",
}


def build_fewshot():
    """(fewshot_block, provenance) from the pinned TRAIN-side rows.

    Train side = the row's zeta2 render is NOT in the materialized eval
    split; asserted, never assumed. Returns the three (input render ->
    target) pairs under one header, plus a closing line tying the examples
    back to the input at the top of the message."""
    ref = {f: set() for f in FAMILIES}
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        if r.get("family") in ref:
            ref[r["family"]].add(r["prompt"])

    parts, prov = [], {}
    for fam, (pkg, path, region_old) in FEWSHOT_PINS.items():
        got = None
        for line in open(SCEN_DIR / f"{fam}.jsonl"):
            row = json.loads(line)
            if (row["package"], row["path"], row["region_old"]) != (pkg, path, region_old):
                continue
            rr = edit_row(dict(row), fam, pkg)
            assert rr is not None, f"fewshot pin {fam}/{pkg} fails edit_row"
            assert rr["prompt"] not in ref[fam], \
                f"fewshot pin {fam}/{pkg} is HELD-OUT; pin drift"
            got = (row, rr)
            break
        assert got is not None, f"fewshot pin not found: {fam} {pkg} {path}"
        row, rr = got
        rid = hashlib.sha1(rr["prompt"].encode()).hexdigest()[:12]
        prov[fam] = dict(id=rid, package=pkg, path=path)
        parts.append(
            f"Example {len(parts) + 1} ({FEWSHOT_LABELS[fam]}):\n\n"
            f"input:\n{rr['prompt']}\n\noutput:\n{rr['target']}"
        )
    block = (
        "Three worked examples of this convention (input in the same "
        "format as above, ending at <[fim-middle]>; output = what a "
        "correct completion looks like):\n\n"
        + "\n\n".join(parts)
        + "\n\nNow complete the input at the top of this message exactly "
          "like these examples: the updated region lines only, then the "
          "line >>>>>>> UPDATED, nothing else."
    )
    return block, prov


class Glm3ShotBackend(ZaiBackend):
    """Identical payload contract to the zero-shot leg's
    GlmScenarioBackend: temperature 0, max_tokens 2048, thinking enabled
    effort low, NO response_format (json_object would fight the format)."""

    name = "zai-glm53-3shot"

    def _payload(self, prompt: str) -> dict:
        return {
            "model": "glm-5.3", "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048, "temperature": 0,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="rows per family cap")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only this many rows per family")
    ap.add_argument("--families", default="",
                    help="comma list to restrict families (smoke tests)")
    ap.add_argument("--out", default=str(OUT_PATH),
                    help="output jsonl (default results_scenarios_glm53_3shot)")
    args = ap.parse_args()
    out_path = Path(args.out)

    fewshot, prov = build_fewshot()
    instruction = FMT_INSTRUCTION + "\n\n" + fewshot
    fams = [f for f in FAMILIES if not args.families
            or f in args.families.split(",")]

    examples, report = load_heldout(150)  # EXACT eval_scenarios selection

    # JOIN GUARD: the rows we are about to score must be EXACTLY the 158
    # rows of the zero-shot leg (same ids), else this is not a control.
    zs_ids = {json.loads(l)["id"] for l in open(ZEROSHOT_PATH)}
    sel_ids = set()
    for fam in FAMILIES:
        for row in examples[fam][: args.n]:
            sel_ids.add(hashlib.sha1(row["_prompt"].encode()).hexdigest()[:12])
    assert sel_ids == zs_ids, (
        f"row-set drift vs {ZEROSHOT_PATH.name}: "
        f"{len(sel_ids)} selected vs {len(zs_ids)} zero-shot, "
        f"sym-diff {sorted(sel_ids ^ zs_ids)[:6]}")
    assert not ({p["id"] for p in prov.values()} & zs_ids), \
        "fewshot example id collides with an eval row"

    backend = Glm3ShotBackend()

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line).get("id"))
            except ValueError:
                pass
        print(f"resume: {len(done)} row(s) already scored", flush=True)

    with open(out_path, "a") as out:
        for fam in fams:
            rows = examples[fam][: args.n]
            if args.limit:
                rows = rows[: args.limit]
            for i, row in enumerate(rows):
                rid = hashlib.sha1(row["_prompt"].encode()).hexdigest()[:12]
                if rid in done:
                    continue
                prompt = row["_prompt"] + "\n\n" + instruction
                rec = dict(id=rid, family=fam, i=i, package=row["package"],
                           path=row["path"], note=row.get("note", ""),
                           model="glm-5.3-3shot", fewshot=3)
                t0 = time.time()
                try:
                    text = backend.complete(prompt)
                    pred = parse_pred("zeta2", text)  # same parse/norm
                    ok, kind, reason = validator_verdict(row, pred)
                    gt = [l.rstrip() for l in row["region_new"]]
                    while gt and not gt[-1]:
                        gt.pop()
                    rec.update(latency_s=round(time.time() - t0, 2),
                               n_pred_lines=len(pred),
                               pred="\n".join(pred)[:400],
                               raw=text[:600],
                               valid_pass=int(ok), fail_kind=kind,
                               valid_reason=reason,
                               exact=int(pred == gt),
                               reward=round(scenarios.exact_reward(
                                   pred, row["region_new"]), 4))
                except Exception as e:  # transport/API: scored 0
                    rec.update(latency_s=round(time.time() - t0, 2),
                               n_pred_lines=0, pred=None, raw=None,
                               valid_pass=0, fail_kind="error",
                               valid_reason=str(e)[:160], exact=0,
                               reward=0.0, error=str(e)[:120])
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps({k: rec[k] for k in
                                  ("id", "family", "i", "valid_pass",
                                   "fail_kind", "exact", "reward",
                                   "latency_s")}), flush=True)

    # ---- per-family aggregate over ALL rows in the file (incl. resumed) ----
    rows = [json.loads(l) for l in open(out_path)]
    agg = {}
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            agg[fam] = dict(n_scored=0)
            continue
        lat = sorted(r["latency_s"] for r in rs)

        def frac(k):
            return round(sum(1 for r in rs if r.get(k)) / len(rs), 4)
        agg[fam] = dict(
            n_scored=len(rs),
            validator_pass=frac("valid_pass"),
            exact=frac("exact"),
            mean_reward=round(sum(r["reward"] for r in rs) / len(rs), 4),
            fail_shape=round(sum(1 for r in rs if r.get("fail_kind") == "shape")
                             / len(rs), 4),
            fail_transform=round(sum(1 for r in rs if r.get("fail_kind")
                                     == "transform") / len(rs), 4),
            fail_error=round(sum(1 for r in rs if r.get("fail_kind") == "error")
                             / len(rs), 4),
            p50_latency_s=lat[len(lat) // 2],
            p95_latency_s=lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))],
        )
    print(json.dumps(dict(aggregate=dict(
        model="glm-5.3-3shot", backend="zai chat completions (ZaiBackend "
        "contract; temperature 0, max_tokens 2048, NO response_format)",
        control_for="results_scenarios_glm53_zeroshot.jsonl (same 158 ids, "
        "asserted before the run)",
        instruction=FMT_INSTRUCTION,
        fewshot=dict(
            n=3, families=sorted(FEWSHOT_PINS),
            source="TRAIN side of scenarios_v1 (render not in sft_v3 "
                   "eval.jsonl; asserted at runtime; ids disjoint from the "
                   "158 eval rows)",
            pins=prov, chars=len(fewshot),
            tokens_est=len(fewshot) // 3,  # ~3 chars/token on R code
            block=fewshot),
        render="assemble_sft_v2.edit_row (zeta2) + instruction + 3-shot "
               "block appended once",
        validator="scenarios.validate_example",
        cap_per_family=args.n, smoke_limit=args.limit or None,
        families_run=fams,
        selection={f: report[f] for f in FAMILIES},
        zai_stats=backend.stats_summary(),
        families=agg, total_rows_scored=len(rows),
    ), indent=1)))


if __name__ == "__main__":
    main()
