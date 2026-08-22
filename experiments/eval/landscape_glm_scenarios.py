#!/usr/bin/env python3
"""Landscape leg: glm-5.3 (frontier generalist) ZERO-SHOT on our scenario task.

Same held-out rows, same render, same validators as eval_scenarios.py — the
only differences are the model (glm-5.3 via the zai chat API, ZaiBackend
contract in experiments/synthetic-data/cases/backends.py: same endpoint, same
ZAI_API_KEY auth, same UA/pacing/retry pattern) and one SHORT format
instruction appended to the prompt once, because glm has never seen our
zeta2 merge format:

  "Complete the edit after <[fim-middle]>: output the updated region lines,
   then end with the line >>>>>>> UPDATED; emit nothing else."

Row selection is IDENTICAL to eval_scenarios.load_heldout (first cap rows per
family in file order, re-rendered through assemble_sft_v2.edit_row); this
script takes the first n<=60 of those per family (API cost cap), so row ids
(sha1 of the zeta2 prompt, the eval_scenarios convention) join 1:1 against
results_scenarios_sft_v7_minicpm5.jsonl for the apples-to-apples gap.

Scoring is the eval_scenarios code path, imported, not reimplemented:
parse_pred("zeta2") normalization, validator_verdict (the EXACT
scenarios.validate_example call), exact line match, exact_reward.

Payload deviation from ZaiBackend (flagged in the aggregate): temperature 0
(greedy, matching the local evals) and NO response_format json_object (the
task is a raw completion; forcing JSON would fight the edit format). If glm
persistently fights the format, rerun with --oneshot to append a single
worked example to the instruction (the prompt change is recorded).

Usage (ZAI_API_KEY must be in env):
  python3 experiments/eval/landscape_glm_scenarios.py [--n 60] [--limit N]

Writes results_scenarios_glm53_zeroshot.jsonl next to this script (resume by
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
from eval_scenarios import FAMILIES, load_heldout, validator_verdict  # noqa: E402
from run_eval import parse_pred                                      # noqa: E402
import scenarios                                                     # noqa: E402

OUT_PATH = HERE / "results_scenarios_glm53_zeroshot.jsonl"

FMT_INSTRUCTION = (
    "You are a code-completion engine for a merge-style edit format. "
    "Complete the edit after <[fim-middle]>: output the updated region lines, "
    "then end with the line >>>>>>> UPDATED; emit nothing else — no prose, "
    "no code fences."
)

# used only with --oneshot (format-handicap fallback; flagged in aggregate)
ONESHOT_EXAMPLE = """
Example of the expected input->output convention (input ends with the lines
<<<<<<< CURRENT / the old region / ======= / <[fim-middle]>; the old region is
what you rewrite):

input (tail): ...
<<<<<<< CURRENT
    on.exit(file_delete(built_path), add = TRUE)
=======
<[fim-middle]>

correct output (only this, nothing else):
    on.exit(file_delete(built_path2), add = TRUE)
>>>>>>> UPDATED
"""


class GlmScenarioBackend(ZaiBackend):
    """ZaiBackend contract, adapted for a raw completion task: temperature 0,
    no response_format (json_object would fight the edit format), generous
    max_tokens."""

    name = "zai-glm53-zeroshot"

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
    ap.add_argument("--oneshot", action="store_true",
                    help="append a worked example to the instruction "
                         "(format-fight fallback; recorded in aggregate)")
    args = ap.parse_args()

    instruction = FMT_INSTRUCTION + (ONESHOT_EXAMPLE if args.oneshot else "")
    examples, report = load_heldout(150)  # EXACT eval_scenarios selection
    backend = GlmScenarioBackend()

    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                done.add(json.loads(line).get("id"))
            except ValueError:
                pass
        print(f"resume: {len(done)} row(s) already scored", flush=True)

    with open(OUT_PATH, "a") as out:
        for fam in FAMILIES:
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
                           model="glm-5.3-zeroshot", oneshot=int(args.oneshot))
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
    rows = [json.loads(l) for l in open(OUT_PATH)]
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
        model="glm-5.3-zeroshot", backend="zai chat completions (ZaiBackend "
        "contract; temperature 0, max_tokens 2048, NO response_format)",
        instruction=instruction[:250], oneshot=args.oneshot,
        render="assemble_sft_v2.edit_row (zeta2) + instruction appended once",
        validator="scenarios.validate_example",
        cap_per_family=args.n, smoke_limit=args.limit or None,
        selection={f: report[f] for f in FAMILIES},
        zai_stats=backend.stats_summary(),
        families=agg, total_rows_scored=len(rows),
    ), indent=1)))


if __name__ == "__main__":
    main()
