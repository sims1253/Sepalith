#!/usr/bin/env python3
"""T0 zero-std group census (docs/research/2026-08-28-slime-miles-adoption-plan.md §B).

Pre-registered decision rule for the DAPO-style zero-std group filter:
  fraction of GRPO groups with reward std < 1e-6
    < 5%   -> drop the filter idea
    > 20%  -> T1 A/B is mandatory before any production RL phase
    else   -> T1 only if a GPU window is otherwise idle

Data: the pvf POC replay (1,500 prompts x K=8 completions, sft_v7 GGUF at
the rl_smoke generation regime — the t=0 policy of the RL line), scored
with the verbatim exact + 0.2*line_f1 reward at replay time (rewards[K]
field). Read-only on the parked pvf artifacts.

Secondary context: the rl_grpo_v*/rl_metrics.jsonl exact_no_op series —
the replay measures the t=0 degenerate rate; the live runs' no_op ascent
is the same mechanism arriving mid/late training.

CPU-only, no GPU claim. Output: one JSON blob to stdout.
"""
import json
import statistics
import sys
from collections import defaultdict

REPLAY = "/mnt/h/sepalith/datasets/pvf_poc_v1/replay_scenarios.jsonl"
RUNS = {
    "v1": "/mnt/h/sepalith/runs/rl_grpo_v1/rl_metrics.jsonl",
    "v4_tether": "/mnt/h/sepalith/runs/rl_grpo_v4_tether/rl_metrics.jsonl",
    "v5_loo_unnorm": "/mnt/h/sepalith/runs/rl_grpo_v5_loo_unnorm/rl_metrics.jsonl",
}
EPS = 1e-6  # the pre-registered degeneracy threshold (slime uses the same)
GROUPS_PER_STEP = 4  # metrics n=32 at K=8


def no_op_series(path, tail=40):
    """Per-40-step-window mean of exact_no_op, or None if the run lacks it."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "exact_no_op" in r and r.get("n_no_op")]
    if not rows:
        return None
    out = []
    for i in range(0, len(rows), tail):
        w = rows[i:i + tail]
        n = sum(r["n_no_op"] for r in w)
        v = sum(r["exact_no_op"] * r["n_no_op"] for r in w)
        if n:
            out.append(round(v / n, 3))
    return out


def main():
    groups = [json.loads(l) for l in open(REPLAY) if l.strip()]
    n_rewards = {len(g["rewards"]) for g in groups}
    assert n_rewards == {8}, f"expected K=8 everywhere, got {n_rewards}"

    deg = defaultdict(int)      # family -> count of zero-std groups
    tot = defaultdict(int)      # family -> count of groups
    by_split = defaultdict(lambda: [0, 0])  # split -> [deg, tot]
    distinct_hist = defaultdict(int)        # n distinct reward values -> groups
    mass_nondeg = 0.0
    mass_all = 0.0

    for g in groups:
        r = g["rewards"]
        sd = statistics.pstdev(r)
        is_deg = sd < EPS
        tot[g["family"]] += 1
        by_split[g["split"]][1] += 1
        distinct_hist[len(set(r))] += 1
        mass_all += sum(r)
        if is_deg:
            deg[g["family"]] += 1
            by_split[g["split"]][0] += 1
        else:
            mass_nondeg += sum(r)

    n = len(groups)
    frac = sum(deg.values()) / n

    # Per-step expectation at 4 groups/step: E[degenerate groups] and
    # P(step fully wasted) under the census rate (binomial, iid draws).
    p_step_all = (frac) ** GROUPS_PER_STEP
    e_deg_per_step = frac * GROUPS_PER_STEP

    result = {
        "n_groups": n,
        "k": 8,
        "zero_std_frac": round(frac, 4),
        "verdict_rule": {"drop_below": 0.05, "mandatory_above": 0.20},
        "verdict": ("DROP" if frac < 0.05 else
                    "MANDATORY-T1" if frac > 0.20 else "OPPORTUNISTIC-T1"),
        "by_family": {f: {"deg": deg[f], "n": tot[f],
                          "frac": round(deg[f] / tot[f], 4)}
                      for f in sorted(tot)},
        "by_split": {s: {"deg": d, "n": t, "frac": round(d / t, 4)}
                     for s, (d, t) in by_split.items()},
        "distinct_reward_hist": {str(k): v for k, v in sorted(distinct_hist.items())},
        "reward_mass_in_nondegenerate_groups": round(mass_nondeg / mass_all, 4),
        "per_step_expectation_4groups": {
            "expected_degenerate_groups_per_step": round(e_deg_per_step, 3),
            "p_step_fully_degenerate": round(p_step_all, 5),
        },
        "context_exact_no_op_windows_tails": {
            run: no_op_series(path) for run, path in RUNS.items()
        },
    }
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
