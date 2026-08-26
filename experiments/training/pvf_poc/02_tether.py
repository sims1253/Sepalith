#!/usr/bin/env python3
"""PVF POC step 2: TETHER offline variance analysis (CPU post-processing).

Tests the Le Critique TETHER claim on our replayed groups + step-1 critics:
does blending the value baseline into the GRPO group baseline reduce
advantage-estimator variance, and what rho does the least-squares fit pick?

All quantities are sequence-level (our rewards are per-completion), so the
gradient-noise proxy for an estimator A = R - baseline(R, V) is Var(A)
pooled over val groups. Baselines compared per group member i (K members):

  mean : R_bar (group mean incl. self — the vanilla GRPO baseline)
  loo  : (sum_g R - R_i) / (K - 1)            (dr_grpo/GSPO-era default)
  blend: (1-rho) * loo_i + rho * V_i           (TETHER; rho in [0, 1])

rho protocol (paper's): fit rho on the PREVIOUS batch by least squares —
minimizing sum_i (R_i - [(1-rho) loo_i + rho V_i])^2 within batch groups —
with EMA smoothing (0.9); evaluate variance on the current batch so the
adaptive choice can't peek. Batches: 32 groups, seeded shuffle. rho=0 start.

Also the BPCO single-rollout check (K=1): per-prompt gradient-variance
proxy comparison:
  K=4 group LOO  : Var(A_loo) / K      (gradient averages K members)
  K=8 group LOO  : Var(A_loo) / K
  K=1 + critic   : Var(R - V)          (one rollout, critic baseline)
K=4 uses seeded subsamples of our K=8 groups.

Inputs: /mnt/h/sepalith/runs/pvf_value_{plain,priv}/predictions_val.jsonl.
Usage: python 02_tether.py [--batch-groups 32]
"""
import argparse
import json
import random
from pathlib import Path

RUNS = Path("/mnt/h/sepalith/runs")
EMA = 0.9
SEED = 3407


def load(arm):
    groups = {}
    for line in open(RUNS / f"pvf_value_{arm}" / "predictions_val.jsonl"):
        r = json.loads(line)
        groups[r["pid"]] = dict(rewards=r["rewards"], values=r["values"],
                                family=r["family"])
    return groups


def loo_baselines(rewards):
    s = sum(rewards)
    return [(s - r) / (len(rewards) - 1) for r in rewards]


def var(seq):
    if len(seq) < 2:
        return 0.0
    m = sum(seq) / len(seq)
    return sum((x - m) ** 2 for x in seq) / len(seq)


def pooled_var_adv(groups, rho, use_v):
    """Var of R - [(1-rho) LOO + rho V] pooled over all members."""
    advs = []
    for g in groups:
        loos = loo_baselines(g["rewards"])
        for r, loo, v in zip(g["rewards"], loos, g["values"]):
            b = (1 - rho) * loo + rho * v if use_v else loo
            advs.append(r - b)
    return var(advs)


def fit_rho(groups):
    """Least-squares rho minimizing || R - [(1-rho) LOO + rho V] ||^2:
    closed form rho* = sum (R - LOO)(V - LOO) / sum (V - LOO)^2."""
    num = den = 0.0
    for g in groups:
        loos = loo_baselines(g["rewards"])
        for r, loo, v in zip(g["rewards"], loos, g["values"]):
            num += (r - loo) * (v - loo)
            den += (v - loo) ** 2
    if den == 0:
        return 0.0
    return min(1.0, max(0.0, num / den))


def analyze(arm, batch_groups):
    all_g = load(arm)
    pids = sorted(all_g)
    rng = random.Random(SEED)
    order = pids[:]
    rng.shuffle(order)
    batches = [order[i:i + batch_groups]
               for i in range(0, len(order) - batch_groups + 1, batch_groups)]

    # mean-baseline variance (R - R_bar) for reference
    advs = []
    for p in pids:
        rs = all_g[p]["rewards"]
        m = sum(rs) / len(rs)
        advs += [r - m for r in rs]
    v_meanbase = var(advs)

    # TETHER trajectory: fit on previous batch, evaluate on current
    rho, traj = 0.0, []
    v_blend_eval, n_eval = 0.0, 0
    for bi, b in enumerate(batches):
        cur = [all_g[p] for p in b]
        v_blend = pooled_var_adv(cur, rho, use_v=True)
        v_loo = pooled_var_adv(cur, 0.0, use_v=False)
        traj.append(dict(batch=bi, rho=round(rho, 4),
                         var_blend=round(v_blend, 6),
                         var_loo=round(v_loo, 6),
                         ratio=round(v_loo / v_blend, 4) if v_blend > 0
                         else None))
        v_blend_eval += v_blend * len(b)
        n_eval += len(b)
        rho = EMA * rho + (1 - EMA) * fit_rho(cur)
    v_blend_avg = v_blend_eval / max(n_eval, 1)
    rho_final = fit_rho([all_g[p] for p in pids])

    # fixed-rho sweep (oracle rho = final fit, for the headline number)
    sweep = {}
    for rho_x in [0.0, 0.25, 0.5, 0.75, 1.0, round(rho_final, 3)]:
        v = pooled_var_adv([all_g[p] for p in pids], rho_x, use_v=True)
        sweep[round(rho_x, 3)] = round(v, 6)

    # BPCO single-rollout check (privileged arm only)
    single = None
    if arm == "priv":
        rng2 = random.Random(SEED)
        v1 = []  # Var(R - V), per member — K=1 estimator
        v4 = []  # Var(LOO advantage) on K=4 subsamples / K
        v8 = []
        for p in pids:
            g = all_g[p]
            for r, v in zip(g["rewards"], g["values"]):
                v1.append(r - v)
            idx = list(range(len(g["rewards"])))
            rng2.shuffle(idx)
            sub = idx[:4]
            rs = [g["rewards"][i] for i in sub]
            loos = loo_baselines(rs)
            v4 += [r - l for r, l in zip(rs, loos)]
            loos8 = loo_baselines(g["rewards"])
            v8 += [r - l for r, l in zip(g["rewards"], loos8)]
        single = dict(
            var_single_rollout_critic=round(var(v1), 6),
            var_k4_group_per_member=round(var(v4) / 4, 6),
            var_k8_group_per_member=round(var(v8) / 8, 6),
        )

    v_loo_all = pooled_var_adv([all_g[p] for p in pids], 0.0, use_v=False)
    out = dict(
        arm=arm, n_groups=len(pids), n_batches=len(batches),
        var_mean_baseline=round(v_meanbase, 6),
        var_loo=round(v_loo_all, 6),
        var_tether_prevfit_avg=round(v_blend_avg, 6),
        reduction_vs_loo=round(v_loo_all / v_blend_avg, 4)
        if v_blend_avg > 0 else None,
        rho_oracle_final=round(rho_final, 4),
        rho_sweep=sweep,
        rho_trajectory_last5=traj[-5:],
        mean_rho_traj=round(sum(t["rho"] for t in traj) / len(traj), 4)
        if traj else None,
        single_rollout_check=single,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-groups", type=int, default=32)
    args = ap.parse_args()
    results = {}
    for arm in ("plain", "priv"):
        p = RUNS / f"pvf_value_{arm}" / "predictions_val.jsonl"
        if not p.exists():
            print(f"skip {arm}: {p} missing")
            continue
        results[arm] = analyze(arm, args.batch_groups)
    out = Path("/mnt/h/sepalith/datasets/pvf_poc_v1/tether_analysis.json")
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
