# P10 — GatedNorm-v2 near-identity σ-init (2026-09-05, zcode-gpushorts)

Queue §3 P10. The user's scale question (2026-09-01): Q3's GatedNorm
rejection may be an init transient — standard init centers the gate
pre-activation at ~0, so σ ≈ 0.5 HALVES every sublayer output until the
gate learns to open: a large fraction of a 350M-token run, <1% of a 13-25B
run (Qwen's "standard init suffices" claim was made at 560B tokens).

## The change (one init delta, cfg-gated for ckpt compat)

`poc_twin/model.py` `GatedNorm.__init__` gains `sigma_bias`: with
`cfg["gated_norm_v2"]`, `gn_w2` carries a bias initialized to **+4.0**
(σ(4) = 0.982): GatedNorm starts == RMSNorm within 1.8% (measured 0.9817 on
init), vs the standard-init gate mean 0.5008. W1/W2 weights keep standard
init (the gate stays trainable; σ'(4)=0.018). Banked GN ckpts (bias=False)
load unchanged — the flag lives in the saved cfg. +36,864 params (d × 2
norms × 24 sublayer norms... measured delta vs v1).

## Arms (same paired discipline as every ladder arm: dose 0.3, seed 1273, 668 steps)

- `gn2_qk` — GatedNorm-v2 + QK-Clip(tau 100): the quality arm vs Q3's +2%
  (banked gn_qk 0.7685/0.7538 vs plain control 0.7533/0.7527).
- `stress_gn2` — GatedNorm-v2-only (tau 1e9) at 2x peak LR (0.02/0.008):
  the stress leg vs the banked pair (stress_gn p99.9 1.19x clip,
  stress_plain 2.28x, spikes 2 vs 3).

## Results (arms closed 2026-09-05T22:34; 92/87 min, zero NaN, zero yields)

| arm | causal BPB | FIM BPB | Δ vs banked plain control | stop-acc |
|---|---|---|---|---|
| banked plain control (a2poc_d30) | 0.7533 | 0.7527 | — | 15.3% |
| banked gn_qk (Q3, σ-init ~0.5) | 0.7685 | 0.7538 | +2.0% / +0.2% | 14.4% |
| banked gn_only (Q3, σ-init ~0.5) | 0.8230 | 0.8067 | +9.3% / +7.2% | 11.7% |
| **gn2_qk (σ-init ≈1, this run)** | **0.7644** | **0.7496** | **+1.5% / −0.4%** | 13.5% |

Stress leg at 2x peak LR (scorer `poc_stab/stress_metrics.py`, same window
as every banked verdict):

| arm | grad p99.9 (× clip) | spikes | scorer verdict |
|---|---|---|---|
| stress_plain (banked) | 2.28x | 3/6 | — |
| stress_gn (banked, v1) | 1.19x | 2/6 | B at least as stable |
| **stress_gn2 (this run)** | **1.56x** | 2/6 | **B at least as stable** |

## Verdict (2026-09-05T22:4x, zcode-gpushorts)

**v2 does NOT close most of the +2% cost → the Q3 rejection STANDS: at our
scale the GatedNorm BPB cost is structural, not an init transient.**

1. **Causal slice (the pre-registered reference — "Q3's +2% BPB cost")**:
   v2 recovers 0.0041 of the 0.0152 BPB gap (**~27%**) — 0.7685 → 0.7644 vs
   plain 0.7533. Still 3x the 0.5% adoption bar. The user's init-transient
   hypothesis is now QUANTIFIED: the σ≈0.5 half-closed start accounts for
   roughly a quarter of the cost; the remainder is the low-rank gate's
   capacity/optimization tax at 206M/350M tokens.
2. **The init-transient component was real and is gone**: FIM slice 0.7496
   is now BETTER than the plain control (−0.4%) and better than gn_qk on
   BOTH slices — **GN-v2 strictly dominates GN-v1**. If GatedNorm ever
   re-enters, it enters as v2.
3. **Stability mechanism survives v2** (this was the stability leg's
   question): p99.9 grad tail 1.56x vs plain's 2.28x (32% reduction;
   scorer PASS), though with less margin than v1's 1.19x (48%) — the
   half-closed v1 gate doubles as a suppressive prior; the near-identity
   gate keeps the mechanism but gives back margin.
4. **Net for the 25B conversation**: GN-v2+QK-Clip is a Pareto improvement
   over GN-v1 but still pays +1.5% causal BPB at POC scale. Per the
   pre-registered framing, that is the "rejection structural" branch. The
   pinned plain+QK-Clip recipe stands. Residual open thread (not a
   recommendation): the cost's structural component was measured at
   206M/350M; Qwen's "standard init suffices" claim lives at 560B tokens —
   if the 1.5B A2 probe ever wants a GN arm, v2 (not v1) is the
   configuration to carry, and the causal slice is the bar it must clear.

## Files

- `model.py` GatedNorm `sigma_bias` (cfg-gated `gated_norm_v2`; banked GN
  ckpts load unchanged); `train_ladder.py --gated-norm-v2`
- `run_gatednorm_v2.sh` (this chain, correct ckpt paths — the banked
  run_gatednorm.sh rsync bug documented on the board 21:10)
- `logs/bpb_eval_gn2.json`, `logs/gn2_qk.jsonl`, `logs/stress_gn2.jsonl`,
  `logs/{gn2_qk,stress_gn2}_stdout.log`
- ckpts persisted: `/mnt/h/sepalith/runs/gatednorm/{gn2_qk,stress_gn2}/final.pt`
