# Decay/CMA + MuonH POC — RESULTS & VERDICT (2026-08-30)

Five 1BT arms at the 206M ladder instrument, identical block multisets/seed
(only order/schedule/optimizer-wrapper differ). Primary readout: held-out
R-BPB (r_eval_causal 4.13M tok / r_eval_rc 0.28M / so_r_qa_eval 0.56M).
Plan: `docs/research/2026-08-28-decay-cma-muonh-poc-plan.md`. Raw JSONs:
`/tmp/poc_cma/readout/` (post-reboot rebuild: draw/eval_blocks deterministic,
byte-identical recipe).

## Arm table (BPB, lower = better)

| arm | r_eval_causal | r_eval_rc | so_r_qa | vs C (causal) |
|-----|--------------|-----------|---------|---------------|
| **C** control (uniform, decay .2, Muon-mix) | **0.6336** | 0.5518 | **0.6493** | — |
| D (uniform, decay .5) | 0.6536 | **0.5385** | 0.6592 | +3.2% worse |
| K (curriculum, decay .2) | 0.7612 | 0.6196 | 0.7626 | +20.1% worse |
| KT (curriculum + const-tail) | 0.7282 | 0.6175 | 0.7361 | +14.9% worse |
| KT_avg (KT + SMA6 of tail ckpts) | 0.7164 | 0.6230 | 0.7294 | +13.1% worse |
| H (MuonH, wd=0 + radius proj.) | 0.7577 | 0.8111 | 0.8082 | +19.6% worse |
| C_avg3 (C endpoint avg, no tail — readout only) | 0.6433 | 0.6138 | 0.6817 | +1.5% worse |

Canary (verbatim n-gram, mean holdout loss): H 2.111 vs C 1.608 — H
non-inferiority FAILS. ELR telemetry: C ran at effective LR ≈0.049 vs
nominal η=0.01 (plain Muon's implicit ELR ≈5× nominal at these norms).

## Verdicts per pre-registered adoption rule

1. **Decay fraction: KEEP 0.2.** D does not beat C by ≥0.5% (worse on 2 of
   3 slices; the r_eval_rc win −2.4% is a 0.28M-token slice, far below
   gate). Longer decay did NOT help at 1BT/206M — contra Puro's
   longer-decay trend (their regime: higher peak LR, 938B-token phase).
2. **Curriculum: KILLED.** K ≤ C and KT ≤ C on essentially everything →
   the pre-registered kill test fires; CMA closed for our regime at low
   repetition. Honest reading: per-block CE of the scorer is a BAD quality
   proxy here — "hard" conflates with "noisy", and ending the run on the
   highest-loss blocks under decaying LR hurt broadly. Puro's CMA used
   curated quality scores on a largely-unique 938B corpus; our 1BT draw at
   1–5 R-epochs is a different regime (pre-registered threat (a) realized).
3. **Tail + averaging package: mechanism confirmed, base rejected.** Within
   the curriculum family, KT > K (−4.3% causal) and KT_avg > KT (−1.6%)
   — the const-LR tail and SMA6 averaging both helped exactly as Puro's
   mechanism predicts. But from a base 15–20% behind uniform C, the package
   cannot recover. C_avg3 (no-tail average) is WORSE than C (+1.5–11%) —
   confirming "averaging without const-LR tail is neutral-to-negative".
   No production adoption (gates require beating C).
4. **MuonH: REJECTED at pinned LR.** H is ~20% worse across slices with
   40× gradient norms mid-run and canary non-inferiority failing. The
   radius projection + wd=0 at lr 0.01 under our RMS-matched
   parameterization is a mis-scale (Puro's recipe uses an explicit 10×
   multiplier over AdamW base — our lr transfer does not carry it). Since
   H ≉ C, the plan's LR-matched isolation arm is NOT triggered. v2 note:
   if MuonH is ever revisited, tune its LR down first (the ELR telemetry
   now exists to do this rationally); do not carry the pinned-lr result as
   evidence against the mechanism.
5. **Scale-up (2BT): NOT triggered.** No arm beat C; the rule scales up
   only top-2 arms for adoption candidates.

## Bottom line for the production run

**The pinned recipe is validated as-is: uniform order, decay_frac 0.2,
Moonlight Muon-mix (lr 0.01/embed 4e-3/wd 0.1), QK-Clip, no tail, no
averaging, no MuonH.** `a2-cluster-runbook.md` §3.2 needs no change. The
adopt-now snapshot densification stands (free); FP8 rental-hour-1 validation
stands (separate axis, untested here).

## E2 (post-persistence): MOOTED, recorded deviation

E2 existed to check whether pretrain-side WINS survive the post stage. There
are no wins to persist (C is the winner and is the status quo). Running
C-vs-K post-stage would test persistence of a regression — not the decision
the plan needed. Skipped with this note; if a future pretrain win appears,
re-invoke it.

## Deviations log

- Eval holdouts position-disjoint (strata lack per-block package metadata) —
  recorded at Task 1; same holdouts for all arms, so paired comparisons hold.
- KT restarted at ~10 min (wrong order file caught by owner session QA).
- Owner sessions died twice (usage limits) + a 12:34 reboot wiped /tmp;
  draw/eval_blocks rebuilt deterministically; readout run by zcode-main.
- Runbook amendment task resolves to "no change" (see bottom line).

## E3 — proxy strata selection (2026-08-30 23:02 done)

Protocol: shared scorer ckpt + 480-step continuation each (identical schedule,
seed 1273); ramps 0->~44-80% candidate share (e3_orders RAMP_WINDOWS); readout
= capability vector vs the control continuation. Delta vs e3_control:

| arm | r_eval_causal | r_eval_rc | so_r_qa_eval (target probe) | own-stratum holdout |
|-----|--------------|-----------|------------------------------|---------------------|
| so_r_qa ramp | +2.0% | +5.9% | **-23.7%** | +3.5% |
| bioc ramp | +2.2% | +7.5% | +2.9% | +4.0% (bioc) |
| curated_py ramp | +7.4% | +11.8% | +8.8% | +7.3% (curated_py) |

**Calls per the frozen rule** (adopt iff >= neutral on held-out R-BPB AND
positive on target probe; drop iff negative on both):
- **so_r_qa: NO adoption under the rule** (R-BPB -2.0% fails strict
  neutrality) — but it is the ONE strong signal of the program: a -23.7%
  target-probe gain for only +2% R cost at an extreme 80% dose implies the
  current 1.2% draw share under-doses the stratum. v2 recommendation (not
  adopted): a dose-response pair at 2x/4x production share (~2.4%/4.8%),
  where the R cost scales down with dose; expect the crossover in between.
- **bioc: DROP candidate** (negative on both: R +2.2%, own holdout +4.0%
  — the stratum fails to help even itself at high dose).
- **curated_py: DROP** (worst everywhere; +7.4% R, +7.3% own holdout).

Caveats: single seed; ramps are extreme-dose probes (capability-axis
exposure, not production shares); all arms share the LR-re-entry quirk
(schedule resumes at peak after the scorer's decayed end — equal across
arms incl. control, so pairing holds); the so_r_qa_eval slice is the packed
QA eval (distribution-matched to the ramp corpus — read the -23.7% as
upper-bound capability transfer).

## E3 deviations log
- run_e3.sh order-file alignment bug (absolute-step indexing vs
  continuation-sized orders): fixed by padding 245,760 unread rows;
  originals kept as order.npy.unpadded. Trainer untouched.
- Scorer resume path restored post-reboot via NAS symlink.

## Post-verdict supplement (2026-08-31): MuonH mechanism precision, full-text Puro read

For anyone who revisits this axis (source: arXiv:2608.27370 full text +
Hyperball arXiv:2606.16899; papers-recon session, 2026-08-31):

- Puro's MuonH is the Hyperball wrapper: it pins BOTH the weight Frobenius
  radius AND the update norm to constants — the update is normalized, so
  relative displacement ‖ΔW‖/‖W‖ per step equals the LR exactly, and the
  MuonH param group runs at 10× the AdamW-base LR schedule. Effective LR
  is a directly prescribed quantity in their formulation.
- Arm H here keeps plain-Muon update scaling at the pinned LR 0.01 and
  adds only the post-hoc radius projection — a different operator. Under
  it the projection fights the optimizer's natural radius growth every
  step (the 40× mid-run grad norms are the symptom), so verdict 4's
  +19.6% rejects "projection-only MuonH at pinned LR," NOT the Puro
  mechanism (their receipts: 1.19× compute-equivalent for the complete
  recipe vs a tuned-Muon baseline; 170M isolation MuonH 3.029 / Muon
  3.073 / ELR-matched Muon 3.030; Hyperball: 20–30% token-equivalent
  over weight-decay baselines ≤1.2B).
- If the axis reopens (13B-gate re-cut class decisions), the faithful
  recipe is update-norm pinning + an ELR-matched LR search — the Task-0
  ELR telemetry is the instrument for it.
