# O3-S0 — suffix-entropy + length telemetry on banked RL runs

Agent: zcode-o3-telemetry. Date: 2026-09-04/05. Queue: `docs/EXPERIMENT-QUEUE.md`
§3 O3 (pre-registered). Premise source: OPD papers 2604.13016 / 2609.04172
mapped to our teacher-free GRPO (`experiments/training/rl_smoke.py`, reward =
exact + 0.2·line_f1, `num_generations=4`, 192-token completion cap, no EOS).

## VERDICT S0: **NOT LAND** (pre-registered criterion: telemetry predicts
stall BEFORE the exact metric does)

O3-S1 (curriculum arm) stays closed. What the exercise DID bank: a
run-state saturation instrument (`frac_reward_zero_std` / `reward_std`),
two non-signals (length, grad-norm) now known degenerate under the
current trainer config, and a hard artifact-resolution blocker
(`save_total_limit=2`) that makes within-run entropy lead/lag untestable
from banked artifacts. Scale caveat: banked runs are 220–300 optimizer
steps; the papers' stall regime is 3K–10K+ steps — the banked corpus
ends before the regime where their entropy claim has native support, so
this verdict is "not demonstrated here", not "papers' premise false".

## 1. Inventory of banked telemetry (stale-log checks done)

10 runs under `/mnt/h/sepalith/runs/`: `rl_grpo_v1, v2, v2b, v2c, v2d,
v3, v3b, v4_tether, v5_loo_unnorm, t1_dapo`. Verified per run: metrics
mtime vs checkpoint mtimes vs step ranges vs families-in-log vs
`adapter_config.json` base paths vs night notes (2026-08-23) — all
consistent, no stale reads. Configs pinned by cumulative family shares:

| run | steps | base (SFT LoRA) | data | profile | verified |
|---|---|---|---|---|---|
| v1 | 220 | sft_v6 | sft_v6 | 4-family, no_op 10.7% | mtime 08-21, board 08-20/21 |
| v2 (2a) | 300 | sft_v7 | sft_v7 | run2 + refine, no_op 31.2% | 08-23 |
| v2b (2b) | 300 | sft_v7 | sft_v8 | run2 + refine, no_op 30.6% | 08-23 |
| v2c | 300 | sft_v7 | sft_v8_1 | run2 + refine, no_op 31.6% | 08-23 |
| v2d | 300 | sft_v7 | sft_v8_1 | run2 + refine, no_op 17.9% (knee, quota 800) | 08-23 |
| v3/v3b | 149/60 | v7-refine | — | warm-start arm, killed | 08-24 |
| v4_tether | 220 | sft_v6 | sft_v6 | PVF tether ρ=0.6 | 08-27 |
| v5_loo_unnorm | 220 | sft_v6 | sft_v6 | unnormalized LOO | 08-27 |
| t1_dapo | 220 | sft_v6 | sft_v6 | DAPO clip-higher | 09-01 |

What was logged:
- `rl_metrics.jsonl` — per optimizer step: reward, exact, per-family
  exact/n. No entropy, no grad norm, no completions (trl's
  `log_completions` prints to stdout only; no stdout logs survive).
- checkpoint `trainer_state.json` (last ckpt carries FULL history, 10-step
  cadence): `grad_norm`, `frac_reward_zero_std` (= share of 4-completion
  groups with zero reward std = unanimous groups = zero advantage — the
  advantage-concentration signal, already banked), `reward_std`, `kl`
  (≡0: β=0), `loss` (≡~1e-9: mean-zero group advantages make the GRPO
  loss value uninformative BY CONSTRUCTION — grad_norm is the real one),
  completion lengths, clip ratios (≡0: single on-policy pass).
- Checkpoints: **2 per run** (`save_total_limit=2`) + `final_lora`
  (≡ last ckpt). This is the blocker for entropy-trajectory analysis:
  weight points per run = base + 2.

## 2. Methods

Script: `experiments/training/rl/entropy_telemetry.py` (subcommands
`curves` / `predict` / `replay` / `inventory`). Pre-registered
definitions (docstring, fixed before reading curves): smoothed exact
(W=21), slope30 OLS stall (first t≥40 flat-for-good given a real rise),
collapse (argmax if final ≥0.10 below max), sustained signal onsets
(3 consecutive logs). Replay: re-merged bases in memory
(MiniCPM5-1B + banked SFT LoRA), rebuilt each run's prompt set with the
trainer's own `build_dataset` (same seed/quotas/data; rebuild shares
match logged shares — v2 30.9% vs 31.2% no_op etc.), 32 fixed prompts,
greedy rollout + teacher-forced pass on the base's reference completion
per weight point (base, ckpt-250/300 or 200/220). GPU claim
23:05–23:59 (comms/gpu.md), sole CUDA workload, W37 clean.

## 3. CPU-leg results (all 10 runs; `results/o3_s0/*_curves.{png,json}`)

- 7/10 runs never stall mid-run — still improving when stopped. Stalls:
  v2b 268 (collapse 252), v2c 267 (collapse 207), v2d collapse 270.
  The banked runs mostly end pre-plateau.
- Nominal onsets LEAD the stall: v2b Z≥0.70 @160, reward_std≤0.15 @120
  vs stall 268; v2c Z≥0.70 @170, reward_std @110 vs stall 267
  (+97…+157 steps). BUT these are confounded — concentration rises
  because exact rises.
- Honest deconfounded test (`predict`, probes 50–250, n=34):
  remaining-run-gain: r(Z) −0.76 raw; **−0.48 partial given exact+t
  (t=−2.97, p≈0.006)**; reward_std +0.48. Adding run-family as control
  → −0.16 (ns). Within-v2-family partials ≈ ±0.1 (ns). 50-step-lookahead
  gain: nothing (|r|≤0.15 partial).
  → Z/reward_std carry real information about how much run-level
  headroom is left (stop/continue instrument), but no demonstrated
  within-run lead beyond exact itself.
- grad_norm: no collapse signature anywhere. Flat 0.03–0.2 throughout;
  isolated 0.0 log entries are artifacts (single log points). DAPO arm
  RISES (median 0.13 → last 0.33).
- Length telemetry: degenerate. `completions/mean_length` 191.4–192.0,
  clipped_ratio 1.0, mean_terminated_length 0 — the model emits no EOS;
  every completion hits the cap. The papers' length/termination signals
  cannot express on this trainer as configured.

## 4. Replay results (8 runs × 3 weight points; `entropy_replay.json`,
`entropy_summary.png`)

Teacher-forced suffix entropy (nats, positions 96–191, base→mid→last):

| run | outcome (CPU leg) | tf H_sfx | own H_sfx |
|---|---|---|---|
| v2 | format collapse 263 | 0.151→0.185→0.146 | 0.151→**0.305**→0.094 |
| v2b | stall 268 | 0.106→0.087→0.086 | 0.106→0.068→0.100 |
| v2c | stall 267, collapse 207 | 0.154→0.113→0.102 | 0.154→0.054→**0.017** |
| v2d | collapse 270 | 0.106→0.111→0.119 | 0.106→0.083→0.029 |
| v1 | still rising | 0.312→0.236→0.239 | 0.311→0.117→0.158 |
| v4_tether | still rising | 0.312→0.526→**0.742** | 0.311→0.584→**0.827** |
| v5_loo_unnorm | still rising | 0.312→0.279→0.298 | 0.311→0.203→0.293 |
| t1_dapo | still rising | 0.312→0.269→0.258 | 0.311→0.396→0.320 |

- **Suffix-entropy level at run end separates the families — but that is
  the prompt-set/base confound** (v7 base starts at 0.11–0.15, v6 at
  0.31), NOT a stall predictor. Within-prompt-set the deltas do not
  separate: v1 (still rising) dropped suffix entropy 0.31→0.24 (−23%)
  with no stall; v2c (stalled) dropped 0.154→0.102 (−34%); v2d
  collapsed while entropy ROSE slightly.
- **Suffix-first RISE before stall: not observed.** Position-quartile
  shapes: base models peak at Q2 and decay; stalled runs flatten toward
  monotone decay (Q4 falls hardest: v2b final Q4 0.061) — i.e. our
  regime shows suffix COLLAPSE, concurrent with (not preceding) the
  exact-observed stall. The one rise-then-collapse arc (v2 own H_sfx
  0.151→0.305→0.094 across ckpt 250→300) is coincident with the format
  collapse at 263, n=1, and 50-step checkpoint resolution cannot
  establish order.
- Top-k churn (own-rollout top-1 match between consecutive weight
  points): late-window policy stabilization in v2d (0.39→0.70), v1
  (0.33→0.64), v5 (0.43→0.56); continued churn in v2 (0.15→0.22) and
  v2c (0.11→0.15) — the two format-collapse runs kept moving while
  collapsing. Directionally interesting, not a validated instrument.
- Side-finding for the PVF/tether line: v4_tether's policy entropy
  EXPLODES (×2.7 suffix, tf 0.31→0.74) while exact keeps rising —
  entropy growth under the tether objective is not distress there.

## 5. MetricsCb design note (spec only, NOT applied)

Patch to `rl_smoke.py` `MetricsCb` for future runs, ~40 lines, zero
extra forward passes:

1. **Surface what TRL already computes**: copy `frac_reward_zero_std`,
   `reward_std`, `grad_norm` from `state.log_history` tail (or recompute
   group stds in the reward fn — the stash already sees all 4 rewards
   per group if `completion_ids` grouping is plumbed) into
   `rl_metrics.jsonl` each step. Cheap, makes the saturation instrument
   per-step instead of per-10.
2. **Completion-position entropy at zero cost**: hook
   `on_step_end`… not possible for logits (TRL discards them); instead
   compute in the reward fn via `completion_ids` is also unavailable
   there. Minimal-cost option: subclass `GRPOTrainer._compute_loss`,
   from the existing per-token logits/logps compute mean entropy split
   forward (0–95) / suffix (96–191) + per-position decile means, stash
   like `STASH`, flush per step. Overhead ≈ one `log_softmax` on already
   materialized logits (<2% step time).
3. **Top-1 stability**: for a fixed probe batch (first 8 dataset rows,
   seeded), greedy-generate 192 tokens with per-position top-1 ids every
   `save_steps`, store alongside the checkpoint (`probe_top1.json`);
   churn = match rate vs previous probe. ~10 s per probe.
4. **Resolution fix — the actual blocker**: `save_total_limit=2` → 6
   (or `save_steps=50` kept with limit 6; disk ~260 MB per ckpt on NAS).
   Without intermediate weight points no future S0-style question is
   answerable post-hoc.
5. Log `n_zero_advantage` groups explicitly (group-level, not batch
   mean) — the per-family unanimous share is the O2 admission-filter
   readout anyway; one instrumentation, two experiments.

Adopt only if O3 is reopened with longer runs (the S1-gate question at
native 3K+ scale) or O2 lands (shared instrumentation).

## 6. Artifacts

- `experiments/training/rl/entropy_telemetry.py` — analysis (curves /
  predict / replay / inventory)
- `experiments/training/rl/results/o3_s0/` — per-run `*_curves.{png,json}`,
  `curves_summary.json`, `predict_test_{all,v2fam,v6fam}.json`,
  `entropy_replay.json`, per-run `*_entropy.json`, `entropy_summary.png`
- Board post: 2026-09-05 comms/board.md (verdict + numbers)
- GPU ledger: claim/release 2026-09-04 23:05–23:59

## 7. Bottom line for the queue

O3-S0 verdict NOT LAND: no telemetry in the banked runs predicts the
stall before exact does. The concentration signals are real but
concurrent state descriptors (and their nominal ~100-step "leads" do not
survive deconfounding). The papers' regime (3K–10K+ steps) is untested
here — banked runs are 220–300 steps and 7/10 end still improving. If
the line is ever reopened at native scale, §5 items 4+5 are the
prerequisites.
