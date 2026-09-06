# X5-S1 — FRM recurrent self-conditioning + Fixed-Point Forcing on the MD span head

Run 2026-09-06 10:28–16:06 +02, zcode-x5-s1. Pre-registration:
`docs/EXPERIMENT-QUEUE.md` §3 X5 S1 row (verbatim bars) + FRM digest
`docs/research/2026-09-05-frm-intel.md` §1–2 + S0 readout
`X5_S0_RESULTS.md` (banked baseline this run re-measures). GPU chain:
Stage A = two-pass self-conditioning continuation from banked
`md_final.pt` (400 steps x 524,288 tok = 0.21B, Muon lr 3e-3, fresh
optimizer); Stage B = FPF (260 steps = 0.14B, lr 1.5e-3,
t_start ~ Beta(2,2) capped at supervision t, rollout depth 2,
FPF prob 0.5, objective unchanged). Surgery: zero-init d x d carry
channel ingesting prev-pass RAW span logits as the soft clean-prediction
embedding (softmax(logits) @ tied-embed — a raw-logit (V,d) projection
would add 100.3M params at V=130,562; the paper leaves ingestion
unspecified). Gates: 11 CPU unit tests (G1–G6) green pre-GPU, G1
(zero-init at load) verified on the real banked ckpt, depth-1 sampler
parity vs `sample.sample_spans` bit-exact, S0 replay parity byte-exact
in-process.

Recon (row requirement): **Yoo et al. 2026 = arXiv 2607.00714**
"Self-conditioned Flow Map Language Models via Fixed-point Flows"
(COLMW 2026) — the language-domain existence proof FRM v2 cites; shows
self-conditioning solves a fixed-point iteration distillable into flow
maps. Also re-fetched FRM v2 App-A: their code draws t_start ~ U(0,t);
the queue row pre-registers Beta(2,2)-capped — the row was followed
(support [t,1] in our mask-rate mirror either way); flagged as a
recipe deviation risk, not a verdict input.

## VERDICT (pre-registered): **NO-WIN — the 51–256 lift-off does not happen. Write-the-negative.**

- exact > 0.0741 at ≤2x anchor NFE: **PASS** (s8k2 = 0.0787 at NFE 16 —
  a QUARTER of the 64-anchor's nominal NFE; 11-50 0.1250)
- **51–256 lift-off: FAIL — 0.000 at every (steps, depth) leg up to NFE
  256** (first nonzero long-span exact NOWHERE in the S1 arm; the bucket
  stays at the paradigm-wide floor)
- 11–50 regression vs 0.110: **no regression** (0.1250 at the best ≤128
  leg; ≥0.110 at every leg except s8k1/s8k8 short-budget legs) → no KILL
- ≥4x NFE branch (latency-negative): s32k8 = 0.0833 (18/216, best
  absolute) at NFE 256 — an exact gain but STILL no lift-off → the
  "win only at ≥4x NFE" branch's condition set is also unmet. Verdict
  logic (`verdict_x5_s1.txt`): **NO-WIN**.

## Leg table (216-row harness verbatim; greedy, batch-1, CUDA-event latency)

| leg | nominal NFE | as-run NFE | exact | vs banked anchor | 11-50 | 51-256 | p50 ms |
|---|---|---|---|---|---|---|---|
| s8k1 | 8 | 7.5 | 0.0463 (10) | banked@8 0.0509 | 0.0735 | 0.000 | 169 |
| s16k1 | 16 | 13.7 | 0.0602 (13) | banked@16 0.0417 | 0.0956 | 0.000 | 297 |
| s32k1 | 32 | 23.5 | **0.0694 (15)** | **banked@32 0.0694 — exact parity** | 0.1103 | 0.000 | 481 |
| s64k1 | 64 | 35.1 | **0.0741 (16)** | **banked@64 0.0741 — exact parity** | 0.1176 | 0.000 | 757 |
| s8k2 | 16 | 15.1 | **0.0787 (17)** | > 64-anchor at 1/4 NFE | 0.1250 | 0.000 | 327 |
| s8k4 | 32 | 30.2 | 0.0694 (15) | | 0.1103 | 0.000 | 692 |
| s8k8 | 64 | 60.4 | 0.0648 (14) | | 0.1029 | 0.000 | 1197 |
| s16k2 | 32 | 27.4 | 0.0741 (16) | = 64-anchor at 1/2 NFE | 0.1176 | 0.000 | 570 |
| s16k4 | 64 | 54.7 | 0.0694 (15) | | 0.1103 | 0.000 | 1068 |
| s16k8 | 128 | 109.4 | 0.0741 (16) | | 0.1176 | 0.000 | 2045 |
| s32k2 | 64 | 47.0 | 0.0741 (16) | = 64-anchor at 1x NFE | 0.1176 | 0.000 | 884 |
| s32k4 | 128 | 93.9 | 0.0741 (16) | | 0.1176 | 0.000 | 1730 |
| s32k8 | 256 | 187.9 | **0.0833 (18)** | best absolute, 4x NFE | 0.1324 | 0.000 | 3504 |
| s32k1null | 32 | 23.5 | 0.0694 (15) | null-carry diagnostic = anchor | 0.1103 | 0.000 | 445 |
| s64k1null | 64 | 35.1 | 0.0741 (16) | null-carry diagnostic = anchor | 0.1176 | 0.000 | 672 |

Equal-NFE ladder (exact counts): NFE 64 → (64,1)=16 / (32,2)=16 /
(16,4)=15 / (8,8)=14; NFE 128 → (32,4)=16 / (16,8)=16; NFE 256 →
(32,8)=18. best-of-8 (k=1 legs): 0.0602/0.0741/0.0787/0.0833 at
8/16/32/64 vs banked 0.0602/0.0648/0.0741/0.0880 — within ±1-row noise.

## What the negative says (mechanism read)

1. **The long-span claim fails in our substrate.** The paper's central
   transfer — recurrent carry converts depth into consistency where
   parallel token updates are inconsistent — does NOT reach the 51–256
   bucket: 0/80 rows at every depth, and post-FPF the bucket's residual
   dynamics look *more* converged (r_final med 0.090 @64 vs S0's 0.108)
   without any correctness following. Convergence-as-correctness keeps
   separating (see AUROC below) — the model knows when it is unsure —
   but the long-span failure is not a fixable-by-recurrence inconsistency
   at this scale/budget; it behaves like a capability/coverage wall
   (0.14B continuation tokens vs the anchor's 2B; the paper's DiTs were
   trained to saturation on ~1k puzzles).
2. **Depth does not beat schedule passes at equal NFE** (16/16/15/14 at
   NFE 64) — the one clear efficiency point is at the LOW-NFE corner:
   s8k2 (17 rows at 15.1 as-run forwards) vs the banked @16 (9 rows) and
   vs the @64 anchor (16 rows at 35.1 as-run) — recurrence compresses
   ~2.3x of the schedule's as-run forwards at anchor parity. Wall-clock
   caveat: at equal NFE the recurrent legs cost ~1.3-1.6x the schedule
   legs' latency (the per-forward (L,V)@(V,d) carry ingestion matmul);
   s8k2's p50 327ms vs the 64-anchor's 757ms is nonetheless a real
   ~2.3x wall reduction.
3. **The continuation is safe**: null-carry legs reproduce the banked
   anchors exactly (15/216, 16/216) — the 50%-null two-pass mix held
   zero-carry competence (null-carry eval_loss ended 2.26, inside the
   anchor's 1.90-2.04±estimator-noise band; transient bumps at steps
   100/300 recovered both times).

## Secondary readouts (the bankable half)

**Post-FPF residual AUROC** (S0 instrument verbatim, positive =
incorrect; paper predicts →1.0; S0 banked 0.860):

| leg | AUROC | 95% CI | S0 baseline |
|---|---|---|---|
| steps=64, k=1 | **0.927** | 0.846–0.980 | 0.860 [0.741–0.952] |
| steps=32, k=1 | **0.911** | 0.807–0.983 | 0.897 [0.819–0.962] |
| steps=32, k=8 | **0.974** | 0.943–0.995 | (no S0 analog) |
| steps=32, k=8 recurrent-residual variant | 0.938 | — | paper's Fig-6 instrument |

**Residual-gated abstain (noopFP-adjacent; post-hoc operating points):**

| leg | incorrect suppressed @ 0 exact lost | @ 1 lost | S0 @0/16 |
|---|---|---|---|
| steps=64, k=1 | **44.5%** | 84.5% | 28.5% |
| steps=32, k=1 | 34.3% | 67.7% | (48% @1/16 @64-equivalent) |
| steps=32, k=8 | **80.3%** | 90.9% | — |

**Early fixation: NOT delivered.** fixed-by-k≤8 = 2.3% @64 (S0 3.2%),
10.2% @32d8 overall / 16.2% within 11-50 / 0% within 51-256; median k*
33 @64 — nowhere near the paper's 97.5%. The big latency prize (early
commit) remains open; what S1 banked instead is the LOW-NFE parity
point (s8k2) + the abstain upgrade.

## Caveats

1. Continuation budget 0.35B tokens total (A+B) vs the anchor's 2B —
   the paper's protocol trains to saturation; undertrained-carry is a
   live alternative explanation for the missing lift-off (carry_w
   absmax was still growing ~linearly at stage ends: 0.29 @A-end).
2. Warm-start surgery (their stages are from-scratch) — the pre-registered
   main falsifier (intel doc §3.3) stands unresolved-but-not-triggered:
   nothing diverged; the channel trained stably throughout.
3. Small-count noise: exact deltas of ±1-2 rows (n=216, 16 positives)
   are ~1pp; only s8k2 (+8 rows vs banked@16) and s32k8 (+2 vs anchor)
   exceed it.
4. Beta(2,2) vs U(0,t) t_start deviation (row vs paper code) — flagged
   above; the row was followed.
5. Rollout depth fixed at 2 (their depth 16 on 81-900-token puzzle
   grids would cost ~8 supervised passes/step here at 130k vocab);
   deeper rollouts untested.
6. Eval legs greedy/temperature-0 only for depth>1 (bo8 run only on the
   k=1 harness-verbatim grid); multimodality at depth unmeasured.

## Ops log

- Chain `scripts/run_x5_s1_chain.sh` (detached, setsid; owner-session
  heartbeats q30min; taskset 16-23; memfrac 0.42 — actual peak 15.7GB).
- Two restarts, both caught by gates with zero lost training: (a) an
  nlines() edge on a not-yet-created log (fixed pre-smoke), (b) the
  B-stage resume path pointing at the old `ckpt/` staging dir (A was
  already banked; script gained an A-skip guard).
- The tmpfs training flat-pack had evaporated; regenerated via
  `data_prep` — `eval_triples.jsonl` verified BYTE-IDENTICAL (md5
  8ffac9e1…) and train tokens 44,475,535 = the frozen budget.
- Training telemetry: A loss 0.745→0.641, B 0.688→0.597, QK max 38-54
  (tau 100, no clipping), grad norms 0.13-0.18, 28-34k tok/s (A) /
  23-25k (B) on the idle box, VRAM 13.2-15.7GB.
- Wall: 10:28–16:06 (5h38m incl. restarts + eval; GPU-busy ~5h05m).

## Artifacts

- Ckpts: `/mnt/h/sepalith/runs/x5_s1/x5_s1_a_final.pt` (stage A),
  `x5_s1_b_final.pt` (stage B = the S1 head), + latest/yield variants
- Per-leg rows + summaries: `results_x5_s1/eval_x5_s1.json`,
  `rows_s{S}k{K}*.jsonl`; residuals `residuals_steps*.jsonl` +
  `analysis_x5_s1.json`; verdict `verdict_x5_s1.txt` (repo copies
  gitignored-adjacent → mirrored whole to
  `/mnt/h/sepalith/runs/x5_s1/results_x5_s1/`, B8b convention)
- Training logs: `logs_x5_s1_a.jsonl` / `logs_x5_s1_b.jsonl`
- Chain log: `/mnt/h/sepalith/runs/x5_s1/chain.log`

## Recommendation to the queue

Close X5-S1 as a **negative on the primary (long-span lift-off) with two
bankable secondary wins**: (1) the residual-gated abstain gate is now
80%-class at zero exact-loss when depth-8 recurrence is affordable, 44%
at the anchor's own NFE — directly feeds noopFP/H-series; (2) the s8k2
operating point (anchor parity at ~2.3x fewer forwards / ~2.3x less
wall) is the first measured latency-positive test-time-scaling point on
the MD head. The FPF-v1 follow-ons (renoise-CE verifier, FlowDPO with
AST masks) remain NOT pre-registered; the 51-256 bucket's next lever is
probably budget/scale (or a different mechanism entirely), not deeper
recurrence — 0/80 at 4x NFE is a strong "not inconsistency-limited"
signal.
