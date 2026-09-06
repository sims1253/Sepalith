# X5-S0 — convergence-residual probe on the banked MD span head

Run 2026-09-06 01:47–04:5x +02, zcode-x5-s0. Pre-registration:
`docs/EXPERIMENT-QUEUE.md` §2c X5 S0 row (verbatim bars) + FRM digest
`docs/research/2026-09-05-frm-intel.md` §1 (Fig 6 instrument) / §2 M3 /
§3 non-transfers. Zero training: banked `md_final.pt` (step 3815,
206M) replayed on the 216-row poc_diff harness (eval triples
regenerated deterministically from `astfim_v1/fixed/eval.jsonl`,
216 rows = banked count). CPU-only replay (no CUDA context).

Instrument: `x5_s0_residual.py` — the existing low-confidence-remasking
schedule (`sample.py`) replayed per row at temperature 0, with the
model's full predictive distribution p_k probed at **all** span-region
positions every step (read-only; frozen positions included — the trunk
is bidirectional, so their distributions keep evolving as neighbors
freeze). r_k = mean_i SKL(p_k[i], p_{k−1}[i]) over span positions
(nats, full 130,562-vocab). Final-step residual R = r_{n_fwd}
(n_fwd = forwards as-run; the sampler early-exits when all positions
freeze). Argmax stability: a_k = argmax p_k; k* = first step of the
stable tail.

Validity anchors:
- Instrumented replay committed outputs **byte-identical to
  `sample.sample_spans`** on validation rows (6/6) — the probe does not
  perturb the sampler.
- Full-216 exact = **16/216 = 0.0741 = the banked MD@64 anchor
  exactly** (CPU fp32 reproduces the banked GPU run; no drift rows).

## Primary readouts (steps=64, the S1 anchor depth)

| readout | value | bar | pass |
|---|---|---|---|
| (a) AUROC of final-step residual vs span-exact (positive = incorrect) | **0.860** (bootstrap 95% CI 0.741–0.952) | ≥ 0.65 | **YES** |
| (b) early-stoppable rows (k* ≤ n_fwd − 2, as-run schedule) | **32.9%** (71/216) | ≥ 20% | **YES** |
| (b-strict) + committing the stable argmax reproduces the run's output text | 28.2% (61/216) | — | — |

**Both pre-registered bars pass ⇒ VERDICT: the confidence signal AND the
free-latency lever exist TODAY on the banked head.** Feeds (1) the
noopFP-abstain question — the residual is a working verifier-free
correctness signal (AUROC 0.86), and (2) S-series latency — a
commit-when-stable rule is immediately available. This is NOT the
"≈chance ⇒ stability is FPF-created" branch; S1 no longer carries the
whole claim — it now upgrades a measured-today signal (paper ceiling:
AUROC 1.00 under FPF) rather than creating it from nothing.

### Separation structure (n=216, 16 exact rows)

- exact rows' r_final: p50 = 0.0024, 13/16 below 0.0082
- incorrect rows' r_final: p10/p50/p90 = 0.016 / 0.108 / 0.304
- 5 exact rows churn (r_final 0.027–0.162) — exact-despite-instability
  exists; the signal is a strong rank-order separator, not a hard
  threshold on correctness.
- Confound check (length): AUROC(span_len as score) = 0.741 vs
  residual 0.860; **within the 11–50 bucket residual AUROC 0.851 vs
  length-alone 0.568**; spearman(r_final, span_len) among incorrect
  rows = −0.11. The signal is not a span-length proxy.

### Abstain-gate sizing (secondary; noopFP-adjacent)

Threshold on r_final (post-hoc operating points on this set — a shipped
gate needs held-out calibration):

| threshold | incorrect suppressed | exact rows lost |
|---|---|---|
| 0.162 (max exact) | 28.5% | 0/16 |
| 0.110 (2nd-max exact) | 48.0% | 1/16 (6.2%) |
| 0.008 (keeps 13/16 exact) | 94.0% | 3/16 (18.8%) |

### Latency lever arithmetic (steps=64)

- 32.9% of rows are early-stoppable (argmax stable ≥2 steps early);
  strict variant (stopping reproduces the output) 28.2%.
- Mean forward-pass saving among early-stoppable rows: **21.2%** of
  their schedule (23.9% within 11–50); fleet-wide mean NFE saving:
  **7.8%** — the lever is concentrated in short spans.
- fixed-by-k≤8: **3.2%** (median k* = 33 vs median n_fwd = 34). The
  banked head is NOT the paper's early-fixation regime — residuals
  separate correctness without the schedule itself converging fast.

## Paper reference points (context, not gates — FRM v2, 7–25M puzzle DiTs)

- AUROC 1.00 under FPF vs 0.50 vanilla self-conditioning → we measure
  **0.860 on a plain remasking head with zero recurrence/FPF** — most
  of the convergence-as-correctness signal is already present in our
  substrate.
- Fixed point by k≤8 in 97.5% under FPF → we measure 3.2% by k≤8,
  median stabilization at the schedule's end. The DYNAMICS gap (early
  fixation, and with it the big latency win) is what S1's Stage-B FPF
  actually targets; S0 says the SIGNAL half is bankable today.

## Per-bucket (the 51–256 story that matters for X5-S1)

| bucket | n | exact | r_final med (p25–p75) | early-stop frac | fixed-by-8 | median k* | post-freeze drift |
|---|---|---|---|---|---|---|---|
| 11–50 | 136 | 16 | 0.089 (0.027–0.193) | 44.9% | 5.1% | 20 | 1.20% |
| 51–256 | 80 | 0 | 0.108 (0.055–0.141) | 12.5% | 0.0% | 52 | 2.05% |

- Long spans end LESS converged (r_final 0.108 vs 0.089), almost never
  stabilize early (12.5% vs 44.9%), stabilize far later (median k* 52
  vs 20), and show 1.7x more post-freeze argmax drift (2.05% of
  positions vs 1.20% — the model's own later beliefs disagree with its
  commitments: the exposure-bias-style signature).
- Median residual trajectory, 51–256: 0.150@k4 → 0.130@8 → 0.114@16 →
  0.096@32 → 0.087@k48 — a slow ~1.7x decay with no floor by schedule
  end; 11–50 oscillates 0.054–0.146 without a clean decay either
  (composition-shift caveat: curve points condition on n_fwd ≥ k).
- Reading for S1: the long-span regime is precisely the unconverged,
  late-stabilizing, drift-carrying one — the paper's claim (recurrent
  carry + FPF convert depth into consistency where parallel token
  updates are inconsistent) targets exactly the bucket where S0 finds
  no stability and no correctness. AUROC within 51–256 is undefined
  (0 exact rows, banked and paradigm-wide).

## Steps=32 leg (secondary — depth trend + S1 NFE context)

Validity: exact = **15/216 = 0.0694 = the banked MD@32 anchor exactly**
(both legs reproduce the banked run bit-for-bit at the count level).

| readout | steps=32 | steps=64 |
|---|---|---|
| AUROC (positive = incorrect) | **0.897** [0.819–0.962] | 0.860 [0.741–0.952] |
| early-stoppable rows (as-run) | **24.5%** (still > 20% bar) | 32.9% |
| strict early-stop | 22.7% | 28.2% |
| fixed-by-8 | 3.2% | 3.2% |
| 11–50: r_final med / early-stop / median k* | 0.126 / 36.8% / 18 | 0.089 / 44.9% / 20 |
| 51–256: r_final med / early-stop / median k* | **0.200 / 3.8% / 28** | 0.108 / 12.5% / 52 |

Depth trend (the S1-relevant reading): long spans USE the extra
schedule depth — doubling 32→64 halves their final residual (0.200 →
0.108), triples the early-stoppable share (3.8% → 12.5%) and cuts
post-freeze drift (3.45% → 2.05%), while short spans are near-saturated
at 32. I.e. at the S1 verdict's anchor NFE the 51–256 bucket is still
CONVERGING, not converged — the "≤2× anchor NFE" budget in the S1
verdict bars is a binding constraint precisely where the bucket needs
to lift off, and the residual instrument now has a two-depth baseline
to measure whether recurrent carry buys the same convergence at fewer
passes. The AUROC is stable/slightly better at the shallower depth —
the correctness signal does not need deep schedules to separate.

## Caveats

1. n=16 positives (all in 11–50): CI is wide-ish (0.74–0.95); the
   AUROC point estimate is robust to the bootstrap but the exact tail
   (r_final 0.11–0.16 on rows 72/117) caps hard-threshold separation.
2. Abstain operating points are post-hoc on the same 216 rows.
3. CPU fp32 replay vs banked GPU fp32: committed outputs matched the
   banked anchor exactly (16/216), so reduction-order drift did not
   move any row here; it could on other checkpoints.
4. The early-stop fraction uses the as-run schedule end (n_fwd); the
   nominal-steps variant (k* ≤ steps−2) is 99.1% and vacuous for
   spans < 64 tokens (they exit early by construction) — reported
   only to close the definition.
5. r_open (open-positions-only residual) was recorded per step
   (secondary definition) but not used in readouts; data persisted.

## Artifacts

- Script: `experiments/training/poc_diff/x5_s0_residual.py`
  (`run` replays; `analyze` emits `analysis_steps{K}.json`)
- Per-row residual data: `results_x5_s0/residuals_steps64.jsonl`
  (216 rows: per-step r_all/r_open, conf trajectory, k*, committed
  ids, texts, metrics) + `residuals_steps32.jsonl` + analysis JSONs
- Log: `/tmp/x5_s0_full.log`; board posts 2026-09-06T01:52 → verdict
- Run metadata: taskset 8-15 → moved to 16-23 at T+12min (measured
  quietest; 8-15 went battery-saturated), nice 19, 8 torch threads,
  zero CUDA contexts. ~2h05m for the 64 leg (contended box).

## Verdict + S1 recommendation

**S0 PASSES BOTH BARS — signal + free-latency lever exist on the banked
head.** For the S1 GO decision: S1's value proposition shifts from
"create the confidence signal" to (a) push AUROC 0.86 → paper's 1.00
class, (b) create EARLY fixation (3.2% → 97.5%-class by k≤8) where the
real latency prize is, and (c) the pre-registered primary: lift the
51–256 bucket off 0.000 exact. S0 banked a baseline for all three
 readouts on the same instrument (re-measurable post-FPF verbatim).
Recommendation: **S1 deserves the GO** — with the S0 caveat attached
that the abstain signal alone is already usable WITHOUT recurrence, so
S1's kill bars stay exactly as pre-registered (exact > 0.0741 at ≤2x
NFE, 51–256 lift-off; 11–50 regression kill unchanged).
