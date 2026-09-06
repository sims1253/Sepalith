# V1d — blind pairwise preference: calibration results (2026-09-05)

Design: `docs/research/2026-09-02-eval-strategy-v2.md` §3 V1d + §4 (additive
eval-v2 column; no pre-registered verdict rule touched). Instrument:
`experiments/eval/pairwise_pref.py` (+ `test_pairwise_pref.py`).
Per-call rows: `results_pairwise_pref_{anchor_gt_corrupt,v8_2_vs_base,
v7_vs_rl_v2c}.jsonl` (+ `.done.jsonl` resume sidecars); aggregate:
`results_pairwise_pref_analysis.json`; run log `pairwise_pref.run.log`.

Correction (2026-09-06): ties-half now gives each model half credit for
order flips as well as explicit ties, as the design specifies. The saved
counts give v8_2 0.8311 (previously 0.8041) and v7 0.4122 (previously
0.3615). Thus rl_v2c scores 0.5878; the original table's 0.639 matches the
complement of v7's defective score. Counts, strict win rates, sign-test
p-values, and Wilson intervals are unchanged. These intervals and p-values
are historical, pointwise estimates without adjustment for shared
trajectories. This correction uses saved counts only; no new judgments or
trajectory-cluster correction were made.

## 1. Setup

- Inputs (EXISTING artifacts, zero new serving): the V1a episode-metrics
  calibration runs, which persist per trajectory point the exact completion
  prompt and each arm's raw proposal at IDENTICAL cursor states (paired by
  traj key + variant + t_ms + ctx window):
  - v8_2 vs base — `/mnt/h/sepalith/runs/episode_judged_{base,sft_v8_2}.jsonl`
    (V1a live leg, trajectories_v2, n=40 traj; 370 usable common points).
  - v7 vs rl_v2c — `/mnt/h/sepalith/datasets/sim_trajectories_v1/
    judged_{v7,rl_v2c}.jsonl` (V1a banked leg, n=100 traj; 507 usable).
  - anchor gt source — the sft_v8_2 file's typing points with non-empty gt
    (173/228 corruptible).
- Candidates = EXTENSION-FAITHFUL ghost text: `parse_prediction()`
  (eval_noop_fp.py, the extension's parsePrediction line-for-line) applied to
  the persisted raw proposal — `>>>>>>>` stop, marker-line filter, triple-line
  collapse. A point enters the pool only if BOTH arms parse non-empty
  (abstention discipline is V1a/noopFP's column, not V1d's; scope note §5).
- Sample: n=150/pair (mission default — the design doc gives no n guidance),
  proportional typing/noop stratification, seed 3407, largest-remainder
  allocation; anchor n=50.
- Blindness: judge sees only session goal + code context + the two candidates
  under generic slot labels A/B. A pre-spend sentinel (`check_blindness`)
  refuses any render containing identity/origin/outcome strings (model names,
  paths, decision enums); identity mapping is persisted per row for analysis
  only. Cost of the conservative gate: 14/704 calls self-refused on
  natural-text false positives ("lora", "adapter" as substrings of R source)
  -> 1 anchor + 2+2 pair points dropped (n=49/148/148 judged).
- Position debias: every point judged TWICE, orders swapped; first-order slot
  assignment is a seed-locked coin flip. A model WINS a point only if picked
  in BOTH orders; tie-in-either-order or order-flip = tie (chosen convention;
  secondary ties-half metric credits ties 1/2).
- Judge: gemini-3.7-flash-low via the `agy` CLI (`cases/backends.py`
  AgyBackend, 0.5s pacing) — one of the doc's panel_judge backends
  (gemini/muse/ox). Seat chosen per board coordination: zai/glm-5.3 was
  consumed by other agents (tu2-prep solve pass, H1 proposer) and muse is
  H1's fallback. gemini-3.7-flash passed glm-5.3's own three-gate judge
  calibration 120/120 with balanced agreement (no directional bias)
  [`docs/research/judge-calibration-gemini-opus.md`]. Single judge, not a
  3-panel majority (2 calls/point budget); panel swap is a one-flag change.
- Spend: 690 live calls, 0 unparsed, 0 429/transport errors (backend stats
  per leg in the run log), mean latency 7.3-7.7 s/call, ~86 min wall.
  Tokens are chars/4 ESTIMATES (agy CLI reports no usage): ~0.97M in,
  ~31k out.

## 2. Anchor — GT vs corrupted twin (n=49 points x 2 orders)

| wins GT | wins corrupted | ties | flips | win-rate (strict) | ties-half | first-slot rate |
|---|---|---|---|---|---|---|
| 43 | 0 | 6 | 0 | 0.878 [0.758, 0.943] | 0.939 | 0.517 |

The corrupted twin NEVER wins and never survives an order flip; the 6 ties
are one-order ties / double ties on word-swap corruptions that are
preference-neutral (e.g. two arbitrary variable names swapped — "identical
logic, differing only by arbitrary variable", per the judge). Doc expected
"~100%": 0.878 strict / 0.939 ties-half with a zero-loss record — the
shortfall is corruption sharpness, not judge blindness. PASS.

## 3. Calibration pair 1 — v8_2 vs base (n=148)

| wins v8_2 | wins base | ties | flips | v8_2 win-rate | ties-half | sign p |
|---|---|---|---|---|---|---|
| 103 | 5 | 32 | 8 | **0.696 [0.618, 0.764]** | 0.831 | 7.2e-25 |

Per label: typing 67-3 (n=91), noop 36-2 (n=57). First-slot pick rate 0.559
(mild); the both-orders rule withholds wins on order flips. The original
pointwise Wilson lower bound is 0.618, above 0.5. **CALIBRATION PASS** was
the historical readout; the observed preference follows the V1a anchor
(v8_2 8 accepts vs 0; intent-suite norm 0.117 -> 0.685). Notably the win-rate
lands at the intent-suite mean of v8_2 itself (0.685) — consistent with the
doc's shaping reference (audit 0.170 -> 0.596 dropout-vs-v8 intent pair)
without being pinned to it. Judge flavor: base candidates are "garbage
prompt metadata"/marker soup; v8_2 "correctly continues the function argument
list leading into the suffix".

## 4. Calibration pair 2 — v7 vs rl_v2c (n=148)

| wins v7 | wins rl_v2c | ties | flips | v7 win-rate | rl_v2c ties-half | sign p |
|---|---|---|---|---|---|---|
| 17 | 43 | 73 | 15 | 0.115 [0.073, 0.176] | 0.588 | 0.0011 |

Per label: typing 33 rl_v2c - 16 v7 (n=107); noop 10 - 1 (n=41).
First-slot rate 0.587.

DIRECTION READOUT — does NOT follow the intent suite (v7 0.809 vs rl_v2c
0.511 norm); DOES follow the noise-discipline axis (noopFP 0.706 vs 0.466;
V1a fp_rate 0.991 vs 0.437): among completions BOTH models actually show,
the judge more often prefers rl_v2c's. v7's raw proposals leak
edit markers and ramble (visible in the persisted rows); the extension-faithful
parse rescues format but not substance. This is a genuine divergence between
the two anchors (intent-following vs proposal quality), not an instrument
failure — but it means V1d is NOT a redundant re-measurement of the intent
suite. Explicit ties account for 73/148 points (49%); another 15/148 (10%)
flip with order. Both receive half credit in ties-half. A tie can mean
equally useful or equally unwanted proposals, and a flip records order
sensitivity. These counts alone do not measure how many completions are
keepable. Reported as an additive column, with no kill rule.

## 5. Position bias

| set | flip rate | first-slot pick rate |
|---|---|---|
| anchor | 0.000 (0/49) | 0.517 |
| v8_2-vs-base | 0.054 (8/148) | 0.559 |
| v7-vs-rl_v2c | 0.101 (15/148) | 0.587 |

First-slot picks increase across these three sets. Requiring agreement
across both orders withholds wins on flips; it does not establish that all
position bias has been removed. All sets fall within the historical
0.35-0.65 sanity band.

## 6. Honest caveats

- Repeated cursor points share trajectories. The reported Wilson intervals
  and sign tests treat points as independent and remain unadjusted historical
  statistics. A trajectory-level analysis is needed before using their
  uncertainty estimates for new adoption decisions.
- Scope: points where BOTH arms proposed. A model that (correctly) stays
  silent never enters — restraint is scored by V1a/noopFP, preference here is
  conditional on a proposal existing. The v7-vs-rl_v2c readout in particular
  must be read beside rl_v2c's own fp_rate, not instead of it.
- noop points are lesser-evil judgments (both completions are unwanted); the
  noop subsample is reported separately and is small for rl_v2c (n=41).
- Single judge (gemini-3.7-flash-low), single seed, product-shaped prompt —
  proxy-preference, not user joy (design doc §5); the extension's real
  telemetry remains the eventual ground truth.
- Sentence-level corruption is deterministic but variable in sharpness
  (arg-swap sharp, word-swap sometimes neutral) -> anchor ceiling ~0.94.
- est tokens are chars/4; agy reports no usage. Call counts exact.

## 7. Verdict

V1d EARNS its place as a standing eval-v2 battery column:

1. Anchor passes with a zero-loss record (43-0-6).
2. The known-good pair separates in the expected direction (103 wins vs 5).
   The original pointwise CI and p-value remain in the table; uncertainty
   after adjustment for shared trajectories has not been established.
3. It is ADDITIVE, not redundant: pair 2 splits the intent-suite and
   FP-discipline axes apart, which no existing battery member does.
4. Cheap and retroactive: judges EXISTING persisted outputs (690 calls /
   ~1M est tokens for two 150-point pairs + anchor; any arm with persisted
   episode rows is re-judgeable with `--pair`-style specs).

Battery integration note for the queue manager: the instrument is pair-spec
driven (`PAIR_SPECS`); future arms need episode_judged-style per-point rows
(judge_loop already persists them) or any paired raw-output store.
