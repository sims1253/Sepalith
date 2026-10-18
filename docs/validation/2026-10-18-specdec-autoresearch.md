# Speculative decoding autoresearch — lossless CPU spec-dec for Sepalith

**Mission:** find the fastest lossless speculative-decoding setup for the arm-A
model on the real PRM03 workload (CPU llama.cpp b10453), measure end-to-end
honestly, deliver setup + evidence. Autoresearch session `.auto/` at the repo
root; all artifacts under
`/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/`.

## Headline

The fastest measured configuration is **`ngram-simple` with lookup n=2,
draft m=4** — warm (post-warm-up-flow, the PR-#66 product flow) end-to-end
**1.29–1.44× median** (221-trace full set: 1.294×; 60-trace subset: 1.40–1.44×,
two runs), **p95 1.46×, pooled 1.57×**; decode-only median 1.33×; TTFT
unchanged (~82 ms). Cold (single-shot) end-to-end is only **1.06×** because
prefill dominates (11.8 s median cold prefill vs 2.0 s decode under session
load; see §2).

**But no configuration passes the promotion gate as written.** Two independent
reasons:

1. **The 100 % token-identity requirement is unattainable at b10453.** Any
   speculative drafting — even one drafted token per step — changes the target
   model's batch shape (k+1 rows instead of 1), which changes GEMM reduction
   order enough to flip near-tie argmax decisions. Measured flip rates:
   5–13 % of requests across every n-gram mode and depth probed
   (§5). The baseline itself is perfectly deterministic under identical
   request shape (warm-repeat parity = 1.000), so the flips are caused by the
   verify batch, not by the rig. Per the brief this is written up with a
   minimal reproduction (`experiments/serving/specdec/parity_repro.py`,
   artifacts `…/runs/parity_repro_cases.txt`); it is not worked around.
2. **Cold end-to-end is prefill-bound (1.06×)** — speculation can only touch
   the decode share. With PR #66's warm-up flow in the product, the
   request-at-pause time is exactly the warm number this report leads with.

Calibration for decision-making (§5.3): the KV-cache warm-up flow the product
already ships (PR #66) drifts 0.851 token-identity vs cold greedy on the same
frozen set, and Q8_0 quantization itself matches bf16 greedy on only 66.8 % of
DEV250 (quant gate receipt, 2026-10-17). Speculation's 0.87–0.88 is the same
class of near-tie perturbation, one order of magnitude smaller than the
quantization step the product already accepts.

## 1. Setup

| Item | Value |
|---|---|
| Target | `arm-a-Q8_0.gguf` (sha256 `44b01828…70de290fe`, read-only quant-work copy) |
| Engine | llama.cpp **b10453 CPU build** (`~/.local/opt/llama.cpp-b10453/build-cpu`), `llama-server`, `--parallel 1`, `-c 16384`, `--metrics`, `nice -n 10`, `CUDA_VISIBLE_DEVICES=""` |
| Threads | 8 (default) and 4 (laptop band) |
| Traces | `experiments/serving/specdec/traces.jsonl` — 221 TRAIN rows, family-balanced (7 families × short/long bands), even target-length spacing, seed 20261018; manifest sha-pins every row (`manifest.json`) |
| Request | token-id prompts (`input_ids[:target_start]`, BOS included), greedy (`temperature 0`, `top_k 1`), `n_predict 1024`, server-native EOG stop (token 1; no trace ever emitted 130073), `cache_prompt` per flow |
| Warm flow | warm-up request (`n_predict 0`) then timed request — mirrors the extension's post-PR-#66 pause flow; per-request `cache_n` confirms full-prefix reuse (cache_frac = 1.0) |
| Cold flow | timed request with `cache_prompt false` (full prefill) |
| Pairing | baseline and candidate servers run in the same session, interleaved per trace, alternating order — host-load drift hits both arms |
| Reference outputs | 221 greedy outputs at cap 1024 (`reference_outputs.json`), generated once, sha-committed |
| Host | Ryzen 9 5900X (12C/24T) WSL2; other missions ran concurrently (recorded load 8–23); all speedups are same-session paired ratios |

Bench: `experiments/serving/specdec/bench_spec.py` (adapted from the S1 rig
and the prefill mission's SSE client; per-request JSONL + summaries under
`…/specdec-20261017/runs/`). Quality gates: `python3 scripts/check_quality.py`
clean; `ruff`/`ty` clean on `experiments/serving/specdec`.

## 2. Prefill/decode split (why cold e2e barely moves)

From the full-set cold pass (221 traces, cap 1024, load 15–19):

- cold prefill median **11.8 s** (long prompts, contended host; the prefill
  mission measured 2.5–8.2 s cold on a quieter host — same conclusion),
- decode median **2.0 s** (baseline), TTFT(warm) **83 ms**.

So at pause time in the product (warm flow) speculation touches ~96 % of the
remaining latency; in a cold single-shot world it touches ~15 %.

## 3. Method: exact offline simulation, then paired server runs

`sim_draft.py` contains bit-faithful ports of the b10453 draft algorithms
(`ngram-simple` incl. its `copy_max < size_n` quirk; `ngram-mod` incl. chunked
adds and the low-acceptance reset) plus a region-copy oracle (direction 2 of
the brief). Validated: on matched traces the simulator reproduces the
server's per-request draft/accept counters **exactly** (e.g. 15/24, 3/16,
17/40 identical to the token).

Findings that shaped the sweep:

- **b10453 flag semantics**: n-gram draft depth comes from each mode's own
  knobs (`size_m` for simple/map-k/k4v, `n_max`/`n_min` for mod), *not*
  `--spec-draft-n-max`. `ngram-simple` requires `size_m ≥ size_n` or it never
  drafts at all. `ngram-mod` defaults (n_min 48 / n_max 64) are multi-user
  serving defaults and never draft on 26-token outputs.
- **Region-copy oracle is dominated** (pooled 1.11 vs 2.3 for n-gram): the
  output copies from the *whole* prompt context (file prefix, references,
  contract text), not just the CURRENT region, and generic unigram/bigram
  lookup captures it. The brief's direction 2 (C++ drafter patch) is closed
  as unjustified by evidence.
- **CPU verify-cost law** (`llama-batched-bench`, TG t/s at batch B):
  B=1: 7.6, B=2: 15.6, B=4: 40.9, B=9: 37.3, B=17: 47.0 — a verify step with
  k+1 rows costs ~2× a single-token step at k=8. Wall speedup therefore
  tracks **short, high-precision drafts**, not tok/step. The tok/step-optimal
  configs (n=1, m=8–24; sim ceiling 2.2–2.3×) are wall-optimal nowhere.

## 4. Configuration sweep (60-trace subset, cap 256, paired, load recorded)

| config | flags | median warm | pooled | accept (tok/step @ rate) | mean draft len |
|---|---|---:|---:|---|---:|
| baseline clone | — | 1.04× | — | — | — |
| n1m8 | `--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 8` | 1.04–1.10× | 0.99 | 2.4 @ 0.14 | ~5.6 |
| n1m4 | n=1, m=4 | 1.17× | 1.05 | 1.68 @ 0.26 | 2.63 |
| n1m6 | n=1, m=6 | 1.24× | 0.98 | 1.76 @ 0.20 | 3.78 |
| n2m2 | n=2, m=2 | 1.17× | 1.12 | 1.42 @ 0.57 | 0.73 |
| mapk n2m4 | `--spec-type ngram-map-k` n=2 m=4 | 1.40× | 1.21 | 1.32 @ 0.54 | 0.60 |
| mapk4v n2m4 | `--spec-type ngram-map-k4v` n=2 m=4 | 1.18× | 1.17 | 1.25 @ 0.56 | 0.45 |
| **n2m4 (champion)** | `--spec-type ngram-simple --spec-ngram-simple-size-n 2 --spec-ngram-simple-size-m 4` | **1.40–1.44×** (CI95 1.15–1.55, two runs) | **1.24–1.25×** | **1.52 @ 0.42** | **1.26** |

`ngram-mod` (all n_match/n_min/n_max grids) and n≥3 were closed by
simulation + spot server runs (mod never fires on short outputs at sane
settings; its best pooled sim ≈1.5 with median 1.00). `ngram-cache` was
closed by bound rather than a dedicated run: its draft depth is fixed at 8,
where the measured wall ceiling is 1.04–1.24× (§3 cost law), and its
conservative firing thresholds cannot exceed ngram-simple's draft
frequency. Multi-type stacking (`ngram-simple,ngram-map-k`) was run and
closed: the fallback never fires (simple n2m4 drafts on every step;
acceptance counters bit-identical), 1.379× vs 1.40–1.44×.

Acceptance counters reproduce bit-exactly across runs (1.525 tok/step,
0.417 rate, both n2m4 sessions) — drafting and greedy acceptance are
deterministic; only wall time varies with load.

## 5. Parity (the gate-blocker), written up per the brief's hard-stop rule

### 5.1 Controls

- **Baseline warm-repeat parity = 1.000** (identical request repeated on the
  baseline server, every session): the server is fully deterministic for a
  fixed request+batch shape.
- **Baseline warm-vs-cold parity = 0.851** (221 traces): ordinary KV-cache
  reuse alone (the warm-up flow the product ships since PR #66) flips 15 % of
  greedy outputs vs cold processing.

### 5.2 Spec arms (same-session baseline vs candidate, warm flow)

| config | flips | parity | divergence positions (subset) |
|---|---|---|---|
| n1m1 (1 drafted token/step!) | 5/60 | 0.917 | 1, 5, 8, 29, 61 |
| n2m2 | 8/60 | 0.867 | 1, 21, 41, 61, 62, 65, 100, 108 |
| n2m3 | 8/60 | 0.867 | same set |
| n3m3 | 5/60 | 0.917 | 1, 29, 61, 65, 67 |
| n2m4 (warm, 60-trace) | 7/60 | 0.883 | early (3–16) |
| n2m4 (warm, 221-trace) | 29/221 | 0.869 | — |
| n2m4 (cold-vs-cold, 221) | — | 0.873 | — |
| n1m8 | 3/56 | 0.946 | — |

Any drafting flips ~5–13 % of requests; the rate scales with the number of
drafted steps, not depth. There is **no lossless n-gram configuration** on
this build. Moreover, *which* requests flip is **environment-dependent**:
the 4-thread and 8-thread full-set runs flip 31 vs 29 of 221 requests with
only 20 in common — the flipped set is a per-environment numerics coin-flip
on near-ties (thread partition changes the GEMM reduction order), while the
~13 % rate is stable. A fixed "spec changes these requests" list does not
exist.

### 5.3 Minimal reproduction + mechanism

`parity_repro.py` isolates cases and captures the baseline top-3 logprobs at
the divergence position (artifacts: `runs/parity_repro_cases.txt`):

- `0d318a5c…` (finish_block), diverge @61: baseline top-2 = "ling" −0.6881 /
  "led" −0.7010 (Δ 0.013 logprob) — spec arm picks "led", baseline "ling";
  both are valid English suffix continuations; outputs stay the same length.
- `37c1d27e…` (format_propagation), diverge @1: " co" −0.9156 / " c" −0.9192
  (Δ 0.0036 logprob) — a 0.4 % probability gap decides the token.

Mechanism: the verify batch evaluates k+1 positions in one `llama_decode`;
the CPU backend's batched matmuls use a different reduction order than the
single-row path, so logits differ in the last ulps and near-tie argmaxes can
flip. The same mechanism explains the 0.851 warm-vs-cold drift (prefill batch
shape differs) and — at much larger magnitude — the 66.8 % token identity of
Q8_0 vs bf16 on DEV250 (quant gate receipt). **Flipped ≠ wrong**: these are
ties by construction; but the gate as written demands identity, so this is a
decision point, not something to work around.

### 5.4 Consequence for draft-model directions

DSpark / trained drafters (brief directions 3–4) face the *same* verify-batch
numerics; a better drafter cannot restore bitwise parity. On CPU the draft
model also adds forward passes per step (the 0.8B stand-in already lost
end-to-end under load in the September smoke). Direction 3 was therefore not
pursued beyond this analysis; direction 4 (training) is moot until a
determinism fix lands in llama.cpp (see proposals).

## 6. Full-set gate table (champion n2m4, 221 traces, cap 1024, cold+warm)

Two independent paired repetitions: rep 1 = two merged sessions (147+74
traces, host load 15–19); rep 2 = one session (load ~12). Per-request JSONL
retained for both under `runs/`.

| metric | baseline | n2m4 | ratio rep 1 | ratio rep 2 |
|---|---:|---:|---:|---:|
| warm e2e median | 2104 / 2308 ms | 1625 / 1628 ms | **1.294×** | **1.418×** (CI95 lo 1.265) |
| warm e2e p95 | 24545 / 23846 ms | 16837 / 17021 ms | **1.458×** | **1.401×** |
| warm e2e pooled (Σt) | — | — | **1.567×** | **1.577×** |
| decode median | 2027 / 2202 ms | 1527 / 1529 ms | 1.328× | 1.440× |
| decode tok/s (median) | ~12–12.8 | ~15.0 | 1.18× | 1.25× |
| TTFT median | 83 / 88 ms | 82 / 88 ms | 1.01× | 1.00× |
| cold e2e median | 17748 / 15720 ms | 16788 / 15181 ms | 1.057× | 1.035× |
| acceptance | — | — | 1.52 tok/step @ 0.42 (60-sub) | 1.81 tok/step @ 0.47 |
| parity (warm, vs same-session base) | — | 192/221 | 0.869 | **0.869 (same 192 — deterministic flip set)** |
| baseline warm-vs-cold (control) | — | — | 0.851 | 0.8507 |
| baseline warm-repeat (control) | — | — | 1.000 | 1.000 |

Acceptance on the full set is higher than on the 60-trace sweep subset
(1.81 vs 1.52 tok/step): the full set carries more long, copy-heavy outputs.

By band (warm e2e median): short ≤1k 1.34× / long ≥1k 1.36×; edit 1.35× /
no-op 1.15×. By family: na_rm 1.58×, format 1.28×, pipe 1.21×, roxygen
1.17×, no_op 1.15×, rename 1.13×, finish_block 0.96× (median trace; its long
tail is what drives pooled to 1.57×).

**4-thread band** (laptop-representative; 60-trace subset cap 256: 1.233×
median, CI95 lo 1.09; full 221-trace set cap 1024 at load ~4: **1.276×**
median, CI95 lo 1.198, decode 1.303×, TTFT unchanged, p95 1.099× — the tail
improves less at 4 threads). Acceptance transfers (1.83 tok/step @ 0.47).

## 7. DEV75 final panel (the single recorded DEV use)

Panel `corrected-dev75-v1` (sha `7c144bcd…`, same panel the prefill mission
recorded), converted with `build_dev75_traces.py` (BOS + PRM03 render,
75 traces, median 1169 prompt tokens). One paired session, cap 1024,
cold+warm, speed only — recorded in `runs/dev75-final.log` and
`runs/dev_use_record.json`.

| metric | baseline | n2m4 | ratio |
|---|---:|---:|---:|
| warm e2e median | 2530 ms | 1728 ms | **1.464×** (bootstrap 95 % CI ≥ 1.227) |
| warm e2e p95 | 34526 ms | 19515 ms | **1.77×** |
| warm e2e pooled | — | — | 1.306× |
| decode median | 2446 ms | 1646 ms | 1.486× |
| decode tok/s (median) | 12.8 | 15.5 | 1.21× |
| TTFT median | 83 ms | 82 ms | 1.01× |
| cold e2e median | 16945 ms | 17002 ms | 1.003× |
| acceptance | — | — | 1.725 tok/step @ 0.484 |
| parity (vs same-session base) | — | 66/75 | 0.880 |
| baseline warm-vs-cold (control) | — | 65/75 | 0.867 |
| baseline warm-repeat (control) | — | — | 1.000 |

Acceptance on real DEV prompts is *higher* than on the TRAIN subset (1.73 vs
1.53 tok/step): real edit outputs copy more. The warm median speedup on DEV75
crosses the 1.4× gate level; parity and cold e2e do not (same reasons as
TRAIN).

## 8. Recommendation

**Conditional recommendation** — the decision belongs to the user because the
gate's lossless clause conflicts with the measured numerics:

- If near-tie drift of the warm-up-flow class is acceptable (it is already
  shipping via PR #66 at 0.851; spec adds the same mechanism at 0.87–0.88),
  adopt: `llama-server … --spec-type ngram-simple --spec-ngram-simple-size-n 2
  --spec-ngram-simple-size-m 4` — warm e2e median 1.29–1.44×, p95 1.46×,
  decode 1.33×, TTFT unchanged, zero extra memory, no draft model.
- If strict token identity vs the no-spec path is required, do not enable
  speculation at b10453; the parity writeup (§5) is the deliverable, and the
  unlock is a llama.cpp determinism fix (proposal 1), not configuration.

Q4 cross-check: see §9 (acceptance transfers across quants at n=2: the n-gram
drafter reads token ids, not weights; the verifier's near-ties move, so the
flip set changes but the class of drift is the same).

## 9. Quant cross-check (imat-Q4_K_M, provisional tier)

Ran the champion unchanged on `arm-a-imat-Q4_K_M.gguf` (60-trace sweep
subset, cap 256, paired; note: **no Q4 tier had passed the C09 quant gate at
report time** — this is a provisional cross-check, not a shipping
recommendation):

| metric | Q4 base | Q4 + n2m4 | ratio |
|---|---:|---:|---:|
| warm e2e median | 1334 ms | 988 ms | **1.350×** (CI95 lo 1.15) |
| warm e2e pooled | — | — | 1.162× |
| decode tok/s | 20.2 | 23.2 | 1.15× |
| acceptance | — | — | 1.475 tok/step @ 0.376 |
| parity vs same-session base | — | 54/60 | 0.900 |

The n-gram drafter reads token ids, not weights, so the *drafting* is
quant-independent; the quantized verifier flips a different but
same-magnitude set of near-ties (0.90 vs 0.88–0.92 on Q8). Wall speedup is
slightly lower on Q4 (1.35× vs 1.40–1.44×) because Q4's baseline decode is
already faster (20.2 vs ~12.8 tok/s at comparable load), shrinking the head
speculation can recover.

## 10. Proposals needing user decisions

1. **llama.cpp determinism / upgrade.** The parity blocker is b10453 CPU
   batch-verify numerics. (Ruled out at b10453: `-fa on` — bit-identical
   outputs and speed-neutral on this CPU build, so the attention flag is
   not a lever; kernel-path changes must come from a different build.) Proposals: (a) ask upstream / patch so the greedy
   verification path uses the same matmul path as single-token decode (e.g.
   force per-row vec-mat for verify batches ≤ some size), making speculation
   bitwise-lossless; (b) evaluate a newer llama.cpp — release notes/PRs
   around spec-decode acceptance and CPU batched GEMM may have changed both
   the verify cost curve (§3) and determinism. Evidence for the current
   behavior: §5. I did not upgrade (pinned by the brief).
2. **DSpark / drafter training: not recommended now.** Same parity blocker;
   CPU draft-model cost eats the gain (September smoke + §3 cost law); the
   130k-vocab output layer dominates small-drafter cost (wiki slimspec). If
   proposal 1 lands and the gate reopens with headroom, the trained-drafter
   direction becomes worth a TRAIN-only night-queue job (≤2 GPU-hours, format
   `docs/training/plan-20260927/c09/jobs/*.json`).
3. **If spec ships despite drift:** re-run the C09-style quality gate
   (exact-match/no-fp on DEV250) with speculation enabled, since flips land
   on near-ties inside real edits; the quant gate's token-identity record
   (66.8 % for Q8 vs bf16) is the precedent for accepting a measured drift.

## 11. DEV usage log

| date | panel | use | configs |
|---|---|---|---|
| 2026-10-18 | corrected-dev75-v1 (sha 7c144bcd…) | final speed report, single paired session | baseline + ngram-simple n2m4 |

No other DEV/final data was read or measured. TRAIN-only traces were used for
all tuning; the frozen manifest pins them.
