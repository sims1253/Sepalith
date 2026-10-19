# Autoresearch: lossless speculative decoding for Sepalith (CPU llama.cpp)

## Objective
Find the fastest **lossless** speculative-decoding setup for Sepalith's real
workload (PRM03 edit prompts, greedy, batch 1, Q8_0 arm A first) on the pinned
CPU llama.cpp build b10453. Lossless = token-id-identical outputs vs a paired
no-speculation baseline for 100% of measured requests. Full brief:
`/home/m0hawk/sandbox/sepalith-specdec-autoresearch-brief.md`.

Key context: prefill mission (PR #66) landed a warm-up request flow that makes
the product's pause→suggestion path decode-bound (~64 ms fresh prefill +
decode). So the loop optimizes **warm end-to-end time** (prompt already in KV
cache, mirroring the post-PR66 flow), with cold end-to-end reported for the
final gate table.

## Metrics
- **Primary**: `speedup_warm` (unitless, HIGHER is better) — median warm e2e of
  the paired same-session baseline ÷ median warm e2e of the candidate arm.
  Paired/interleaved per trace so host-load drift hits both arms.
- **Secondary**: `e2e_warm_ms`, `decode_ms`, `tok_s`, `ttft_ms`, `p95_warm_ms`,
  `accept_tok_per_step`, `accept_rate`, `parity` (must be 1.0 for a config to
  be promotable), `parity_ref` (vs frozen reference outputs), `load1`.
- Gate extras (final runs only): bootstrap 95% CI of speedup (LB > 1.0, target
  ≥1.4×), no p95 regression, cold e2e, per-band/per-family splits, 4-thread
  numbers, DEV75 once for ≤3 configs.

## How to Run
- Loop benchmark: `./.auto/measure.sh` → runs
  `experiments/serving/specdec/bench_spec.py --config .auto/candidate.json
  --n-predict 256 --n-traces 90` (sweep mode) and emits `METRIC name=value`.
  Write the candidate arm's server flags into `.auto/candidate.json`
  `{"name": ..., "server_args": [...]}`.
- Full verification: same script without `--n-predict/--n-traces` limits, plus
  `--cold` and `--reference experiments/serving/specdec/reference_outputs.json`,
  repeated ≥3×.
- Reference outputs (frozen baseline, product cap 1024):
  `experiments/serving/specdec/reference_outputs.json` (from clone-base run).
- Offline drafter simulator (no server): `experiments/serving/specdec/sim_draft.py`
  — exact ports of the b10453 n-gram draft algorithms + region-copy oracle,
  run over reference outputs. USE THIS to prune the sweep space.

## Files in Scope
- `experiments/serving/specdec/*` — bench, freeze, sim, configs, traces, manifest.
- `.auto/*` — session files.
- `docs/validation/2026-10-17-specdec-autoresearch.md` + `.json` — final report.
- Artifacts (logs, per-request jsonl): `/mnt/e/sepalith/campaign-20260915/serving-work/specdec-20261017/runs/`.

## Off Limits
- `~/.local/opt/llama.cpp-b10453` (read-only), other missions' servers/processes.
- Sealed final data, DEV sets (DEV75/DEV250: ONE final report use only, ≤3
  configs), old spec_traces, pools predating the TRAIN/DEV/final split.
- No CUDA ever (`CUDA_VISIBLE_DEVICES=""`), max 8 threads, `nice -n 10`.
- No training, no uploads, no writes into quant-work arm-a dir.
- Never modify the frozen traces.jsonl/manifest/reference outputs.

## Constraints
- `python3 scripts/check_quality.py` (+ `--research`) must stay clean.
- Parity < 1.0 disqualifies a config (log as failed experiment, not keep).
- Record host load; if load looks pathological (>~20), distrust the run.
- llama.cpp spec flag semantics (verified from b10453 source):
  - ngram modes' draft depth comes from their OWN knobs, NOT
    `--spec-draft-n-max`: simple/map-k/map-k4v → `size_m`; mod → `n_max`
    (min draft `n_min` or none). ngram-cache → fixed 8.
  - ngram-simple REQUIRES size_m ≥ size_n (else it never drafts at all —
    `copy_max < n_draft_min` quirk in ngram-map.cpp).
  - ngram-mod defaults n_min=48/n_max=64 (multi-user serving defaults) —
    useless for ~26-token outputs; n_min must be ≤ a few.
  - ngram-simple searches the MOST RECENT match only; the current region sits
    at the very end of the prompt, so long-n lookups often fail or copy the
    wrong context. Small n (2–4) simulates best.
- Prefill ≈ speculation-independent; decode is the target. Warm e2e ≈
  decode + ~60-100 ms.

## What's Been Tried
- (setup) Old `spec_bench.py` was built for the Qwen base/PSM prompts — new
  `bench_spec.py` written for PRM03 ids, cap 1024, both stop ids via server
  EOG (works: stop_type=eos on token 1; targets end with token 1).
- Smoke (4 traces): stream `tokens` arrive incrementally per SSE chunk (fixed
  parser). ngram-simple (12,8) drafts NOTHING (size_m < size_n quirk) —
  verified against b10453 source and reproduced by exact offline port.
- Offline sim (exact ports of ngram-simple/mod + region-copy oracle, 221 refs):
  sim == server counters bit-exact on matched traces. Region-copy oracle
  DOMINATED by ngram-simple everywhere (1.11 pooled vs 2.3) → brief direction 2
  (C++ patch) closed as unjustified.
- WALL-COST LAW (llama-batched-bench + runs): verify batch of k+1 costs real
  CPU — B=1 131ms/tok (slow mat-vec), B>=4 ~27ms/tok (GEMM path), i.e.
  c(9)~=2xc(1). Wall speedup tracks SHORT PRECISE drafts (high accept rate ×
  small batches), NOT tok/step. Deeper is worse after m~4.
- Sweep results (60-trace subset, cap 256, paired): n1m8 1.04×, n1m4 1.17×,
  n1m6 1.24×, n2m2 1.17×, mapk-n2m4 1.40×, mapk4v-n2m4 1.18×,
  **n2m4 1.40-1.44× median (pooled 1.24-1.25)** — champion.
- Full-set gate run n2m4 @ cap 1024, 221 traces, cold+warm (load 15-19):
  warm median 1.294×, p95 1.458×, pooled 1.567×; cold 1.057× (prefill-bound:
  11.8s median cold prefill vs 2.0s decode); decode median 1.328×; TTFT ~82ms
  both arms. By family (warm median): na_rm 1.58, format 1.28, pipe 1.21,
  roxy 1.17, no_op 1.15, rename 1.13, finish_block 0.96 (median; tail helped).
- PARITY (the binding finding): baseline warm-repeat = 1.000 (server fully
  deterministic under identical batch shapes). ANY drafting flips near-ties:
  n1m1 5/60, n2m2 8/60, n2m3 8/60, n3m3 5/60, warm runs 7-8/60, n1m8 3/56.
  ~9-13% of requests diverge; divergence positions early (1-100).
  Calibration: baseline warm-vs-cold (KV-cache reuse alone) drifts 0.851 —
  the product's warm-up flow ALREADY accepts this drift class; Q8 quantization
  itself only matches bf16 greedy on 66.8% of DEV250 (quant gate receipt).
  Minimal repro: parity_repro.py + runs/parity_repro_cases.txt — e.g.
  "ling" -0.688 vs "led" -0.701 near-tie flips under a verify batch of 2.
  → NO lossless spec config exists at b10453 CPU; promotion gate (100%
  parity) unattainable. Writeup required, not a workaround.
- Multi-type fallback chain (simple,ngram-map-k): counters bit-identical to
  plain n2m4 — fallback NEVER fires (simple drafts every step). Stacking
  closed. ngram-cache closed by bound (fixed depth 8 = wall ceiling
  1.04-1.24x, cannot exceed simple's draft frequency).
- Full-set gate rep #2 (load ~12): warm median 1.418x (CI lo 1.265), pooled
  1.577x, decode 1.44x, acceptance 1.81 tok/step @0.472 (full set richer in
  long outputs than the 60-subset); parity 0.869 with the SAME 192/221
  matches as rep #1 (deterministic flip set). Two reps: 1.294x (load 15-19)
  and 1.418x (load 12).
- PR #68 open (not merged), checks green, comms registered (rzcode-specdec),
  DEV use recorded. MISSION DELIVERABLES COMPLETE; remaining loop = PR babysitting.
- Q6_K rep #3 + final synthesis (run #25): 1.408x; six full-set sessions
  (1326 paired requests): pooled 1.560x [1.464, 1.659], median-of-medians
  1.423x — CI entirely above the 1.4x gate level.
- Cascade prototype (run #24): direction 2 closed BY MEASUREMENT — patched
  fork (bigram->unigram cascade) hits +16%/+24% tok/step ceiling but wall
  1.443x ties stock n2m4 (accept 0.29, draft len 2.62 -> cost law eats it).
  Patch + build instructions kept in experiments/serving/specdec/patches/.
  Bench gained --server-bin for patched-binary paired runs.
- Q6_K full-set rep #2 (run #23): 1.438x (CI lo 1.33), pooled 1.542,
  acceptance bit-identical, same flip set. Ship tier: 2 full reps
  1.428-1.438x, abs median ~1000ms. ALL EVIDENCE CELLS NOW EVEN.
- Ship-tier full-set Q6_K (run #22): 1.428x median (CI lo 1.34), pooled
  1.537x, decode 1.46x, abs median 990ms. Extension spawn checked: no
  conflicts, drop-in. Run-artifact manifest (sha256) committed.
- Q4 x 4T cell + noop analysis (run #21): matrix 6/6 complete (Q4x4T
  1.266x, only p95 regression 0.97x, non-shippable tier). NO-OP finding:
  zero flips, zero [NO_EDIT]<->edit mode switches across all 3 full-set
  runs — product no-op behavior identical with spec on.
- 4T x Q6_K cell (run #20): 1.294x (CI lo 1.02), acceptance bit-identical
  to 8T Q6_K. Recommendation matrix complete: Q8 1.40-1.50/1.28, Q6_K
  1.55/1.29, Q4 1.35. 4T verify-cost penalty ~1.41x/step vs 1.26x at 8T;
  m=4 stays optimal at 4T by arithmetic.
- Q6_K cross-check (run #19): n2m4 warm median 1.549x (CI lo 1.26), decode
  1.60x, acceptance 1.517@0.401, parity 0.90. Cross-tier: Q8 1.40-1.50x /
  Q6_K 1.55x / Q4 1.35x — flags quant-robust; Q6_K is the likeliest ship
  tier (prefill mission decode finding).
- Mid-abort flow (run #18, raw-socket close 120ms): e2e 1.039x — aborted
  warm-ups DRAIN server-side (TTFT 8.8s both arms; prefill completes before
  the real request); decode still +1.55x; partial KV retention confirmed
  (2105/2359). Flow continuum fully mapped: 1.50x warm / 1.04x mid / 1.03x
  cold; speculation never hurts.
- Aborted-warmup flow (run #17): 700ms abort = complete-warmup endpoint
  (1.503x subset, best yet); 150ms = effectively-cold reuse with decode
  speedup intact (1.68x) and no regression. Continuum bracketed; spec
  never hurts. Instrument limit: true mid-prefill aborts need raw-socket
  close timing (prefill mission's probe owns that mechanics).
- Rep #3 (run #16, load ~8): 1.429x median (CI lo 1.33), pooled 1.578x,
  parity 192/221 again. Three full-set reps: 1.294/1.418/1.429 — >=3 reps
  satisfied; m-refinement closed by per-position acceptance mechanism.
- Flip QUALITY-NEUTRALITY (run #15, offline): spec exact-match vs TRAIN
  ground truth == baseline's exactly (158/221 at 8T AND 4T); every flipped
  trace had already diverged from target under baseline (0 ground-truth
  matches gained/lost); spec outputs at least as well-formed (terminal
  28/29 vs 24/29). Flips land only on already-diverged generations.
- Full-set 4T band (run #13, load ~4): warm median 1.276x (CI lo 1.198),
  decode 1.303x, p95 1.099x. NEW MECHANISM FINDING: flip SET is thread-
  dependent (4T 31 flips vs 8T 29, only 20 overlap) — per-environment
  numerics coin-flips, rate ~13% stable. Measurement matrix now complete:
  8T/4T x subset/full x Q8/Q4 + DEV75.
- Full-set rep #3 was INTERRUPTED at ~130/221 traces by user request (host
  needed for gaming) — partial per_request.jsonl retained in runs/20261018T*;
  two complete full-set reps + DEV75 + subset doubles already satisfy the
  >=3 repetitions requirement. All specdec processes killed 2026-10-19.
- Footgun learned: log_experiment auto-revert on discard/checks_failed
  restores TRACKED files — commit large artifacts (refs) immediately or lose
  working-tree copies.

## Rig facts
- Ports 18601 (baseline) / 18602 (candidate); `--metrics` enabled; tracked
  PIDs; servers spawn under `nice -n 10` with `CUDA_VISIBLE_DEVICES=""`.
- Other missions share this host (load ~8-16); interleave arms per trace,
  never trust single-arm wall time across runs.
- A few TRAIN rows ramble to the 1024 cap (e.g. tiny 126-token prompts) —
  genuine greedy divergence, product cap applies; sweep runs use cap 256.
