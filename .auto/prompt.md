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
- Offline sim on 4 reference outputs: ngram-simple small-n dominates
  (n=2-4, m=6-16): tok/step 1.75–3.3 on real edits vs 1.0–1.4 for defaults.

## Rig facts
- Ports 18601 (baseline) / 18602 (candidate); `--metrics` enabled; tracked
  PIDs; servers spawn under `nice -n 10` with `CUDA_VISIBLE_DEVICES=""`.
- Other missions share this host (load ~8-16); interleave arms per trace,
  never trust single-arm wall time across runs.
- A few TRAIN rows ramble to the 1024 cap (e.g. tiny 126-token prompts) —
  genuine greedy divergence, product cap applies; sweep runs use cap 256.
