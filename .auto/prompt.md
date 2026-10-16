# Autoresearch: cut Sepalith's time-to-suggestion (prefill) on CPU llama.cpp

## Objective
Reduce time from "user pauses" to "suggestion shown" for Sepalith (2.5B R-edit
model) served by llama.cpp b10453 CPU (`--parallel 1`, temp 0), where the time
is dominated by PRM03 prompt prefill. Work the levers: server/thread flags,
cross-request KV reuse (`--cache-prompt`, `--cache-reuse`, ctx-checkpoints,
`--cache-ram`/multi-slot), pre-warming (n_predict=0 requests with the stable
prompt head), quant/KV-type speed tables (info for the quant mission), and —
as proposals only — prompt budget/layout changes.

## Metrics
- **Primary**: `edit_ttft_p50_ms` (ms, lower better) — median client TTFT over
  intra-file edit events (type/accept/diag/cursor_move) in the frozen session
  simulation, 8 threads, arm-a-Q8_0.gguf, 2 passes.
- **Secondary**: `edit_ttft_p95_ms`, `cold_ttft_p50_ms` (open), `switchback_ttft_p50_ms`,
  `type/accept/cursormove_ttft_p50_ms`, `edit_cached_frac`, `warmup_total_s`
  (CPU cost of pre-warming), `host_load1`.

## How to Run
`./.auto/measure.sh` (config under test: `experiments/serving/prefill/configs/current.json`,
env `PASSES`, `PORT`). Also `.auto/checks.sh` for lint/types/integrity.
Reference outputs for parity: `experiments/serving/prefill/reference_outputs.json`
(frozen once from the baseline run).

## Files in Scope
- `experiments/serving/prefill/build_sessions.py` — session freeze builder (TRAIN-only data).
- `experiments/serving/prefill/sessions.jsonl`, `manifest.json` — frozen workload (sha-pinned).
- `experiments/serving/prefill/bench.py` — measurement rig (server lifecycle, TTFT, parity ids).
- `experiments/serving/prefill/configs/*.json` — candidate configs (data, not code).
- `.auto/*` — session files.
- Artifacts (not committed): `/mnt/e/sepalith/campaign-20260915/serving-work/prefill-20261017/`.
- Report (final): `docs/validation/2026-10-17-prefill-autoresearch.md` + `.json`.

## Off Limits
- The sealed final set (`DAT-07-final-*`, `sealed-final/`), pre-split pools,
  old `spec_traces`, `experiments/.cache`, old `results_*`.
- `~/.local/opt/llama.cpp-b10453/` (read-only), `~/.local/state/sepalith/night-queue/`,
  the main checkout `/home/m0hawk/sandbox/Sepalith`, the extension code,
  the prompt format (PRM03) except in proposals.
- CUDA (night window owns the GPU): always `CUDA_VISIBLE_DEVICES=""`.
- DEV75/DEV250: only the final latency report + ≤3 quality confirmations.
- Max 8 threads, everything at `nice -n 10`; never kill/renice other projects.

## Constraints
- `python3 scripts/check_quality.py` (default mode) must stay clean (ruff+ty).
  Note: `--research` mode is pre-existing-broken repo-wide (1060 legacy ty
  diagnostics in old experiment files) — not our gate.
- Output parity: configs claiming to be free must produce token-identical
  outputs vs the frozen reference on the frozen sessions.
- No benchmark overfitting/cheating: real session sim, honest medians, paired
  repeats; note host load (night queue shares the box until ~09:00 CEST).

## What's Been Tried
(baseline pending — update as experiments accumulate)
