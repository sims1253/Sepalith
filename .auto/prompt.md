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

Baseline (extension flags, freeze v2, load 7.95): edit p50 382.8ms, p95 8.1s; type 291.5 (cached .954); accept 3153 (.453); diag 2111 (.661); cursor_move 4175 (.233); open 2545; file_switch 5010; switch_back 120 (default KV pool makes it free).

- `--cache-reuse 256` (run 2): DISCARDED — apparent -15% was host-load noise; cached fracs unchanged; parity 0.9909. Chunk matcher can't match window shifts/history insertions.
- Warm-up stable-head (run 4): cold 12x, switch 3.5x, cursor 1.6x; accept unchanged (head already cached). KEEP concept.
- **Warm-up FULL EAGER (runs 5,7,10): ALL events 64-65ms p50 / ~71ms p95 — 5.9x pooled edit p50 (v2), 49x slow-edit, 64x cursor_move. THE winner.** CPU 418-441s/pass (~2s/event, FLOPs-conserved, shifted to idle). Parity 0.9455 vs session-flow ref (baseline itself 0.9545 vs cold — prefix-cache splits flip near-ties).
- Warm-up FULL SETTLE (run 6): settles 64-65ms, type 229.6ms → edit p50 195.4ms; -6% CPU vs eager. Battery-gentle fallback only.
- 4-thread eager (run 8): 67.0ms TTFT floor holds; warmup CPU +58%.
- Flags (llama-bench): t8>t6>t4; fa on≈off; -ub 2048 +5-6% pp2048 (numerics, skip); -ctk/-ctv q8_0 pp2048 -28% (REJECT).
- Interrupt probe: overlap queues real request (TTFT=warmup time); client ABORT frees slot fast and KEEPS partial KV (69ms TTFT after abort) → always abort warm-up on pause.
- Layout sim (freeze v2): reordering sections ≈ NO byte-prefix gain (pooled .801 vs .788 stable-first vs .801 history-cap1); cursor moves stay .23 under all orders (prefix growth is the divergence). Retraining-for-ordering NOT justified.
- Quant table: pp flat 106-132 across tiers (Q5_K_M 65-80 outlier); tg: Q6_K 21.1 > Q4_K_M 15.8 ≈ Q8_0 16.1 (int8 dots already fast at 8T); 4T Q4_K_M 15.5 > Q8_0 12.3.
- DEV75 single latency use done (see report §7).

Recommended config = baseline flags + eager warm-up flow (report §4). Status: report drafted; PR after DEV75 + final verification.
