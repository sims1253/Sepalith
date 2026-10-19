#!/bin/bash
set -euo pipefail
# Autoresearch measure loop for specdec. Candidate flags come from
# .auto/candidate.json ({"name": ..., "server_args": [...]}).
# Env overrides: SPECDEC_N_TRACES (default 60), SPECDEC_N_PREDICT (default 256),
# SPECDEC_COLD=1 to add the cold (full-prefill) pass for gate runs,
# SPECDEC_THREADS (default 8) for the laptop band.
cd "$(dirname "$0")/.."
CONFIG="${SPECDEC_CONFIG:-.auto/candidate.json}"
N_TRACES="${SPECDEC_N_TRACES:-60}"
N_PREDICT="${SPECDEC_N_PREDICT:-256}"
THREADS="${SPECDEC_THREADS:-8}"
COLD_ARGS=()
if [[ "${SPECDEC_COLD:-0}" == "1" ]]; then
  COLD_ARGS=(--cold)
fi
ABORT_ARGS=()
if [[ "${SPECDEC_WARMUP_ABORT_MS:-0}" != "0" ]]; then
  ABORT_ARGS=(--warmup-abort-ms "${SPECDEC_WARMUP_ABORT_MS}")
fi
MODEL_ARGS=()
if [[ -n "${SPECDEC_MODEL:-}" ]]; then
  MODEL_ARGS=(--model "${SPECDEC_MODEL}")
fi
BIN_ARGS=()
if [[ -n "${SPECDEC_SERVER_BIN:-}" ]]; then
  BIN_ARGS=(--server-bin "${SPECDEC_SERVER_BIN}")
fi
exec python3 experiments/serving/specdec/bench_spec.py \
  --config "$CONFIG" \
  --n-traces "$N_TRACES" \
  --n-predict "$N_PREDICT" \
  --threads "$THREADS" \
  "${ABORT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${BIN_ARGS[@]}" \
  --reference experiments/serving/specdec/reference_outputs.json \
  "${COLD_ARGS[@]}"
