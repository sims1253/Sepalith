#!/bin/bash
set -euo pipefail
# Autoresearch measure loop for specdec. Candidate flags come from
# .auto/candidate.json ({"name": ..., "server_args": [...]}).
cd "$(dirname "$0")/.."
CONFIG="${SPECDEC_CONFIG:-.auto/candidate.json}"
N_TRACES="${SPECDEC_N_TRACES:-90}"
N_PREDICT="${SPECDEC_N_PREDICT:-256}"
exec python3 experiments/serving/specdec/bench_spec.py \
  --config "$CONFIG" \
  --n-traces "$N_TRACES" \
  --n-predict "$N_PREDICT" \
  --reference experiments/serving/specdec/reference_outputs.json
