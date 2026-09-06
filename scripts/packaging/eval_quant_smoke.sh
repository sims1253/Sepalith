#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
exec 9>/tmp/b_battery.lock
flock 9
for STEM in packaging_b4-Q8_0 packaging_b4_control-Q4_K_M packaging_b4_imatrix-Q4_K_M; do
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_scenarios.py \
    --model "$ROOT/experiments/models/$STEM.gguf" --port 18279 --cap "${SEPALITH_EVAL_CAP:-3}" \
    --server-bin "$ROOT/experiments/bin/llama/llama-b10453/llama-server" \
    > "experiments/models/quant-calibration/eval-$STEM.log" 2>&1
done
