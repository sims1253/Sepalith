#!/bin/bash
# Quant-tier prefill/decode speed table (CPU) for the quant mission.
# pp512 / pp2048 / tg32 per tier at 8 threads; Q8_0 and imat-Q4_K_M also at 4.
set -euo pipefail
BIN=/home/m0hawk/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-bench
DIR=/tmp/prefill-models
OUT=/mnt/e/sepalith/campaign-20260915/serving-work/prefill-20261017/quant-speed-table.csv
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
for m in arm-a-bf16 arm-a-Q8_0 arm-a-imat-Q6_K arm-a-imat-Q5_K_M arm-a-imat-Q4_K_M \
         arm-a-uncal-Q4_K_M arm-a-q4plus-Q4_K_M arm-a-attn-out-q8_0-Q4_K_M arm-a-head-q6_k-Q4_K_M; do
  echo "== $m @8t" >&2
  nice -n 10 "$BIN" -m "$DIR/$m.gguf" -t 8 -fa on -r 3 -p 512,2048 -n 32 -o csv 2>/dev/null | tail -n +2 >> "$OUT"
done
for m in arm-a-Q8_0 arm-a-imat-Q4_K_M; do
  echo "== $m @4t" >&2
  nice -n 10 "$BIN" -m "$DIR/$m.gguf" -t 4 -fa on -r 3 -p 512,2048 -n 32 -o csv 2>/dev/null | tail -n +2 >> "$OUT"
done
echo "wrote $OUT"
