#!/bin/bash
# llama-bench flag sweep for prefill (pp) and decode (tg) on the CPU build.
# Arms: fa on/off, ubatch size, threads (t/tb split), KV cache type.
# Everything at nice -n 10; results to the artifacts dir as CSV.
set -euo pipefail
BIN=/home/m0hawk/.local/opt/llama.cpp-b10453/build-cpu/bin/llama-bench
MODEL=${1:-/tmp/prefill-models/arm-a-Q8_0.gguf}
OUT=${2:-/mnt/e/sepalith/campaign-20260915/serving-work/prefill-20261017/llama-bench-sweep.csv}
mkdir -p "$(dirname "$OUT")"

run() {
  echo "== $*" >&2
  nice -n 10 "$BIN" -m "$MODEL" -r 3 -o csv "$@" | tail -n +2 >> "$OUT"
}

: > "$OUT"
# thread split (8 total): t8/tb8, t4/tb8, t6/tb8, t8 default tb
for t in 8 4 6; do
  run -t "$t" -tb 8 -fa on  -ub 512  -p 512,2048 -n 32
done
# flash attention on/off (default auto)
for fa in on off; do
  run -t 8 -tb 8 -fa "$fa" -ub 512  -p 512,2048 -n 32
done
# ubatch size (b>=ub)
for ub in 512 1024 2048; do
  run -t 8 -tb 8 -fa on -ub "$ub" -p 2048 -n 32
done
# KV cache type (numerics-changing; speed only)
for ct in f16 q8_0; do
  run -t 8 -tb 8 -fa on -ub 512 -ctk "$ct" -ctv "$ct" -p 512,2048 -n 32
done
# 4-thread laptop-like reference row
run -t 4 -tb 4 -fa on -ub 512 -p 512,2048 -n 32
echo "wrote $OUT"
