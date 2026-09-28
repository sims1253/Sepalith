#!/bin/sh
set -eu
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
OUTPUT=/mnt/e/sepalith/campaign-20260915/data-work/CPT-all-eligible-v1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
exec ionice -c3 nice -n 10 taskset -c 0,2 \
  python3 "$PLAN/docs/campaign/work/lead/r2-cpt-all-eligible-v2/materialize_all_eligible.py" \
  --output "$OUTPUT"
