#!/usr/bin/env bash
# Root-owned execution template. This is intentionally not invoked by the
# preparation agent. It begins with a metadata-only terminal gate.
set -euo pipefail

PYTHON=${PYTHON:-/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python}
PACKET=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-final-corpus-launch-inputs-v1
FINAL_UNION=${FINAL_UNION:-/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1}
TERMINAL_GATE_REPORT=${TERMINAL_GATE_REPORT:-$FINAL_UNION/terminal-gate.json}
RECHUNK_OUTPUT=${RECHUNK_OUTPUT:-/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1-ctx16384}
RECHUNK_ROWS=$RECHUNK_OUTPUT/cpt_train_ctx16384.jsonl
RECHUNK_RESULT=$RECHUNK_OUTPUT/result.json
SCHEDULE=${SCHEDULE:-/mnt/e/sepalith/campaign-20260915/data-work/CPT-final-union-v1-draw-schedule.json}
CACHE=${CACHE:-/mnt/e/sepalith/campaign-20260915/data-work/CPT-streaming-input-v1/final-union-cache}

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2

# This opens only terminal metadata; the union payload remains unopened here.
"$PYTHON" -B "$PACKET/verify_terminal_union.py" \
  --union "$FINAL_UNION" --report "$TERMINAL_GATE_REPORT"

"$PYTHON" -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-cpt-three-repair-closure-v1/source/lossless_rechunk.py \
  --input-manifest "$FINAL_UNION/input-manifest.json" --output "$RECHUNK_OUTPUT"

"$PYTHON" -B "$PACKET/make_one_pass_schedule.py" \
  --rows "$RECHUNK_ROWS" --rechunk-result "$RECHUNK_RESULT" \
  --output "$SCHEDULE" --seed 3407 \
  --split-id cpt_train_final_union_ctx16384_v1

ROWS_SHA256=$(sha256sum "$RECHUNK_ROWS" | awk '{print $1}')
SCHEDULE_SHA256=$(sha256sum "$SCHEDULE" | awk '{print $1}')
mkdir -p "$(dirname "$CACHE")"
"$PYTHON" -B \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-full-weight-cpt-full-corpus-trainer-v3/source/experiments/training/cpt_streaming_cache.py \
  --rows "$RECHUNK_ROWS" --schedule "$SCHEDULE" --output "$CACHE" \
  --max-sequence-tokens 16384 --rows-sha256 "$ROWS_SHA256" \
  --schedule-sha256 "$SCHEDULE_SHA256"
