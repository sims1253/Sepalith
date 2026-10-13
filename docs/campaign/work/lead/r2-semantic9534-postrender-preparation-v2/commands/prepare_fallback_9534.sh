#!/usr/bin/env bash
set -euo pipefail

PACKET="/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic9534-postrender-preparation-v2"
INPUTS="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-inputs-v1"
RENDER16="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1"
TERMINAL_REVIEW="$PACKET/metadata/render16-terminal-merge.json"
FALLBACK="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-32k-semantic9534-postrender-v1/inputs"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python3 -B "$PACKET/source/prepare_fallback_9534.py" \
  --inputs "$INPUTS" --render16 "$RENDER16" \
  --terminal-review "$TERMINAL_REVIEW" --output "$FALLBACK"
