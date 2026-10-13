#!/usr/bin/env bash
set -euo pipefail

PACKET="/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic9534-postrender-preparation-v2"
INPUT_MANIFEST="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-inputs-v1/manifest.json"
RENDER16="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1"
OUT="$PACKET/metadata/render16-terminal-merge.json"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python3 -B "$PACKET/source/verify_render16.py" \
  --input-manifest "$INPUT_MANIFEST" --render16 "$RENDER16" --output "$OUT"
