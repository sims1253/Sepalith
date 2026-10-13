#!/usr/bin/env bash
set -euo pipefail

PACKET="/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic9534-postrender-preparation-v3"
ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1"
SELECTED="$ROOT/semantic9534-context-policy-v1/selected-contexts.jsonl"
TOKENIZER="/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0/tokenizer.json"
BINDING="$PACKET/dedup-binding.root.json"
OUT="$ROOT/Semantic9534-materialization-v1"

[[ -f "$BINDING" ]] || { echo "root-bound dedup binding required: $BINDING" >&2; exit 3; }
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false \
  python3 -B "$PACKET/source/materialize9534.py" \
  --selected "$SELECTED" --tokenizer "$TOKENIZER" \
  --dedup-binding "$BINDING" --output "$OUT"
