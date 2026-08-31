#!/bin/bash
# GatedNorm ladder arms (Qwen Eq.29) — paired vs the banked plain 668-step
# control (0.7533/0.7527 causal/FIM BPB, the A2-structure validation's
# plain arm). Four runs, same dose/seed/streams as every ladder arm:
#   gn_qk        GatedNorm + QK-Clip(tau 100)   — stack question
#   gn_only      GatedNorm, QK-Clip disabled (tau 1e9) — replacement Q
#   stress_plain plain recipe at 2x peak LR     — stability margin pair
#   stress_gn    GatedNorm-only at 2x peak LR
# Pre-registered verdicts (see close-out doc):
#   ADOPT gn_only iff final causal+FIM BPB within 0.5% of the banked
#   plain control AND stress_gn spike-rate < stress_plain (else keep
#   QK-Clip; adopt the stack only if gn_qk beats plain by >=0.3% BPB).
set -u
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
LAD=$TWIN/ladder
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
LOGS=$LAD/logs

# ladder streams are /tmp-wiped on reboots — deterministic rebuild
if [ ! -f /tmp/poc_twin/ladder/train_blocks_causal.npy ]; then
  echo "[gn] rebuilding ladder streams (deterministic)" >&2
  $PY -u $TWIN/data_prep.py > /tmp/gn_prep.log 2>&1 || exit 1
  $PY -u $LAD/data_prep_ladder.py >> /tmp/gn_prep.log 2>&1 || exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM() {  # tag, extra flags
  local TAG=$1; shift
  if [ -f /tmp/poc_twin/ckpt_${TAG}/final.pt ]; then echo "[gn] $TAG done"; return 0; fi
  echo "[gn] === $TAG start $(date -Is)" >&2
  nice -n 5 $PY -u $LAD/train_ladder.py --dose 0.3 --seed 1273 \
    --steps 668 --compile --tag $TAG "$@" \
    > $LOGS/${TAG}_stdout.log 2>&1 \
    || { echo "[gn] $TAG FAILED" >&2; return 1; }
  mkdir -p /mnt/h/sepalith/runs/gatednorm/${TAG}
  rsync -a /tmp/poc_twin/ckpt_${TAG}/final.pt /mnt/h/sepalith/runs/gatednorm/${TAG}/ 2>/dev/null
  echo "[gn] === $TAG done $(date -Is)" >&2
}

ARM gn_qk     --gated-norm --tau 100
ARM gn_only   --gated-norm --tau 1e9
ARM stress_plain --lr 0.02 --lr-embed 0.008
ARM stress_gn --lr 0.02 --lr-embed 0.008 --gated-norm --tau 1e9
echo "[gn] ALL DONE $(date -Is)"
