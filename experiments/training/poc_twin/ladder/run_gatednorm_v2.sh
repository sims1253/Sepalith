#!/bin/bash
# P10 GatedNorm-v2 arms (queue §3 P10, 2026-09-05, zcode-gpushorts):
# sigma-init ~1 (gate bias +4, sigmoid(4)=0.982 — GatedNorm starts ==
# RMSNorm) instead of the standard ~0.5 center that halves sublayer outputs
# until the gate learns open. Question: does v2 close most of Q3's +2% BPB
# cost (gn_qk 0.7685/0.7538 vs banked plain control 0.7533/0.7527) -> GN
# re-enters the 25B conversation, or not -> rejection structural/scale-proof.
# Arms (paired discipline: dose 0.3, seed 1273, 668 steps, same streams):
#   gn2_qk     GatedNorm-v2 + QK-Clip(tau 100)  — the quality arm (vs +2%)
#   stress_gn2 GatedNorm-v2-only (tau 1e9) at 2x peak LR — the stress leg
#              (banked pair: stress_gn p99.9 1.19x clip vs stress_plain 2.28x)
# Eval: bpb_eval -> logs/bpb_eval_gn2.json; stress scorer vs banked stress_plain.
set -u
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
LAD=$TWIN/ladder
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
LOGS=$LAD/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM() {  # tag, extra flags
  local TAG=$1; shift
  if [ -f /tmp/poc_twin/ladder/ckpt_${TAG}/final.pt ]; then echo "[gn2] $TAG done"; return 0; fi
  echo "[gn2] === $TAG start $(date -Is)" >&2
  taskset -c 16-23 nice -n 5 $PY -u $LAD/train_ladder.py --dose 0.3 --seed 1273 \
    --steps 668 --compile --tag $TAG "$@" \
    > $LOGS/${TAG}_stdout.log 2>&1 \
    || { echo "[gn2] $TAG FAILED" >&2; return 1; }
  mkdir -p /mnt/h/sepalith/runs/gatednorm/${TAG}
  rsync -a /tmp/poc_twin/ladder/ckpt_${TAG}/final.pt /mnt/h/sepalith/runs/gatednorm/${TAG}/ 2>/dev/null
  echo "[gn2] === $TAG done $(date -Is)" >&2
}

ARM gn2_qk     --gated-norm-v2 --tau 100
ARM stress_gn2 --lr 0.02 --lr-embed 0.008 --gated-norm-v2 --tau 1e9

# paired BPB vs the banked plain control + Q3 arms
taskset -c 16-23 $PY -u $LAD/bpb_eval.py \
  --arms gn2_qk:/tmp/poc_twin/ladder/ckpt_gn2_qk/final.pt \
         stress_gn2:/tmp/poc_twin/ladder/ckpt_stress_gn2/final.pt \
  --out $LOGS/bpb_eval_gn2.json > $LOGS/bpb_eval_gn2_stdout.log 2>&1

echo "[gn2] ALL DONE $(date -Is)" >&2
