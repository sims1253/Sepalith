#!/bin/bash
# Q6 — batch-size probe (512k -> 1M tokens/step) + LR probe at 1M.
# Paired at the 206M twin scale on the SAME data (train_blocks.npy),
# matched 400M-token budget per arm; readout = train loss at 400M tokens
# (jsonl logs; step 800 for the 512k arm, step 400 for the 1M arms).
# Qwen finding under Muon: too-small batch hurts sharply, too-big flat;
# optimum shifts up with batch -> probe 1x and 2x LR at 1M.
set -u
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
LOGS=$TWIN/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POC_MEM_FRACTION=0.55

ARM() {  # tag steps tps lr
  local TAG=$1 STEPS=$2 TPS=$3 LR=$4
  if [ -f /tmp/poc_twin/ckpt_${TAG}/final.pt ]; then echo "[q6] $TAG done"; return 0; fi
  echo "[q6] === $TAG start $(date -Is)" >&2
  nice -n 5 $PY -u $TWIN/train.py --arm muon --lr $LR --lr-embed 0.004 \
    --wd 0.1 --seed 1273 --steps $STEPS --tokens-per-step $TPS \
    --micro-bs 8 --vocab 32768 --compile --elr \
    --data /tmp/poc_twin/train_blocks.npy \
    --eval-data /tmp/poc_twin/eval_blocks.npy \
    --ckpt-every 400 --eval-every 400 --log-every 50 \
    --tag $TAG > $LOGS/${TAG}_stdout.log 2>&1 \
    || { echo "[q6] $TAG FAILED" >&2; return 1; }
  mkdir -p /mnt/h/sepalith/runs/q6_batchprobe
  rsync -a $TWIN/logs/${TAG}.jsonl /mnt/h/sepalith/runs/q6_batchprobe/${TAG}.jsonl
  find /tmp/poc_twin/ckpt_${TAG} -name 'latest.pt' -delete 2>/dev/null
  echo "[q6] === $TAG done $(date -Is)" >&2
}

ARM q6_b512k  800 524288  0.01     # control: production batch, 400M tok
ARM q6_b1m    400 1048576 0.01     # batch only
ARM q6_b1m_lr2 400 1048576 0.02    # batch + 2x LR
echo "[q6] ALL DONE $(date -Is)"
