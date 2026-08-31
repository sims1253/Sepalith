#!/bin/bash
# E3-v2 — so_r_qa dose-response: 2 constant-share continuations from the
# shared scorer ckpt (paired vs the existing e3_control; same schedule,
# seed, holdouts). Orders from e3v2_orders.py (2x = 0.024, 4x = 0.048).
set -u
HERE=/home/m0hawk/Documents/Sepalith/experiments/training/poc_cma
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
EVALB=/tmp/poc_cma/eval_blocks.npy
RUNS=/mnt/h/sepalith/runs/poc_cma
LOGS=$HERE/logs
if [ ! -f /tmp/poc_cma/e3v2_sorqa_2x/order.npy ]; then
  $PY -u $HERE/e3v2_orders.py || exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POC_MEM_FRACTION=0.42

RUN() {
  local D=$1
  if [ -f /tmp/poc_twin/ckpt_${D}/final.pt ]; then echo "[e3v2] $D done"; return 0; fi
  echo "[e3v2] === $D start $(date -Is)" >&2
  local RESUME=""
  [ -f /tmp/poc_twin/ckpt_${D}/latest.pt ] && RESUME="--resume /tmp/poc_twin/ckpt_${D}/latest.pt"
  [ -z "$RESUME" ] && RESUME="--resume /tmp/poc_twin/ckpt_cma_scorer/final.pt"
  nice -n 5 $PY -u $TWIN/train.py --arm muon --lr 0.01 --lr-embed 0.004 \
    --wd 0.1 --seed 1273 --steps 960 --tokens-per-step 524288 \
    --micro-bs 8 --vocab 32768 --order-file /tmp/poc_cma/${D}/order.npy \
    --eval-data $EVALB --ckpt-every 200 --eval-every 250 --log-every 50 \
    --compile --elr --tag $D $RESUME > $LOGS/${D}_stdout.log 2>&1 \
    || { echo "[e3v2] $D FAILED" >&2; return 1; }
  mkdir -p $RUNS/${D}
  rsync -a /tmp/poc_twin/ckpt_${D}/ $RUNS/${D}/
  rsync -a $TWIN/logs/${D}.jsonl $RUNS/${D}/train.jsonl
  find /tmp/poc_twin/ckpt_${D} -name 'latest.pt' -delete
  find /tmp/poc_twin/ckpt_${D} -name 'mid.pt' -delete
  $PY -u $HERE/eval_arms.py --ckpt $RUNS/${D}/final.pt --tag ${D} \
    > $LOGS/${D}_eval.log 2>&1 || echo "[e3v2] eval $D FAILED" >&2
  echo "[e3v2] === $D done $(date -Is)" >&2
}

RUN e3v2_sorqa_2x && RUN e3v2_sorqa_4x
echo "[e3v2] ALL DONE $(date -Is)"
