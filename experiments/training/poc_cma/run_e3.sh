#!/bin/bash
# Task 7 (E3) — proxy-strata continue-trains from the Task-2 scorer ckpt.
# 4 runs x 480 steps (0.25BT each): control + ramp for so_r_qa/bioc/
# curated_py. Orders pre-built by e3_orders.py (CPU). Then eval_arms.py
# per run = capability vector (held-out R-BPB + per-stratum holdout loss).
# usage: run_e3.sh            (assumes orders built: e3_orders.py)
set -u
HERE=/home/m0hawk/Documents/Sepalith/experiments/training/poc_cma
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
EVALB=/tmp/poc_cma/eval_blocks.npy
RUNS=/mnt/h/sepalith/runs/poc_cma
LOGS=$HERE/logs
if [ ! -f /tmp/poc_cma/e3_control/order.npy ]; then
  $PY -u $HERE/e3_orders.py || exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POC_MEM_FRACTION=0.42

RUN() {  # $1 dirname (tag)
  local D=$1
  if [ -f /tmp/poc_twin/ckpt_${D}/final.pt ]; then echo "[e3] $D done"; return 0; fi
  echo "[e3] === $D start $(date -Is)" >&2
  local RESUME=""
  [ -f /tmp/poc_twin/ckpt_${D}/latest.pt ] && RESUME="--resume /tmp/poc_twin/ckpt_${D}/latest.pt"
  [ -z "$RESUME" ] && RESUME="--resume /tmp/poc_twin/ckpt_cma_scorer/final.pt"
  nice -n 5 $PY -u $TWIN/train.py --arm muon --lr 0.01 --lr-embed 0.004 \
    --wd 0.1 --seed 1273 --steps 960 --tokens-per-step 524288 \
    --micro-bs 8 --vocab 32768 --order-file /tmp/poc_cma/${D}/order.npy \
    --eval-data $EVALB --ckpt-every 200 --eval-every 250 --log-every 50 \
    --compile --elr --tag $D $RESUME > $LOGS/${D}_stdout.log 2>&1 \
    || { echo "[e3] $D FAILED" >&2; return 1; }
  mkdir -p $RUNS/${D}
  rsync -a /tmp/poc_twin/ckpt_${D}/ $RUNS/${D}/
  rsync -a $TWIN/logs/${D}.jsonl $RUNS/${D}/train.jsonl
  find /tmp/poc_twin/ckpt_${D} -name 'latest.pt' -delete
  find /tmp/poc_twin/ckpt_${D} -name 'mid.pt' -delete
  $PY -u $HERE/eval_arms.py --ckpt $RUNS/${D}/final.pt --tag ${D} \
    > $LOGS/${D}_eval.log 2>&1 || echo "[e3] eval $D FAILED" >&2
  echo "[e3] === $D done $(date -Is)" >&2
}

RUN e3_control && RUN e3_so_r_qa_ramp && RUN e3_bioc_ramp && RUN e3_curated_py_ramp
echo "[e3] ALL DONE $(date -Is)"
