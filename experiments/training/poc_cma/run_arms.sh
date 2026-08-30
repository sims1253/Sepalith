#!/bin/bash
# decay/CMA + MuonH POC — Task 2 scorer + Task 4 arm chain (zcode-cma-poc).
# Sequential, resumable per arm (latest.pt). All arms: identical 1BT draw
# multiset (seed 1273), 1900 steps x 524,288 tok, Muon-mix pinned recipe,
# TinyGQA 206M trunk with the A2 32K vocab. Artifacts /tmp/poc_cma ->
# rsync /mnt/h/sepalith/runs/poc_cma/ after each arm (disk is 99% full).
# usage: run_arms.sh [scorer|C|D|K|KT|H|chain]
set -u
HERE=/home/m0hawk/Documents/Sepalith/experiments/training/poc_cma
TWIN=/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
DRAW=/tmp/poc_cma/draw_1bt_seed1273
EVALB=/tmp/poc_cma/eval_blocks.npy
RUNS=/mnt/h/sepalith/runs/poc_cma
LOGS=$HERE/logs
mkdir -p "$LOGS" /tmp/poc_cma
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POC_MEM_FRACTION=0.42
STEPS=1900

TRAIN() {  # $1 arm-args, $2 tag
  local TAG=$2
  if [ -f /tmp/poc_twin/ckpt_${TAG}/final.pt ]; then
    echo "[chain] $TAG already done"; return 0
  fi
  local RESUME=""
  if [ -f /tmp/poc_twin/ckpt_${TAG}/latest.pt ]; then
    RESUME="--resume /tmp/poc_twin/ckpt_${TAG}/latest.pt"
  fi
  echo "[chain] === $TAG start $(date -Is)" >&2
  nice -n 5 $PY -u $TWIN/train.py --arm muon --lr 0.01 --lr-embed 0.004 \
    --wd 0.1 --seed 1273 --steps $STEPS --tokens-per-step 524288 \
    --micro-bs 8 --vocab 32768 --order-file $DRAW --eval-data $EVALB \
    --ckpt-every 400 --eval-every 250 --log-every 50 --compile --elr \
    --tag $TAG $1 $RESUME > $LOGS/${TAG}_stdout.log 2>&1
  local RC=$?
  if [ $RC -ne 0 ] || [ ! -f /tmp/poc_twin/ckpt_${TAG}/final.pt ]; then
    echo "[chain] $TAG FAILED rc=$RC (see $LOGS/${TAG}_stdout.log)" >&2
    return $RC
  fi
  mkdir -p $RUNS/${TAG}
  rsync -a /tmp/poc_twin/ckpt_${TAG}/ $RUNS/${TAG}/
  rsync -a $TWIN/logs/${TAG}.jsonl $RUNS/${TAG}/train.jsonl
  # prune non-final local ckpts to protect the 99%-full disk
  find /tmp/poc_twin/ckpt_${TAG} -name 'latest.pt' -delete
  find /tmp/poc_twin/ckpt_${TAG} -name 'mid.pt' -delete
  echo "[chain] === $TAG done $(date -Is) (rsynced)" >&2
  return 0
}

SCORER() {
  if [ -f $DRAW/curriculum_order.idx.npy ]; then
    echo "[chain] scorer+curriculum already done"; return 0
  fi
  TRAIN "--decay-frac 0.2 --seed 2731 --steps 480" cma_scorer || return 1
  echo "[chain] scoring pass start $(date -Is)" >&2
  $PY -u $HERE/score_blocks.py score --draw $DRAW \
    --ckpt /tmp/poc_twin/ckpt_cma_scorer/final.pt \
    > $LOGS/scorer_score.log 2>&1 || { echo "[chain] score FAILED"; return 1; }
  $PY -u $HERE/score_blocks.py curriculum --draw $DRAW \
    > $LOGS/scorer_curriculum.log 2>&1 || { echo "[chain] curriculum FAILED"; return 1; }
  mkdir -p $RUNS/cma_scorer
  rsync -a $DRAW/block_ce.npy $DRAW/curriculum_order.idx.npy $RUNS/cma_scorer/
  echo "[chain] scorer+curriculum done $(date -Is)" >&2
}

case "${1:-chain}" in
  scorer) SCORER ;;
  C) TRAIN "--decay-frac 0.2" cma_C ;;
  D) TRAIN "--decay-frac 0.5" cma_D ;;
  K) TRAIN "--decay-frac 0.2 --order-file $DRAW/curriculum_order.idx.npy" cma_K ;;
  KT) TRAIN "--decay-frac 0.2 --order-file $DRAW/curriculum_order.idx.npy --const-tail-frac 0.05 --tail-ckpts 6" cma_KT ;;
  H) TRAIN "--decay-frac 0.2 --arm muonh --wd-muon 0" cma_H ;;
  chain)
    SCORER && TRAIN "--decay-frac 0.2" cma_C && TRAIN "--decay-frac 0.5" cma_D \
      && TRAIN "--decay-frac 0.2 --order-file $DRAW/curriculum_order.idx.npy" cma_K \
      && TRAIN "--decay-frac 0.2 --order-file $DRAW/curriculum_order.idx.npy --const-tail-frac 0.05 --tail-ckpts 6" cma_KT \
      && TRAIN "--decay-frac 0.2 --arm muonh --wd-muon 0" cma_H
    echo "[chain] ALL DONE rc=$? $(date -Is)"
    ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
