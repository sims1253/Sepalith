#!/bin/bash
# Follow chain: after the GatedNorm chain finishes -> T1 -> gn_only retry.
set -u
while pgrep -f "run_gatednorm.sh" > /dev/null; do sleep 180; done
echo "[follow] GN chain done $(date -Is); T1 first"
bash /home/m0hawk/Documents/Sepalith/experiments/training/pvf_poc/run_t1.sh
echo "[follow] T1 rc=$? $(date -Is); gn_only retry"
cd /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder
if [ ! -f /tmp/poc_twin/ckpt_gn_only/final.pt ]; then
  nice -n 5 /home/m0hawk/Documents/Sepalith/.venv/bin/python3 -u train_ladder.py \
    --dose 0.3 --seed 1273 --steps 668 --compile --tag gn_only \
    --gated-norm --tau 1e9 --resume /tmp/poc_twin/ckpt_gn_only/latest.pt \
    > logs/gn_only_retry_stdout.log 2>&1
fi
echo "[follow] gn_only retry rc=$? $(date -Is) — FOLLOW CHAIN COMPLETE"
