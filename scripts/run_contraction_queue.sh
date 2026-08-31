#!/bin/bash
# Master chain for the contraction queue (user GO 2026-08-31): runs after
# the already-running e3v2 chain finishes, then T1, then GatedNorm arms.
# Each stage is fail-independent (logs + board notes land per stage).
set -u
echo "[chain] waiting for e3v2 to finish..."
while pgrep -f "run_e3v2.sh" > /dev/null; do sleep 120; done
echo "[chain] e3v2 done at $(date -Is); launching T1"

bash /home/m0hawk/Documents/Sepalith/experiments/training/pvf_poc/run_t1.sh \
  >> /tmp/poc_cma/queue_chain.log 2>&1
echo "[chain] T1 rc=$? at $(date -Is); launching GatedNorm arms"

bash /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder/run_gatednorm.sh \
  >> /tmp/poc_cma/queue_chain.log 2>&1
echo "[chain] GatedNorm rc=$? — QUEUE COMPLETE $(date -Is)"
