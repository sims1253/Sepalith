#!/bin/bash
# Queue continuation (user GO 2026-08-31 "continue running experiments from
# EXPERIMENT-QUEUE.md when current ones are done"): after the Q2/Q3 follow
# chain finishes -> Q6 batch probe (next config-only GPU item per the
# queue's cost-first order). B-series (B1-B3) implementation follows in a
# session; verdicts for Q2/Q3 are read by the owner session on wake.
set -u
while pgrep -f "follow_chain.sh" > /dev/null; do sleep 180; done
echo "[cont] follow chain done $(date -Is); launching Q6 batch probe"
bash /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/run_q6_batchprobe.sh
echo "[cont] Q6 rc=$? $(date -Is) — CONTINUATION STAGE 1 COMPLETE"
