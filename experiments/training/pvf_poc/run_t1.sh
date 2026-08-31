#!/bin/bash
# T1 — DAPO zero-std filter arm (v5 config + filter; paired vs the banked
# rl_grpo_v5_loo_unnorm baseline). Smoke first (10 steps: verifies the
# ZSTD keying + drop counters), then the 220-step full arm.
set -u
PV=/home/m0hawk/Documents/Sepalith/experiments/training/pvf_poc
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python3   # 3.10: the
# 3.14 venv's dill/datasets pickling is broken (fingerprint crash)
OUT=/mnt/h/sepalith/runs/rl_grpo_t1_dapo

echo "[t1] smoke start $(date -Is)"
$PY -u $PV/03_grpo_tether.py --rho 0.0 --smoke --dapo-filter \
  > /tmp/t1_smoke.log 2>&1
grep -q "dapo_dropped" /tmp/rl_tether_smoke/rl_metrics.jsonl \
  || { echo "[t1] SMOKE FAIL (no dapo metrics) — see /tmp/t1_smoke.log"; exit 1; }
echo "[t1] smoke OK: $(grep dapo /tmp/rl_tether_smoke/rl_metrics.jsonl | tail -1 | head -c 200)"

echo "[t1] full arm start $(date -Is)"
$PY -u $PV/03_grpo_tether.py --rho 0.0 --steps 220 --dapo-filter \
  --out $OUT > /tmp/t1_full.log 2>&1 \
  || { echo "[t1] FULL FAILED"; exit 1; }
echo "[t1] DONE $(date -Is): $OUT/rl_metrics.jsonl"
