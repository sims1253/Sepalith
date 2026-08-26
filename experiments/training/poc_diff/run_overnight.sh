#!/usr/bin/env bash
# pocdiff overnight supervisor (ops rule: detached supervisor for anything
# long-lived). Waits for the pvf RL-run-4 GPU release (claim 2026-08-26
# 23:35, ETA ~2.5h), then: 40-step smoke (throughput gate >=30k tok/s,
# VRAM sanity) -> full 3815-step / 2B-token run -> rsync artifacts to
# /mnt/h/sepalith/runs/poc_diff/. Every state change is mirrored to the
# comms ledger (comms/gpu.md) and board (comms/board.md) per protocol.
set -u
ROOT=/home/m0hawk/Documents/Sepalith
GPU_LEDGER=$ROOT/comms/gpu.md
BOARD=$ROOT/comms/board.md
LOG=/tmp/poc_diff/supervisor.log
mkdir -p /tmp/poc_diff

ts() { date "+%Y-%m-%dT%H:%M+02"; }
log() { echo "[$(ts)] $*" >> "$LOG"; }
claim() { echo "[$(ts)] zcode-pocdiff $*" >> "$GPU_LEDGER"; }
board() {
  { echo ""; echo "## [$(ts)] FROM zcode-pocdiff TO ALL — $1"; echo "$2"; } >> "$BOARD"
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
rl_running() { pgrep -f "03_grpo_tether" >/dev/null 2>&1; }

log "supervisor up (pid $$); waiting for pvf RL-run-4 release"
while true; do
  if grep -q "zcode-pvf-poc RELEASE RL-run-4" "$GPU_LEDGER"; then
    log "ledger shows RL-run-4 released"
    break
  fi
  # fallback per the reap house rule: process gone AND >=16GB free
  if ! rl_running; then
    fm=$(free_mib)
    if [ "${fm:-0}" -ge 16384 ]; then
      log "RL-run-4 absent + ${fm}MiB free; reaping stale claim with board note"
      board "stale RL-run-4 claim reaped" \
        "RL-run-4 process not found and >=16GB free; taking the GPU per the reap house rule. @zcode-pvf-poc: correct me on the board if your run is still alive."
      break
    fi
  fi
  sleep 300
done

while [ "$(free_mib)" -lt 16384 ]; do
  log "gate: waiting for >=16GB free (now $(free_mib)MiB)"
  sleep 300
done

claim "CLAIM md smoke+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h"
log "GPU claimed; running smoke"
cd "$ROOT" || exit 1
if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.training.poc_diff.train_md --smoke >> "$LOG" 2>&1; then
  log "SMOKE FAILED (crash)"
  claim "RELEASE md smoke+full (smoke crashed)"
  board "md smoke FAILED" \
    "smoke crashed; see /tmp/poc_diff/supervisor.log tail; GPU released."
  exit 1
fi
tok=$(grep -o '"tok_per_s": *[0-9.]*' experiments/training/poc_diff/logs_md.jsonl \
      | tail -1 | grep -o '[0-9.]*$')
log "smoke finished, last tok_per_s=${tok:-none}"
ok=$(python3 -c "print(1 if float('${tok:-0}') >= 30000 else 0)")
if [ "$ok" != "1" ]; then
  claim "RELEASE md smoke+full (throughput ${tok} < 30k gate)"
  board "md smoke throughput fail" \
    "tok_per_s=${tok} below the pre-registered 30k gate; GPU released; investigating compile/eager + micro-budget tradeoff before re-claiming."
  log "THROUGHPUT GATE FAIL (${tok})"
  exit 1
fi

log "smoke PASSED (${tok} tok/s); starting full run (3815 steps, 2B tokens)"
if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.training.poc_diff.train_md --steps 3815 --compile >> "$LOG" 2>&1; then
  rsync -a /tmp/poc_diff/ckpt/ /mnt/h/sepalith/runs/poc_diff/
  claim "RELEASE md full run (done)"
  board "md full run DONE" \
    "2B-token run complete; checkpoints + telemetry in /mnt/h/sepalith/runs/poc_diff/; paired eval (Task 6) next."
  log "FULL RUN DONE"
else
  rc=$?
  log "FULL RUN FAILED rc=$rc (latest.pt survives for --resume)"
  claim "RELEASE md full run (failed rc=$rc; checkpoint resumable)"
  board "md full run FAILED" \
    "rc=$rc; /tmp/poc_diff/ckpt/latest.pt survives for --resume; see /tmp/poc_diff/supervisor.log."
fi
