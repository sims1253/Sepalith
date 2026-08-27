#!/usr/bin/env bash
# pocdiff overnight supervisor v2 (ops rule: detached supervisor for
# anything long-lived). Chain: drain any in-flight train_md -> wait for
# >=16GB free -> gate the ALREADY-RUNNING/recent smoke off its "done"
# event (tokens/total_s; a 40-step run emits no per-100-step telemetry,
# which is what v1's grep wrongly required) or run a fresh smoke if none
# -> full 3815-step / 2B-token run -> rsync artifacts to
# /mnt/h/sepalith/runs/poc_diff/. Ledger claims + board notes at each
# transition. If the full run fails, latest.pt survives for --resume and
# the GPU is released (zcode-ddot-poc's OT supervisor stands down on our
# failure instead of racing — board 01:21).
set -u
ROOT=/home/m0hawk/Documents/Sepalith
GPU_LEDGER=$ROOT/comms/gpu.md
BOARD=$ROOT/comms/board.md
LOG=/tmp/poc_diff/supervisor.log
LOGMD=$ROOT/experiments/training/poc_diff/logs_md.jsonl
mkdir -p /tmp/poc_diff

ts() { date "+%Y-%m-%dT%H:%M+02"; }
log() { echo "[$(ts)] $*" >> "$LOG"; }
claim() { echo "[$(ts)] zcode-pocdiff $*" >> "$GPU_LEDGER"; }
board() {
  { echo ""; echo "## [$(ts)] FROM zcode-pocdiff TO ALL — $1"; echo "$2"; } >> "$BOARD"
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
md_running() { pgrep -f "poc_diff.train_md" >/dev/null 2>&1; }

# 1. drain: an orphaned smoke from a previous supervisor may still be up
while md_running; do
  log "drain: a train_md process is still running; waiting"
  sleep 60
done

# 2. gate
while [ "$(free_mib)" -lt 16384 ]; do
  log "gate: waiting for >=16GB free (now $(free_mib)MiB)"
  sleep 300
done

claim "CLAIM md smoke-adjudicated+full (train_md.py, memfrac 0.42 <=14GB) ETA 16h"
cd "$ROOT" || exit 1

# 3. adjudicate the most recent COMPLETE smoke (a done event is the
#    completeness proof — a killed smoke's partial telemetry is
#    compile-warmup garbage); else fresh smoke (compiled — the gate
#    targets the full-run config)
tok="none"
if [ -f "$LOGMD" ]; then
  tok=$(python3 - "$LOGMD" << 'PY'
import json, sys
telem = done = None
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("tok_per_s"):
        telem = r["tok_per_s"]  # last 10-step window: steady state
    if r.get("event") == "done" and r.get("total_s") and r.get("tokens"):
        done = r["tokens"] / max(r["total_s"], 1e-9)
print(round(max(filter(None, [telem, done])), 1) if done else "")
PY
)
fi
if [ -z "$tok" ] || [ "$tok" = "none" ]; then
  log "no COMPLETE smoke on record; running a fresh one (compiled)"
  if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -m experiments.training.poc_diff.train_md --smoke --compile >> "$LOG" 2>&1; then
    log "SMOKE FAILED (crash)"
    claim "RELEASE md smoke+full (smoke crashed)"
    board "md smoke FAILED" "smoke crashed; see /tmp/poc_diff/supervisor.log tail; GPU released."
    exit 1
  fi
  tok=$(python3 - "$LOGMD" << 'PY'
import json, sys
telem = done = None
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("tok_per_s"):
        telem = r["tok_per_s"]  # last 10-step window: steady state
    if r.get("event") == "done" and r.get("total_s") and r.get("tokens"):
        done = r["tokens"] / max(r["total_s"], 1e-9)
print(round(max(filter(None, [telem, done])), 1) if done else "")
PY
)
fi
log "smoke adjudicated at ${tok:-?} tok/s"
ok=$(python3 -c "print(1 if float('${tok:-0}') >= 30000 else 0)")
if [ "$ok" != "1" ]; then
  claim "RELEASE md smoke+full (throughput ${tok} < 30k gate)"
  board "md smoke throughput fail" \
    "tok_per_s=${tok} below the pre-registered 30k gate; GPU released; investigating compile/eager + micro-budget tradeoff before re-claiming."
  log "THROUGHPUT GATE FAIL (${tok})"
  exit 1
fi

# 4. full run
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
