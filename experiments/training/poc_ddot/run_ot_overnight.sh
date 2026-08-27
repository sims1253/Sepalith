#!/usr/bin/env bash
# poc_ddot overnight supervisor (pattern: poc_diff/run_overnight.sh — the
# detached-supervisor ops rule; carries the GPU queue without any agent
# session). Waits for zcode-pocdiff's full md run to SUCCEED (ledger
# release "done" or md_final.pt rsynced), then: preflight (regen slots
# bin if the tmpfs dropped it) -> 40-step smoke (throughput gate >=20k
# tok/s — the OT coupling's per-example Sinkhorn allowance vs pocdiff's
# 30k; lower means the coupling integration is broken, not physics) ->
# full 3815-step / 2B-token OT run -> rsync artifacts. Every transition
# mirrors to comms/gpu.md + comms/board.md per protocol. If pocdiff's
# run FAILS, stands down with a board note instead of racing their
# retry.
set -u
ROOT=/home/m0hawk/Documents/Sepalith
GPU_LEDGER=$ROOT/comms/gpu.md
BOARD=$ROOT/comms/board.md
LOG=/tmp/poc_ddot/supervisor.log
mkdir -p /tmp/poc_ddot

ts() { date "+%Y-%m-%dT%H:%M+02"; }
log() { echo "[$(ts)] $*" >> "$LOG"; }
claim() { echo "[$(ts)] zcode-ddot-poc $*" >> "$GPU_LEDGER"; }
board() {
  { echo ""; echo "## [$(ts)] FROM zcode-ddot-poc TO ALL — $1"; echo "$2"; } >> "$BOARD"
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
md_running() { pgrep -f "poc_diff.train_md" >/dev/null 2>&1; }

log "supervisor up (pid $$); waiting for pocdiff md full run success"
while true; do
  if grep -q "zcode-pocdiff RELEASE md full run (done)" "$GPU_LEDGER"; then
    log "ledger shows pocdiff full run done"
    break
  fi
  # v2 lesson (the 01:25 incident): the md_final.pt artifact check fired
  # on a STALE rsync during a gap in pocdiff's supervisor cycles. The
  # artifact path now ALSO requires no live pocdiff claim (their newest
  # ledger line must not be an unreleased CLAIM) and no trainer process.
  if [ -f /mnt/h/sepalith/runs/poc_diff/md_final.pt ] && ! md_running; then
    last_pocdiff=$(grep "zcode-pocdiff" "$GPU_LEDGER" | tail -1)
    case "$last_pocdiff" in
      *"CLAIM"*)
        log "live pocdiff claim in ledger ($last_pocdiff) — keeping wait"
        ;;
      *)
        fm=$(free_mib)
        if [ "${fm:-0}" -ge 16384 ]; then
          log "md_final.pt present, no live pocdiff claim, ${fm}MiB free — proceeding (board note)"
          board "pocdiff run completion detected via artifact" \
            "md_final.pt rsynced, no train_md process, no live claim in the ledger; >=16GB free. Proceeding to the queued OT smoke+full claim. @zcode-pocdiff: correct me if you still need the GPU."
          break
        fi
        ;;
    esac
  fi
  if grep -q "zcode-pocdiff RELEASE md full run (failed" "$GPU_LEDGER"; then
    log "pocdiff full run FAILED — standing down (their retry owns the GPU)"
    board "OT supervisor standing down (pocdiff run failed)" \
      "Saw the failure release in the ledger; your retry owns the GPU next. Re-arm me (bash experiments/training/poc_ddot/run_ot_overnight.sh) once a new completion signal posts."
    exit 0
  fi
  sleep 300
done

while [ "$(free_mib)" -lt 16384 ]; do
  log "gate: waiting for >=16GB free (now $(free_mib)MiB)"
  sleep 300
done

# preflight: /tmp is tmpfs — regenerate the slots bin if it dropped
if [ ! -f /tmp/poc_ddot/train_slots.bin ]; then
  log "slots bin missing; regenerating from /tmp/poc_diff triples"
  if [ -f /tmp/poc_diff/train_triples.jsonl ]; then
    (cd "$ROOT" && .venv/bin/python -m experiments.training.poc_ddot.data_prep_pos) \
      >> "$LOG" 2>&1 || { board "OT preflight FAILED" \
        "slots regeneration crashed; see /tmp/poc_ddot/supervisor.log."; exit 1; }
  else
    board "OT preflight blocked" \
      "/tmp/poc_diff triples gone (tmpfs) and slots bin missing — needs a poc_diff data_prep rerun before the OT run can start. Standing down."
    exit 0
  fi
fi

claim "CLAIM ot smoke+full (train_ot.py, memfrac 0.42 <=14GB) ETA 17h"
log "GPU claimed; running OT smoke"
cd "$ROOT" || exit 1
if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.training.poc_ddot.train_ot --smoke --compile >> "$LOG" 2>&1; then
  log "OT SMOKE FAILED (crash)"
  claim "RELEASE ot smoke+full (smoke crashed)"
  board "OT smoke FAILED" \
    "smoke crashed; see /tmp/poc_ddot/supervisor.log tail; GPU released."
  exit 1
fi
tok=$(grep -o '"tok_per_s": *[0-9.]*' experiments/training/poc_ddot/logs_ot.jsonl \
      | tail -1 | grep -o '[0-9.]*$')
ent=$(grep -o '"ot_plan_entropy": *[0-9.]*' experiments/training/poc_ddot/logs_ot.jsonl \
      | tail -1 | grep -o '[0-9.]*$')
log "smoke finished, last tok_per_s=${tok:-none}, plan_entropy=${ent:-none}"
ok=$(python3 -c "print(1 if float('${tok:-0}') >= 20000 else 0)")
if [ "$ok" != "1" ]; then
  claim "RELEASE ot smoke+full (throughput ${tok} < 20k gate)"
  board "OT smoke throughput fail" \
    "tok_per_s=${tok} below the pre-registered 20k gate (OT Sinkhorn allowance); GPU released; supervisor standing down for diagnosis."
  log "THROUGHPUT GATE FAIL (${tok})"
  exit 1
fi

log "smoke PASSED (${tok} tok/s, plan entropy ${ent}); starting full OT run (3815 steps, 2B tokens)"
if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.training.poc_ddot.train_ot --steps 3815 --compile >> "$LOG" 2>&1; then
  rsync -a /tmp/poc_ddot/ckpt/ /mnt/h/sepalith/runs/poc_ddot/
  claim "RELEASE ot full run (done)"
  board "OT full run DONE" \
    "2B-token OT twin trained; checkpoints + telemetry in /mnt/h/sepalith/runs/poc_ddot/ (ot_final.pt). Three-way eval (eval_ot.py: best-arm vs CAL vs OT, kill test, verdict) unblocked."
  log "FULL RUN DONE"
else
  rc=$?
  log "OT FULL RUN FAILED rc=$rc (latest.pt survives for --resume)"
  claim "RELEASE ot full run (failed rc=$rc; checkpoint resumable)"
  board "OT full run FAILED" \
    "rc=$rc; /tmp/poc_ddot/ckpt/latest.pt survives for --resume; see /tmp/poc_ddot/supervisor.log."
fi
