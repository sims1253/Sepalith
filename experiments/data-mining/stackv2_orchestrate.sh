#!/bin/bash
# Stack v2+v3 R acquisition — phase orchestrator v2 (resume-safe chain).
# Waits for phase B fetch (pid $1) AND corpus tars, then chains:
# B-ext -> B-2 -> C -> D -> E(gate/dedup/tokens). Logs to stack_staging/logs/.
set -u
BPID="${1:-}"
LOGROOT=/mnt/h/sepalith/stack_staging/logs
cd /home/m0hawk/Documents/Sepalith
eval "$(grep -E '^export HF_TOKEN=' ~/.zshrc)"

log() { echo "$(date +%H:%M:%S) ORCH $*" | tee -a "$LOGROOT/orchestrator.log"; }

if [ -n "$BPID" ]; then
  log "waiting for phase B pid $BPID"
  while kill -0 "$BPID" 2>/dev/null; do sleep 60; done
fi
log "phase B exited; tail: $(tail -1 $LOGROOT/phase_b.out)"

log "waiting for corpus tars to finish"
while pgrep -f "tar -cf /tmp/stack_r_fetch/corpus" > /dev/null; do sleep 60; done
log "corpus tars done: $(du -sb /tmp/stack_r_fetch/corpus | cut -f1) bytes"

log "phase B extension pass (no deadline; no-op if complete)"
STACK_DEADLINE=$(( $(date +%s) + 14400 )) uv run python experiments/data-mining/stackv2_phase_b.py \
  >> "$LOGROOT/phase_b_ext.out" 2>&1
log "phase B extension done: $(tail -1 $LOGROOT/phase_b_ext.out)"

log "phase B-2 (cc-by-sa pool)"
uv run python experiments/data-mining/stackv2_phase_b2.py >> "$LOGROOT/phase_b2.out" 2>&1
log "phase B-2 done: $(tail -1 $LOGROOT/phase_b2.out)"

log "phase C (full-scale MinHash cross-corpus dedup)"
uv run python experiments/data-mining/stackv2_phase_c.py >> "$LOGROOT/phase_c.out" 2>&1
log "phase C done: $(tail -1 $LOGROOT/phase_c.out)"
if [ ! -f /mnt/h/sepalith/stack_staging/logs/phase_c_report.json ]; then
  log "FATAL: phase C produced no report — stopping chain"
  exit 1
fi

log "phase D (tokens + measure)"
uv run python experiments/data-mining/stackv2_phase_d.py >> "$LOGROOT/phase_d.out" 2>&1
log "phase D done: $(tail -1 $LOGROOT/phase_d.out)"

log "phase E gate (delta + SPDX-header license gate)"
uv run python experiments/data-mining/stackv3_phase_e.py gate >> "$LOGROOT/phase_e_gate.out" 2>&1
log "phase E gate done: $(tail -1 $LOGROOT/phase_e_gate.out)"

log "phase E dedup"
uv run python experiments/data-mining/stackv3_phase_e.py dedup >> "$LOGROOT/phase_e_dedup.out" 2>&1
log "phase E dedup done: $(tail -1 $LOGROOT/phase_e_dedup.out)"

log "phase E tokens"
uv run python experiments/data-mining/stackv3_phase_e.py tokens >> "$LOGROOT/phase_e_tokens.out" 2>&1
log "phase E tokens done: $(tail -1 $LOGROOT/phase_e_tokens.out)"

log "CHAIN COMPLETE"
