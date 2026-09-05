#!/usr/bin/env bash
# B9 — SeleKT gradient-importance masking A/B on the winner (b4-config
# qwen3.5-2b). Runbook: docs/research/2026-08-31-base-bakeoff-plan.md §2 B9;
# queue §2b B9 row; pre-registered adaptation + verdict rule in
# experiments/training/selekt_data.py (module header).
#
# Arm = the BANKED b4_qwen35_2b recipe VERBATIM (3000 steps, LoRA r32/a64,
# lr 2e-4 cosine, seq 2048, seed 3407, 48k-row shuffle(42) cap, same
# SFT_TARGETS list, unsloth-with-knobs, base experiments/models/
# qwen3.5-2b-base-text-hf, dataset sft_v7) with SELEKT_MASK=1: labels are
# the ONLY delta vs b4 (token-space gradient-importance masking, keep-50%;
# same rows/order/token streams — see the selekt_data.py header). Paired
# control = the BANKED b4 rung (no retrain).
#
# GATES:
#   A) attachment: "Trainable parameters" == 21,823,488 (the measured b4
#      line; B3 incident rule) — the arm keeps the FULL b4 attachment.
#   B) instrument telemetry: [selekt:mask] line seen (probe done, threshold
#      set), NO [midtrain: lines, first finite loss line (quoted-string TRL
#      grep per the B8b ops fix).
#   C) post-train nan/inf loss scan + probe artifact exists
#      (OUT/selekt_probe.json).
# OOM fallback: SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math;
# B13/B8b precedent). B8b-anomaly watch: VRAM transient peaked 32.1GB in
# the longest-row region on sft_v7 — a 30s sampler logs to
# ${STEM}_vram.log; the agent watching heartbeats triggers the fallback
# BEFORE an OOM if the same creep appears.
# Battery = CPU convention under flock /tmp/b_battery.lock (ports
# 18166/18168); midtyping --limit 18 with the (i,sha) join check vs banked
# b4 rows (run offline after via scripts/b8b_verdict.py b9_select_qwen35_2b).
# CPU discipline: trainer + battery pinned 16-23 (24-core box; other work
# may own 0-15). Launch pattern (B13/B8/O1/B8b precedent): chain DETACHED
# (setsid nohup); the session tracks a short watcher, never the workload
# (~1h harness reaper, five kills this session).
set -u
cd /home/m0hawk/Documents/Sepalith || exit 1

STEM=b9_select_qwen35_2b
MODEL=experiments/models/qwen3.5-2b-base-text-hf
DATA=/mnt/h/sepalith/datasets/sft_v7
RUNS=/mnt/h/sepalith/runs
LOG=$RUNS/b9_chain.log
TRAIN_LOG=$RUNS/${STEM}_train.log
EXPORT_LOG=$RUNS/${STEM}_export.log
BATT_LOG=$RUNS/${STEM}_battery.log
EXPECT_TRAINEE=21823488   # the measured b4/b8/b8b line: 21,823,488 (1.15%)

export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
export UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SFT_TARGETS='q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj,gate_proj,up_proj,down_proj'
export SELEKT_MASK=1 SELEKT_KEEP=0.5 SELEKT_PROBE_BUDGET=4096
unset MIDTRAIN_MASK MIDTRAIN_PACK FULL_FT || true   # one masking mechanism per arm

TS() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNS"
log "=== B9 SELEKT CHAIN START (stem $STEM, base $MODEL, data $DATA, keep 0.5) ==="

# ---------- 1) TRAIN (b4 recipe + SELEKT token-grad-imp label masking) ----------
rm -rf "$RUNS/$STEM"
log "B9 TRAIN start (expected trainable $EXPECT_TRAINEE; probe ~25m then ~1h45m train)"
taskset -c 16-23 .venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 3000 "$DATA" "$RUNS/$STEM" "" >> "$TRAIN_LOG" 2>&1 &
TPID=$!
log "B9 trainer pid $TPID (log $TRAIN_LOG)"

# VRAM sampler (B8b anomaly watch) — dies with the trainer
( while kill -0 "$TPID" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    sleep 30
  done ) >> "$RUNS/${STEM}_vram.log" 2>&1 &
VRAMPID=$!

# Gate A: wait for the attachment line, verify the count (up to 30m load)
gate=0
for _ in $(seq 1 90); do
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B9 TRAIN DIED before attachment line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  LINE=$(grep -m1 "Trainable parameters" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    N=$(echo "$LINE" | grep -oE "[0-9,]+ of" | head -1 | tr -d ' ,of')
    log "B9 ATTACHMENT LINE: $LINE"
    if [ "$N" = "$EXPECT_TRAINEE" ]; then
      log "B9 GATE-A PASS (trainable $N == $EXPECT_TRAINEE — exact b4/b8/b8b line)"
      gate=1
    else
      log "B9 GATE-A FAIL: trainable ${N:-?} != $EXPECT_TRAINEE — KILLING trainer $TPID"
      kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
      exit 2
    fi
    break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B9 GATE-A TIMEOUT (no attachment line in 30m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 3
fi

# Gate B0: [selekt:cfg] telemetry (tokenize map done, probe starting; 30m)
gate=0
for _ in $(seq 1 90); do
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B9 TRAIN DIED before [selekt:cfg] (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  if grep -q "\[selekt:cfg\]" "$TRAIN_LOG" 2>/dev/null; then
    log "B9 GATE-B0 PASS (cfg line seen; probe forwarding)"
    grep -m1 "\[selekt:tokenize\]" "$TRAIN_LOG" | tee -a "$LOG" || true
    gate=1; break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B9 GATE-B0 TIMEOUT (no [selekt:cfg] in 30m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 4
fi

# Gate B: [selekt:mask] threshold line + no midtrain telemetry + first
# finite loss (up to 90m: probe ~25m + trainer init + warmup)
gate=0
for _ in $(seq 1 270); do
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B9 TRAIN DIED before first loss line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  if grep -q "\[midtrain:" "$TRAIN_LOG" 2>/dev/null; then
    log "B9 GATE-B FAIL: [midtrain: telemetry present — wrong instrument — KILLING $TPID"
    kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
    exit 5
  fi
  MASK=$(grep -m1 "\[selekt:mask\]" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$MASK" ]; then
    log "B9 instrument: $MASK"
  fi
  LOSS=$(grep -m1 -oE "'loss': '[0-9.]+" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LOSS" ] && [ -n "$MASK" ]; then
    log "B9 GATE-B PASS (masked labels live, first finite loss $LOSS)"
    gate=1; break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B9 GATE-B TIMEOUT (no mask+loss lines in 90m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 6
fi

if wait "$TPID"; then
  log "B9 TRAIN done"
else
  rc=$?
  log "B9 TRAIN FAIL rc=$rc (see $TRAIN_LOG). If CUDA OOM / VRAM creep (B8b 32.1GB anomaly): relaunch SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math) + RESUME auto and note it."
  exit 7
fi

# Gate C: nan/inf scan + instrument recheck + probe artifact
if grep -E "'loss': ?'?(nan|inf)" "$TRAIN_LOG" >/dev/null 2>&1; then
  log "B9 GATE-C FAIL: non-finite loss found in $TRAIN_LOG — arm INVALID, not exporting"
  exit 8
fi
if grep -q "\[midtrain:" "$TRAIN_LOG" 2>/dev/null; then
  log "B9 GATE-C FAIL: [midtrain: telemetry appeared mid-run — instrument contract violated"
  exit 8
fi
if [ ! -f "$RUNS/$STEM/selekt_probe.json" ]; then
  log "B9 GATE-C FAIL: selekt_probe.json missing (probe artifact contract)"
  exit 8
fi
log "B9 GATE-C PASS (finite losses, instrument held, probe artifact present)"
grep -m1 "=== SMOKE: generation ===" -A2 "$TRAIN_LOG" | tail -2 | tee -a "$LOG" || true

# ---------- 2) EXPORT (Q8_0; standard unsloth merge + b10453 converter) ----------
sleep 30
log "B9 EXPORT start"
if taskset -c 16-23 .venv-sft/bin/python experiments/training/export_gguf.py "$MODEL" \
     "$RUNS/$STEM/final_lora" "$STEM" >> "$EXPORT_LOG" 2>&1; then
  log "B9 EXPORT done -> experiments/models/$STEM-Q8_0.gguf"
else
  log "B9 EXPORT FAIL (see $EXPORT_LOG)"; exit 9
fi

# ---------- 3) BATTERY (CPU convention; serialized via flock; pinned 16-23) ----------
GGUF=experiments/models/$STEM-Q8_0.gguf
(
  flock 9
  log "BATTERY start $STEM (CPU convention, pinned 16-23)"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" --threads 8 \
    >> "$BATT_LOG" 2>&1 && log "scen OK" || log "scen FAIL"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu --port 18166 \
    >> "$BATT_LOG" 2>&1 && log "noop OK" || log "noop FAIL"
  taskset -c 16-23 experiments/bin/llama/llama-b10453/llama-server -m "$GGUF" --port 18168 --host 127.0.0.1 \
    -c 8192 --parallel 1 -t 8 -ngl 0 > /tmp/b9_midtyping_server.log 2>&1 &
  SPID=$!; ready=0
  for _ in $(seq 1 150); do
    curl -s -m 2 -X POST http://127.0.0.1:18168/completion -d '{"prompt":"r","n_predict":1}' >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" = 1 ]; then
    for align in raw suffix; do
      suf=""; [ "$align" = suffix ] && suf="_suffix"
      taskset -c 16-23 .venv-sft/bin/python experiments/eval/run_eval.py --port 18168 --model zeta2 \
        --examples /mnt/h/sepalith/datasets/edit_pairs_v1/eval.jsonl \
        --variant midtyping --align "$align" --limit 18 \
        > "experiments/eval/results_${STEM}_midtyping${suf}.jsonl" 2>/dev/null
    done
    log "midtyping OK (raw+suffix, limit 18)"
  else log "midtyping server-not-ready"; fi
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  taskset -c 16-23 experiments/bin/llama/llama-b10453/llama-bench -m "$GGUF" -t 8 -p 512 -n 128 \
    >> "$BATT_LOG" 2>&1 && log "bench OK" || log "bench FAIL"
) 9>/tmp/b_battery.lock
log "=== B9 SELEKT CHAIN END ==="
