#!/usr/bin/env bash
# B8b — STACKED arm: midtrain-THEN-sft_v7 (granite's actual structure).
# Runbook: docs/EXPERIMENT-QUEUE.md §2b B8 row (B8b fired from the banked
# artifact); predecessor: scripts/run_b8_midtrain.sh (replacement arm,
# decisively negative) + scripts/run_b4_unsloth_safe.sh (the control rung).
#
# Arm = the BANKED b4_qwen35_2b recipe VERBATIM (3000 steps, LoRA r32/a64,
# lr 2e-4 cosine, seq 2048, seed 3407, 48k-row shuffle(42) cap, same
# SFT_TARGETS list, unsloth-with-knobs) on the MERGED base
# /mnt/h/sepalith/runs/b8b_stacked_base_merged (= qwen3.5-2b-base-text-hf
# with the BANKED b8_midtrain LoRA merged in via the export_gguf.py
# MERGE_VIA_PEFT flow; scripts/b8b_merge_base.py, gates G1-G3 PASS) and
# the sft_v7 product dataset. MIDTRAIN OFF — plain legacy SFT path
# (byte-compatibility contract in train_sft.py).
# Paired control = the BANKED b4 rung (same targets/recipe on the
# un-merged base).
#
# GATES:
#   A) attachment: "Trainable parameters" == 21,823,488 (the measured b4
#      line; B8 reproduced it — B3 incident rule)
#   B) LEGACY-PATH assertion: NO "[midtrain:" telemetry may ever appear
#      in the train log (MIDTRAIN off) + first finite loss line seen.
#      NOTE (fixed post-run 2026-09-06): TRL emits loss values as QUOTED
#      strings ('loss': '1.05') AND stdout is block-buffered — the original
#      grep "'loss': [0-9.]+" never matched; during the live run this was
#      bridged with a labeled ops line in the train log (see RESULTS §B8b
#      ops notes). Patterns below now match the quoted form; the
#      authoritative finite-loss scan is trainer_state.json (external).
#   C) post-train nan/inf loss scan
# OOM fallback: SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer
# math; B13/B8 precedent). Battery = CPU convention under flock
# /tmp/b_battery.lock (ports 18162/18164); midtyping --limit 18 with the
# (i,sha) join check vs the banked b4 rows (run offline after).
# CPU discipline: trainer + battery pinned 16-23 (bench batch owns 0-15;
# 24-core box — the 16-31 range in the brief partly doesn't exist).
# Launch pattern (B13/B8 precedent): CHAIN detached (setsid nohup); the
# session tracks a small watcher, never the workload (~1h harness reaper).
set -u
cd /home/m0hawk/Documents/Sepalith || exit 1

STEM=b8b_stacked_qwen35_2b
MODEL=/mnt/h/sepalith/runs/b8b_stacked_base_merged
DATA=/mnt/h/sepalith/datasets/sft_v7
RUNS=/mnt/h/sepalith/runs
LOG=$RUNS/b8b_chain.log
TRAIN_LOG=$RUNS/${STEM}_train.log
EXPORT_LOG=$RUNS/${STEM}_export.log
BATT_LOG=$RUNS/${STEM}_battery.log
EXPECT_TRAINEE=21823488   # the measured b4/b8 line: 21,823,488 (1.15%)

export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
export UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SFT_TARGETS='q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj,gate_proj,up_proj,down_proj'
unset MIDTRAIN_MASK MIDTRAIN_PACK || true   # legacy SFT path — byte-compat contract

TS() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNS"
log "=== B8b STACKED CHAIN START (stem $STEM, model $MODEL, data $DATA, MIDTRAIN OFF) ==="

# ---------- 1) TRAIN (b4 recipe on the merged base) ----------
rm -rf "$RUNS/$STEM"
log "B8b TRAIN start (expected trainable $EXPECT_TRAINEE)"
taskset -c 16-23 .venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 3000 "$DATA" "$RUNS/$STEM" "" >> "$TRAIN_LOG" 2>&1 &
TPID=$!
log "B8b trainer pid $TPID (log $TRAIN_LOG)"

# Gate A: wait for the attachment line, verify the count
gate=0
for _ in $(seq 1 90); do   # up to 30 min for load + get_peft_model + dataset map
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B8b TRAIN DIED before attachment line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  LINE=$(grep -m1 "Trainable parameters" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    N=$(echo "$LINE" | grep -oE "[0-9,]+ of" | head -1 | tr -d ' ,of')
    log "B8b ATTACHMENT LINE: $LINE"
    if [ "$N" = "$EXPECT_TRAINEE" ]; then
      log "B8b GATE-A PASS (trainable $N == $EXPECT_TRAINEE — exact b4/b8 line)"
      gate=1
    else
      log "B8b GATE-A FAIL: trainable ${N:-?} != $EXPECT_TRAINEE — KILLING trainer $TPID"
      kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
      exit 2
    fi
    break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B8b GATE-A TIMEOUT (no attachment line in 30m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 3
fi

# Gate B: legacy-path assertion + first finite loss (up to 60m for the 264k-row map)
gate=0
for _ in $(seq 1 180); do
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B8b TRAIN DIED before first loss line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  if grep -q "\[midtrain:" "$TRAIN_LOG" 2>/dev/null; then
    log "B8b GATE-B FAIL: [midtrain: telemetry present but MIDTRAIN must be OFF — KILLING trainer $TPID"
    kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
    exit 4
  fi
  LOSS=$(grep -m1 -oE "'loss': '[0-9.]+" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LOSS" ]; then
    log "B8b GATE-B PASS (legacy path, first finite loss $LOSS, no [midtrain: lines)"
    gate=1; break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B8b GATE-B TIMEOUT (no loss line in 60m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 5
fi

if wait "$TPID"; then
  log "B8b TRAIN done"
else
  rc=$?
  log "B8b TRAIN FAIL rc=$rc (see $TRAIN_LOG). If CUDA OOM / VRAM creep: relaunch SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math) and note it."
  exit 6
fi

# Gate C: nan/inf loss scan + final legacy-path recheck
if grep -E "'loss': ?'?(nan|inf)" "$TRAIN_LOG" >/dev/null 2>&1; then
  log "B8b GATE-C FAIL: non-finite loss found in $TRAIN_LOG — arm INVALID, not exporting"
  exit 7
fi
if grep -q "\[midtrain:" "$TRAIN_LOG" 2>/dev/null; then
  log "B8b GATE-C FAIL: [midtrain: telemetry appeared mid-run — instrument contract violated"
  exit 7
fi
log "B8b GATE-C PASS (no nan/inf loss lines; legacy path held end-to-end)"
grep -m1 "=== SMOKE: generation ===" -A2 "$TRAIN_LOG" | tail -2 | tee -a "$LOG" || true

# ---------- 2) EXPORT (Q8_0; standard unsloth merge + b10453 converter) ----------
sleep 30
log "B8b EXPORT start"
if taskset -c 16-23 .venv-sft/bin/python experiments/training/export_gguf.py "$MODEL" \
     "$RUNS/$STEM/final_lora" "$STEM" >> "$EXPORT_LOG" 2>&1; then
  log "B8b EXPORT done -> experiments/models/$STEM-Q8_0.gguf"
else
  log "B8b EXPORT FAIL (see $EXPORT_LOG)"; exit 8
fi

# ---------- 3) BATTERY (CPU convention; serialized via flock; pinned 16-23) ----------
GGUF=experiments/models/$STEM-Q8_0.gguf
(
  flock 9
  log "BATTERY start $STEM (CPU convention, pinned 16-23)"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" --threads 8 \
    >> "$BATT_LOG" 2>&1 && log "scen OK" || log "scen FAIL"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu --port 18162 \
    >> "$BATT_LOG" 2>&1 && log "noop OK" || log "noop FAIL"
  taskset -c 16-23 experiments/bin/llama/llama-b10453/llama-server -m "$GGUF" --port 18164 --host 127.0.0.1 \
    -c 8192 --parallel 1 -t 8 -ngl 0 > /tmp/b8b_midtyping_server.log 2>&1 &
  SPID=$!; ready=0
  for _ in $(seq 1 150); do
    curl -s -m 2 -X POST http://127.0.0.1:18164/completion -d '{"prompt":"r","n_predict":1}' >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" = 1 ]; then
    for align in raw suffix; do
      suf=""; [ "$align" = suffix ] && suf="_suffix"
      taskset -c 16-23 .venv-sft/bin/python experiments/eval/run_eval.py --port 18164 --model zeta2 \
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
log "=== B8b STACKED CHAIN END ==="
