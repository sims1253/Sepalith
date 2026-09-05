#!/usr/bin/env bash
# PFT1 — full fine-tuning arm: LoRA-vs-full-FT at matched budget (queue §3).
# Runbook: docs/EXPERIMENT-QUEUE.md §3 PFT1 row + docs/research/2026-09-06-lora-vs-fullft.md §3.
# Predecessors: scripts/run_b8b_stacked.sh (chain shape, gates, battery),
# scripts/run_b4_unsloth_safe.sh (B4-saga knobs).
#
# Arm (b) = FULL-FT on the b4-config base: experiments/models/qwen3.5-2b-base-text-hf
# + sft_v7, 3000 steps, seq 2048, seed 3407, bf16, paged 8-bit AdamW, gradient
# checkpointing (use_reentrant=False), lr 1.5e-5 cosine (pre-registered; ONE
# rescue at 5e-6/3e-5 via FULL_FT_LR env — the queue log documents any use),
# bs2 x ga8 = effective 16 (B13 logits-block precedent; identical optimizer
# math to b4's bs4xga4). Trainer stack: unsloth full_finetuning=True (the
# B4-saga finding kills the plain TRL path on this base: 25s/it = 21h).
# Anchor = the BANKED b4 LoRA rung (no rerun); paired control rows already
# in experiments/eval/results_{scenarios,noop_fp}_b4_qwen35_2b.jsonl.
#
# PHASES (all inside ONE card claim, strictly serial = W37):
#   0) preflight: artifacts present + CPU merge of the b4 anchor (probe input)
#   1) SMOKE: 24-step validated-build run (attachment 100%, finite loss, s/it)
#   2) TRAIN: 3000 steps, gates A/B/C, VRAM peak-watch with ONE bs1xga16
#      auto-relaunch if 3 consecutive samples >30.5GB (pre-OOM intervention,
#      B13 pattern)
#   3) EXPORT Q8_0 (NO_LORA)
#   4) BPB forgetting probe (GPU; base + b4-merged + pft1)
#   5) CARD RELEASE marker (watcher posts the comms release)
#   6) CPU battery under flock /tmp/b_battery.lock (scenarios, noopFP,
#      midtyping raw+suffix --limit 18, llama-bench t8, V1a episode metrics
#      --live --n 60) + paired verdict helper (b8b_verdict.py)
# Launch pattern (B13/B8/O1 precedent): CHAIN detached (setsid nohup); the
# session tracks short watchers, never the workload (~1h harness reaper).
set -u
cd /home/m0hawk/Documents/Sepalith || exit 1

STEM=pft1_fullft_qwen35_2b
MODEL=experiments/models/qwen3.5-2b-base-text-hf
DATA=/mnt/h/sepalith/datasets/sft_v7
RUNS=/mnt/h/sepalith/runs
OUT=$RUNS/$STEM
SMOKE_OUT=$RUNS/pft1_fullft_smoke
LOG=$RUNS/pft1_chain.log
TRAIN_LOG=$RUNS/${STEM}_train.log
SMOKE_LOG=$RUNS/pft1_fullft_smoke.log
EXPORT_LOG=$RUNS/${STEM}_export.log
PROBE_LOG=$RUNS/${STEM}_bpb_probe.log
BATT_LOG=$RUNS/${STEM}_battery.log
VRAM_CSV=$RUNS/${STEM}_vram.csv
B4_MERGED=$RUNS/pft1_b4_merged
PEAK_KILL=$RUNS/${STEM}_PEAK_KILL

export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
export UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FULL_FT=1 SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8
export FULL_FT_LR="${FULL_FT_LR:-1.5e-5}"
unset MIDTRAIN_MASK MIDTRAIN_PACK || true

TS() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNS"
log "=== PFT1 FULL-FT CHAIN START (stem $STEM, base $MODEL, lr $FULL_FT_LR) ==="

# ---------- 0) PREFLIGHT ----------
for f in "$MODEL/config.json" "$DATA/train.jsonl" "$DATA/eval.jsonl" \
         "$RUNS/b4_qwen35_2b/final_lora/adapter_config.json" \
         "experiments/eval/results_scenarios_b4_qwen35_2b.jsonl" \
         "experiments/eval/results_noop_fp_b4_qwen35_2b.jsonl"; do
  [ -f "$f" ] || { log "PREFLIGHT FAIL: missing $f"; exit 1; }
done
log "PREFLIGHT PASS (base, sft_v7, b4 anchor artifacts all present)"

if [ ! -f "$B4_MERGED/config.json" ]; then
  log "b4 anchor merge (CPU, gates G1-G3) start"
  taskset -c 16-23 .venv-sft/bin/python scripts/pft1_merge_b4.py >> "$PROBE_LOG" 2>&1 \
    && log "b4 anchor merged -> $B4_MERGED" \
    || { log "b4 MERGE FAIL (see $PROBE_LOG)"; exit 1; }
else
  log "b4 anchor merge cached ($B4_MERGED exists)"
fi

# ---------- helper: VRAM peak-watch (kills the trainer BEFORE OOM) ----------
watch_vram() {  # $1 = trainer pid
  local pid=$1 over=0 m
  while kill -0 "$pid" 2>/dev/null; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$m" ] && echo "$(date +%s),$m" >> "$VRAM_CSV"
    if [ -n "$m" ] && [ "$m" -gt 30500 ]; then
      over=$((over + 1))
      if [ "$over" -ge 3 ]; then
        log "PEAK-WATCH INTERVENTION: 3 consecutive samples >30.5GB (last ${m}MiB) — killing trainer $pid BEFORE OOM (B13 pattern)"
        touch "$PEAK_KILL"
        kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
        return 0
      fi
    else
      over=0
    fi
    sleep 30
  done
}

# ---------- helper: gate scan on trainer_state.json (authoritative) ----------
scan_state() {  # $1 = checkpoint dir; exit nonzero on non-finite loss
  .venv-sft/bin/python - "$1" <<'EOF'
import json, math, sys
st = json.load(open(sys.argv[1] + "/trainer_state.json"))
bad = [h for h in st["log_history"]
       if not math.isfinite(float(h.get("loss", h.get("eval_loss", 0.0))))]
last = st["log_history"][-1]
print(f"state scan: step {st['global_step']}, {len(st['log_history'])} log points, "
      f"non-finite {len(bad)}, last {sorted(last.items())[-3:]}")
sys.exit(1 if bad else 0)
EOF
}

# ---------- 1) SMOKE (validated-build gate; 24 steps -> one loss line at 20) ----------
rm -rf "$SMOKE_OUT"
log "SMOKE start (24 steps, throwaway dir) — validating the FULL-FT build before the arm"
taskset -c 16-23 .venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 24 "$DATA" "$SMOKE_OUT" "" >> "$SMOKE_LOG" 2>&1 &
SPID=$!
smoke_gate=0
for _ in $(seq 1 120); do   # up to 40 min for load + map + 24 steps
  kill -0 "$SPID" 2>/dev/null || break
  LINE=$(grep -m1 "PFT1 FULL-FT Trainable parameters" "$SMOKE_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    PCT=$(echo "$LINE" | grep -oE "\(([0-9.]+)%\)" | grep -oE "[0-9.]+")
    log "SMOKE ATTACHMENT: $LINE"
    if [ "${PCT:-0}" = "100.00" ]; then
      log "SMOKE GATE-A PASS (trainable == ALL params, no adapter)"
      smoke_gate=1
    else
      log "SMOKE GATE-A FAIL: trainable pct ${PCT:-?} != 100.00 — KILLING"
      kill "$SPID" 2>/dev/null; sleep 5; kill -9 "$SPID" 2>/dev/null; exit 2
    fi
    break
  fi
  sleep 20
done
[ "$smoke_gate" = 1 ] || { log "SMOKE GATE-A TIMEOUT (no attachment line)"; kill "$SPID" 2>/dev/null; exit 3; }
if ! wait "$SPID"; then
  log "SMOKE FAIL rc=$? (see $SMOKE_LOG)"; tail -5 "$SMOKE_LOG" | tee -a "$LOG"; exit 4
fi
grep -m1 -oE "'loss': '?[0-9.]+" "$SMOKE_LOG" | tee -a "$LOG" || { log "SMOKE GATE-B FAIL: no loss line"; exit 5; }
scan_state "$SMOKE_OUT/checkpoint-24" 2>/dev/null || true
grep -oE "[0-9.]+ s/it|[0-9.]+it/s" "$SMOKE_LOG" | tail -1 | tee -a "$LOG" || true
grep -m1 "PFT1 FULL-FT lr=" "$SMOKE_LOG" | tee -a "$LOG"
log "SMOKE PASS (finite loss at step 20; build validated) — cleaning up"
rm -rf "$SMOKE_OUT"

# ---------- 2) TRAIN (the arm; 3000 steps) ----------
rm -rf "$OUT"
log "PFT1 TRAIN start (3000 steps, bs2xga8, lr $FULL_FT_LR, expected VRAM ~14-18GB)"
: > "$VRAM_CSV"
taskset -c 16-23 .venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 3000 "$DATA" "$OUT" "" >> "$TRAIN_LOG" 2>&1 &
TPID=$!
log "PFT1 trainer pid $TPID (log $TRAIN_LOG)"
watch_vram "$TPID" &
WPID=$!
train_rc=0
wait "$TPID" || train_rc=$?
kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null

if [ -f "$PEAK_KILL" ] && [ "$train_rc" -ne 0 ]; then
  log "PEAK-KILL relaunch at bs1xga16 (identical optimizer math), resume=auto from last checkpoint"
  rm -f "$PEAK_KILL"
  export SFT_PD_BATCH=1 SFT_GRAD_ACCUM=16
  taskset -c 16-23 .venv-sft/bin/python experiments/training/train_sft.py \
    "$MODEL" 3000 "$DATA" "$OUT" "auto" >> "$TRAIN_LOG" 2>&1 &
  TPID=$!
  log "PFT1 relaunch trainer pid $TPID (bs1xga16)"
  watch_vram "$TPID" &
  WPID=$!
  train_rc=0
  wait "$TPID" || train_rc=$?
  kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null
fi

if [ "$train_rc" -ne 0 ]; then
  log "PFT1 TRAIN FAIL rc=$train_rc (see $TRAIN_LOG; tail follows)"
  tail -15 "$TRAIN_LOG" | tee -a "$LOG"; exit 6
fi
log "PFT1 TRAIN done (3000/3000)"

# Gate C: authoritative finite-loss scan on the final trainer state
CK=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if ! scan_state "$CK"; then
  log "PFT1 GATE-C FAIL: non-finite loss in $CK — arm INVALID, not exporting"; exit 7
fi
log "PFT1 GATE-C PASS (finite losses end-to-end)"
grep -q "\[midtrain:" "$TRAIN_LOG" 2>/dev/null && { log "GATE-C FAIL: midtrain telemetry leaked"; exit 7; }
grep -m1 "=== SMOKE: generation ===" -A2 "$TRAIN_LOG" | tail -2 | tee -a "$LOG" || true
PEAK=$(sort -t, -k2 -n "$VRAM_CSV" | tail -1 | cut -d, -f2)
log "PFT1 VRAM peak ${PEAK}MiB (pre-registration 14-18GB class; CSV $VRAM_CSV)"

# ---------- 3) EXPORT (Q8_0; full model, no adapter) ----------
sleep 30
log "PFT1 EXPORT start (NO_LORA)"
if NO_LORA=1 taskset -c 16-23 .venv-sft/bin/python experiments/training/export_gguf.py \
     "$MODEL" "$OUT/final_model" "$STEM" >> "$EXPORT_LOG" 2>&1; then
  log "PFT1 EXPORT done -> experiments/models/$STEM-Q8_0.gguf"
else
  log "PFT1 EXPORT FAIL (see $EXPORT_LOG)"; exit 8
fi

# ---------- 4) BPB FORGETTING PROBE (GPU; base + b4 anchor + pft1) ----------
log "PFT1 BPB probe start (causal-floor general-R + general-text control)"
if taskset -c 16-23 .venv-sft/bin/python experiments/eval/pft1_bpb_probe.py \
     --model base="$MODEL" --model b4="$B4_MERGED" --model pft1="$OUT/final_model" \
     --out experiments/eval/pft1_bpb_probe.json >> "$PROBE_LOG" 2>&1; then
  log "PFT1 BPB probe done (experiments/eval/pft1_bpb_probe.json)"
else
  log "PFT1 BPB probe FAIL (see $PROBE_LOG) — battery proceeds; probe retry is manual"
fi

# ---------- 5) CARD RELEASE MARKER ----------
log "=== PFT1 CARD RELEASE POINT (train+export+probe done; battery is CPU-only) ==="

# ---------- 6) CPU BATTERY (flock; pinned 16-23) ----------
GGUF=experiments/models/$STEM-Q8_0.gguf
(
  flock 9
  log "BATTERY start $STEM (CPU convention, pinned 16-23)"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" --threads 8 \
    >> "$BATT_LOG" 2>&1 && log "scen OK" || log "scen FAIL"
  taskset -c 16-23 .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu --port 18166 \
    >> "$BATT_LOG" 2>&1 && log "noop OK" || log "noop FAIL"
  taskset -c 16-23 experiments/bin/llama/llama-b10453/llama-server -m "$GGUF" --port 18168 --host 127.0.0.1 \
    -c 8192 --parallel 1 -t 8 -ngl 0 > /tmp/pft1_midtyping_server.log 2>&1 &
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
  # V1a episode metrics (queue §3 PFT1 readout; the documented one-liner)
  taskset -c 16-23 .venv/bin/python experiments/eval/episode_metrics.py --live \
    --model "$GGUF" --tag ${STEM%%_*} --n 60 \
    --traj /mnt/h/sepalith/datasets/sim_trajectories_v1/trajectories_v2.jsonl \
    >> "$BATT_LOG" 2>&1 && log "episode OK" || log "episode FAIL"
  # paired verdict vs the banked b4 anchor
  taskset -c 16-23 .venv/bin/python scripts/b8b_verdict.py "$STEM" \
    >> "$RUNS/${STEM}_verdict.txt" 2>&1 && log "verdict helper OK" || log "verdict helper FAIL"
  cp experiments/eval/pft1_bpb_probe.json "$RUNS/${STEM}_bpb_probe.json" 2>/dev/null || true
) 9>/tmp/b_battery.lock
log "=== PFT1 FULL-FT CHAIN END ==="
