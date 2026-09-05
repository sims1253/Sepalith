#!/usr/bin/env bash
# B13 — LFM2.5-2.6B-Base LoRA SFT rung (conv+GQA at the family ceiling).
# Runbook: docs/research/2026-08-31-base-bakeoff-plan.md §1/§2 + recon
#   docs/research/2026-09-04-base-candidate-recon-ouro-k2-lfm.md
# Recipe = the uniform B-series (3000 steps, sft_v7, LoRA r32/a64, lr 2e-4
# cosine, seq 2048, seed 3407; same as B2/B3/B4 — no step-count divergence).
#
# TARGET SET (the recon's LFM2.5 list, TRUE full attachment): unsloth's
# list->regex conversion drops modules under parents it doesn't know — B3's
# list run attached 72 modules (0 conv: conv.in_proj/conv.out_proj silently
# frozen, verified in its saved adapter). B13 therefore passes a RAW regex
# (train_sft.py "regex:" prefix, added 2026-09-05) that PEFT fullmatches:
#   166 modules = 8x(q,k,v,attn-out) + 22x(conv in_proj) + 30x(conv out_proj)
#                 + 30x(w1,w2,w3)
# EXPECTED TRAINABLE = 48,922,624 of 2,697,198,592 (1.81%)  [B3 INCIDENT RULE]
# A list-run (or any regex typo) under-attaches to 40,271,872 — the gate
# below kills the trainer if the printed count != 48,922,624.
#
# Env knobs: unsloth-with-knobs per the B4 saga (compile + auto-padding-free
# off). OOM fallback: SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (product stays 16).
# Battery: CPU convention (scenarios/noop self-spawn CPU servers; llama-bench
# t8 CPU = the banked comparability row); midtyping server = CUDA build
# (--ngl 99) ONLY if B13_MIDTYPING_SERVE=cuda is exported AND the card is
# unclaimed (gpu.md checked by the operator before launch) — else CPU build,
# both under flock /tmp/b_battery.lock, port 18133.
set -u
cd /home/m0hawk/Documents/Sepalith || exit 1

STEM=b13_lfm25_26b
MODEL=experiments/models/lfm25-2b-base-hf
DATA=/mnt/h/sepalith/datasets/sft_v7
RUNS=/mnt/h/sepalith/runs
LOG=$RUNS/b13_chain.log
TRAIN_LOG=$RUNS/${STEM}_train.log
EXPORT_LOG=$RUNS/${STEM}_export.log
BATT_LOG=$RUNS/${STEM}_battery.log
EXPECT_TRAINEE=48922624

export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
export UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SFT_TARGETS='regex:model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|conv\.(?:in_proj|out_proj)|feed_forward\.(?:w1|w2|w3))'

TS() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNS"
log "=== B13 CHAIN START (stem $STEM, model $MODEL) ==="

# ---------- 1) TRAIN with the B3 INCIDENT-RULE attachment gate ----------
# B13_WIPE=1 (default here) starts fresh; unset it and set B13_RESUME=auto to
# resume from the newest checkpoint (used after the 2026-09-05T00:49 session
# task-kill at step 1394; checkpoint-1000 was intact — house v5 resume pattern).
if [ "${B13_WIPE:-1}" = 1 ]; then rm -rf "$RUNS/$STEM"; fi
RESUME_ARG="${B13_RESUME:-}"
log "B13 TRAIN start (expected trainable $EXPECT_TRAINEE, resume '${RESUME_ARG:-fresh}')"
.venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 3000 "$DATA" "$RUNS/$STEM" "$RESUME_ARG" >> "$TRAIN_LOG" 2>&1 &
TPID=$!
log "B13 trainer pid $TPID (log $TRAIN_LOG)"

# gate: wait for the "Trainable parameters" line, then verify the count
gate=0
for _ in $(seq 1 90); do   # up to 30 min for load+get_peft_model
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B13 TRAIN DIED before attachment line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  LINE=$(grep -m1 "Trainable parameters" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    N=$(echo "$LINE" | grep -oE "[0-9,]+ of" | head -1 | tr -d ' ,of')
    log "B13 ATTACHMENT LINE: $LINE"
    if [ "$N" = "$EXPECT_TRAINEE" ]; then
      log "B13 INCIDENT-RULE GATE PASS (trainable $N == $EXPECT_TRAINEE; 166 modules)"
      gate=1
    else
      log "B13 INCIDENT-RULE GATE FAIL: trainable ${N:-?} != $EXPECT_TRAINEE — KILLING trainer $TPID (under/over-attached; NOT training a broken arm)"
      kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
      exit 2
    fi
    break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B13 GATE TIMEOUT (no attachment line in 30m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 3
fi

if wait "$TPID"; then
  log "B13 TRAIN done"
else
  rc=$?
  log "B13 TRAIN FAIL rc=$rc (see $TRAIN_LOG). If CUDA OOM: relaunch with SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math) and note it."
  exit 4
fi

# ---------- 2) EXPORT (ctx-release guard per l16 lesson) ----------
sleep 30
log "B13 EXPORT start"
if .venv-sft/bin/python experiments/training/export_gguf.py "$MODEL" \
     "$RUNS/$STEM/final_lora" "$STEM" >> "$EXPORT_LOG" 2>&1; then
  log "B13 EXPORT done -> experiments/models/$STEM-Q8_0.gguf"
else
  log "B13 EXPORT FAIL (see $EXPORT_LOG)"; exit 5
fi

# ---------- 3) BATTERY (CPU convention; serialized via flock) ----------
GGUF=experiments/models/$STEM-Q8_0.gguf
(
  flock 9
  log "BATTERY start $STEM"
  # 3a. scenarios (script-owned CPU server, .venv for tree_sitter_r)
  .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" --threads 8 \
    >> "$BATT_LOG" 2>&1 && log "scen OK" || log "scen FAIL"

  # 3b. midtyping raw+suffix on an OWNED server, port 18133.
  # CUDA build only when B13_MIDTYPING_SERVE=cuda (operator verified card free);
  # default = CPU build -ngl 0 (banked convention).
  if [ "${B13_MIDTYPING_SERVE:-cpu}" = cuda ]; then
    SRV=experiments/bin/llama/llama-cuda-b10453/llama-server; NGL=99
    log "midtyping serve: CUDA build --ngl 99 (card verified unclaimed)"
  else
    SRV=experiments/bin/llama/llama-b10453/llama-server; NGL=0
    log "midtyping serve: CPU build -ngl 0"
  fi
  "$SRV" -m "$GGUF" --port 18133 --host 127.0.0.1 -c 8192 --parallel 1 -t 8 -ngl $NGL \
    > /tmp/b13_midtyping_server.log 2>&1 &
  SPID=$!; ready=0
  for _ in $(seq 1 150); do
    curl -s -m 2 -X POST http://127.0.0.1:18133/completion \
      -d '{"prompt":"r","n_predict":1}' >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" = 1 ]; then
    for align in raw suffix; do
      suf=""; [ "$align" = suffix ] && suf="_suffix"
      .venv-sft/bin/python experiments/eval/run_eval.py --port 18133 --model zeta2 \
        --examples /mnt/h/sepalith/datasets/edit_pairs_v1/eval.jsonl \
        --variant midtyping --align "$align" --limit 18 \
        > "experiments/eval/results_${STEM}_midtyping${suf}.jsonl" \
        2>>"$BATT_LOG" && log "midtyping $align OK" || log "midtyping $align FAIL"
    done
  else
    log "midtyping server-not-ready (see /tmp/b13_midtyping_server.log)"
  fi
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null

  # row-id join check (runbook): the 18 row ids must join the banked B-series
  # midtyping files (first-18 of the CURRENT eval.jsonl, drifted-vs-v7 caveat)
  .venv-sft/bin/python - "$STEM" <<'PYEOF' >> "$BATT_LOG" 2>&1 && log "join-check OK" || log "join-check FAIL"
import json, sys
stem = sys.argv[1]
def ids(p):
    try:
        return [json.loads(l)["id"] for l in open(p)]
    except FileNotFoundError:
        return []
new = ids(f"experiments/eval/results_{stem}_midtyping.jsonl")
banked = ids("experiments/eval/results_b4_qwen35_2b_midtyping.jsonl")
print(f"join-check: {len(new)} rows; overlap with b4 banked: {len(set(new) & set(banked))}")
assert len(new) == 18, f"expected 18 rows, got {len(new)}"
assert len(set(new) & set(banked)) == 18, "row ids do not join the banked B-series midtyping set"
print("join-check PASS (18/18 ids identical to b4's)")
PYEOF

  # 3c. noop FP (v8 metric; script-owned CPU server)
  .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu \
    >> "$BATT_LOG" 2>&1 && log "noop OK" || log "noop FAIL"

  # 3d. decode: llama-bench t8 on the CPU build (banked comparability row;
  #     reference: b4 19.2 t/s bar, b3_lfm25_350m 94.1)
  experiments/bin/llama/llama-b10453/llama-bench -m "$GGUF" -t 8 -p 512 -n 128 \
    >> "$BATT_LOG" 2>&1 && log "bench OK" || log "bench FAIL"

  log "BATTERY done $STEM"
) 9>/tmp/b_battery.lock

log "=== B13 CHAIN END ==="
