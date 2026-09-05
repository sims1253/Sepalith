#!/usr/bin/env bash
# B8 — AST-FIM midtrain re-probe on the gate-B-β winner base (GDN/Qwen3.5-2B
# b4-config). Runbook: docs/research/2026-08-31-base-bakeoff-plan.md §2 B8.
# Arm = the BANKED b4_qwen35_2b recipe (3000 steps, LoRA r32/a64, lr 2e-4,
# cosine, seq 2048, seed 3407, 48k-row shuffle(42) cap, same SFT_TARGETS
# list) but on /mnt/h/sepalith/datasets/astfim_v1 with MIDTRAIN_MASK=1 —
# completion-only loss masking + conservative bucket packing
# (train_sampling_strategy=group_by_length; MIDTRAIN_PACK=seq is REFUSED for
# GDN model types by midtrain_data.assert_midtrain_safe — recurrent-state
# bleed). Paired control = the banked b4_qwen35_2b rung (same base, sft_v7,
# no midtrain).
#
# PRE-REGISTERED HEALTH SIGNATURE (queue-mgr brief 2026-09-05; if the seam
# checks fail we do NOT train a broken instrument — that is the exact bug B8
# exists to fix):
#   [midtrain:train] ~100% prefix-route (astfim_v1 root rows end prompt with
#   <|end|>\n), token-seam exact ≈ 48000/48000, ~13-14% completion share
#   (b8-patch measured 13.5% on the real corpus), finite step-1 loss.
# GATES implemented below:
#   A) attachment: "Trainable parameters" == 21,823,488 of 1,903,648,576
#      (the measured b4 line; same base + same target list) [B3 incident rule]
#   B) [midtrain:train] telemetry within the pre-registered envelope
#      (prefix-route ≥99% of routed, seam exact ≥99.5%, completion 10-16%)
#   C) post-train nan/inf loss scan
# Knobs: unsloth-with-knobs per the B4 saga (compile + auto-padding-free OFF
# — mandatory, asserted by midtrain_data.assert_midtrain_safe too).
# OOM fallback: SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math;
# B13 incident precedent). Battery = CPU convention under flock
# /tmp/b_battery.lock, port 18158; midtyping --limit 18 with (i,sha) join
# check vs the banked b4 rows.
#
# Launch pattern (B13 precedent, 2026-09-05): the CHAIN runs detached
# (setsid nohup); the session tracks a small watcher, never the workload.
set -u
cd /home/m0hawk/Documents/Sepalith || exit 1

STEM=b8_midtrain_qwen35_2b
MODEL=experiments/models/qwen3.5-2b-base-text-hf
DATA=/mnt/h/sepalith/datasets/astfim_v1
RUNS=/mnt/h/sepalith/runs
LOG=$RUNS/b8_chain.log
TRAIN_LOG=$RUNS/${STEM}_train.log
EXPORT_LOG=$RUNS/${STEM}_export.log
BATT_LOG=$RUNS/${STEM}_battery.log
EXPECT_TRAINEE=21823488   # the measured b4 line: 21,823,488 of 1,903,648,576 (1.15%)

export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
export UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SFT_TARGETS='q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj,gate_proj,up_proj,down_proj'
export MIDTRAIN_MASK=1

TS() { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(TS)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNS"
log "=== B8 CHAIN START (stem $STEM, model $MODEL, data $DATA, MIDTRAIN_MASK=1) ==="

# ---------- 1) TRAIN ----------
rm -rf "$RUNS/$STEM"
log "B8 TRAIN start (expected trainable $EXPECT_TRAINEE)"
.venv-sft/bin/python experiments/training/train_sft.py \
  "$MODEL" 3000 "$DATA" "$RUNS/$STEM" "" >> "$TRAIN_LOG" 2>&1 &
TPID=$!
log "B8 trainer pid $TPID (log $TRAIN_LOG)"

# Gate A: wait for the attachment line, verify the count
gate=0
for _ in $(seq 1 90); do   # up to 30 min for load + get_peft_model + dataset map
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B8 TRAIN DIED before attachment line (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  LINE=$(grep -m1 "Trainable parameters" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    N=$(echo "$LINE" | grep -oE "[0-9,]+ of" | head -1 | tr -d ' ,of')
    log "B8 ATTACHMENT LINE: $LINE"
    if [ "$N" = "$EXPECT_TRAINEE" ]; then
      log "B8 GATE-A PASS (trainable $N == $EXPECT_TRAINEE)"
      gate=1
    else
      log "B8 GATE-A FAIL: trainable ${N:-?} != $EXPECT_TRAINEE — KILLING trainer $TPID"
      kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
      exit 2
    fi
    break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B8 GATE-A TIMEOUT (no attachment line in 30m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 3
fi

# Gate B: [midtrain:train] telemetry within the pre-registered envelope.
# Line shape (train_sft.py L109):
#   [midtrain:train] N0 rows in -> K kept (prefix-route P, suffix-route S; token-seam exact E/R); T tokens, L loss tokens = C% completion
gate=0
for _ in $(seq 1 240); do   # up to 80 min for the 48k-row map (2 tokenizations/row)
  if ! kill -0 "$TPID" 2>/dev/null; then
    log "B8 TRAIN DIED before midtrain telemetry (see $TRAIN_LOG)"; wait "$TPID"; exit 1
  fi
  LINE=$(grep -m1 "\[midtrain:train\]" "$TRAIN_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    log "B8 MIDTRAIN TELEMETRY: $LINE"
    python3 - "$LINE" <<'PYEOF' && gate=1 || gate=2
import re, sys
line = sys.argv[1]
m = re.search(r"(\d+) rows in -> (\d+) kept \(prefix-route (\d+), suffix-route (\d+); token-seam exact (\d+)/(\d+)\); (\d+) tokens, (\d+) loss tokens = ([\d.]+)% completion", line)
if not m:
    print("PARSE-FAIL: telemetry line did not match the expected shape"); sys.exit(1)
n0, kept, p, s, e, r, tok, loss_tok, comp = map(lambda x: int(float(x)), m.groups())
routed = p + s
ok = True
def chk(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond
chk(n0 == 48000, f"rows-in {n0} == 48000 (shuffle(42) cap)")
chk(routed > 0 and p / routed >= 0.99, f"prefix-route {p}/{routed} = {100*p/max(routed,1):.2f}% >= 99% (pre-registered ~100%)")
chk(e / max(routed, 1) >= 0.995, f"token-seam exact {e}/{routed} = {100*e/max(routed,1):.2f}% >= 99.5% (pre-registered ~48000/48000)")
chk(10 <= comp <= 16, f"completion share {comp}% in [10,16] (pre-registered ~13-14%)")
chk(loss_tok > 0, f"loss tokens {loss_tok} > 0")
sys.exit(0 if ok else 1)
PYEOF
    if [ "$gate" = 2 ]; then
      log "B8 GATE-B FAIL (telemetry outside pre-registered envelope) — KILLING trainer $TPID; NOT training a broken instrument"
      kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
      exit 4
    fi
    log "B8 GATE-B PASS (telemetry within the pre-registered health envelope)"
    break
  fi
  sleep 20
done
if [ "$gate" != 1 ]; then
  log "B8 GATE-B TIMEOUT (no [midtrain:train] line in 80m) — KILLING trainer $TPID"
  kill "$TPID" 2>/dev/null; sleep 5; kill -9 "$TPID" 2>/dev/null
  exit 5
fi

if wait "$TPID"; then
  log "B8 TRAIN done"
else
  rc=$?
  log "B8 TRAIN FAIL rc=$rc (see $TRAIN_LOG). If CUDA OOM / VRAM creep: relaunch SFT_PD_BATCH=2 SFT_GRAD_ACCUM=8 (identical optimizer math) and note it."
  exit 6
fi

# Gate C: nan/inf loss scan over the emitted loss lines
if grep -E "'loss': (nan|inf)" "$TRAIN_LOG" >/dev/null 2>&1; then
  log "B8 GATE-C FAIL: non-finite loss found in $TRAIN_LOG — arm INVALID, not exporting"
  exit 7
fi
log "B8 GATE-C PASS (no nan/inf loss lines)"
grep -m1 "\[midtrain:eval\]" "$TRAIN_LOG" | tee -a "$LOG" || true

# ---------- 2) EXPORT (Q8_0; standard unsloth merge + b10453 converter) ----------
sleep 30
log "B8 EXPORT start"
if .venv-sft/bin/python experiments/training/export_gguf.py "$MODEL" \
     "$RUNS/$STEM/final_lora" "$STEM" >> "$EXPORT_LOG" 2>&1; then
  log "B8 EXPORT done -> experiments/models/$STEM-Q8_0.gguf"
else
  log "B8 EXPORT FAIL (see $EXPORT_LOG)"; exit 8
fi

# ---------- 3) BATTERY (CPU convention; serialized via flock; port 18158) ----------
GGUF=experiments/models/$STEM-Q8_0.gguf
(
  flock 9
  log "BATTERY start $STEM (CPU convention)"
  .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" --threads 8 \
    >> "$BATT_LOG" 2>&1 && log "scen OK" || log "scen FAIL"
  .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu \
    >> "$BATT_LOG" 2>&1 && log "noop OK" || log "noop FAIL"
  experiments/bin/llama/llama-b10453/llama-server -m "$GGUF" --port 18158 --host 127.0.0.1 \
    -c 8192 --parallel 1 -t 8 -ngl 0 > /tmp/b8_midtyping_server.log 2>&1 &
  SPID=$!; ready=0
  for _ in $(seq 1 150); do
    curl -s -m 2 -X POST http://127.0.0.1:18158/completion -d '{"prompt":"r","n_predict":1}' >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" = 1 ]; then
    for align in raw suffix; do
      suf=""; [ "$align" = suffix ] && suf="_suffix"
      .venv-sft/bin/python experiments/eval/run_eval.py --port 18158 --model zeta2 \
        --examples /mnt/h/sepalith/datasets/edit_pairs_v1/eval.jsonl \
        --variant midtyping --align "$align" --limit 18 \
        > "experiments/eval/results_${STEM}_midtyping${suf}.jsonl" 2>/dev/null
    done
    log "midtyping OK (raw+suffix, limit 18)"
  else log "midtyping server-not-ready"; fi
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  experiments/bin/llama/llama-b10453/llama-bench -m "$GGUF" -t 8 -p 512 -n 128 \
    >> "$BATT_LOG" 2>&1 && log "bench OK" || log "bench FAIL"
) 9>/tmp/b_battery.lock
log "=== B8 CHAIN END ==="
