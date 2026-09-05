#!/usr/bin/env bash
# TU2 verdict legs — queue §3 TU2 (zcode-tu2-eval, 2026-09-05).
# CPU-ONLY chain: b113 adapter poll/pull -> PEFT re-merge + Q8_0 export per
# arm (tu2_a624 / tu2_b624 / tu2_c113 / tu2_b113) -> battery under
# /tmp/b_battery.lock (eval_scenarios shared five-family slice + eval_noop_fp
# guardrail + llama-bench t/s row) -> mirror to /mnt/h/sepalith/runs/tu2_*.
# NO CUDA context by construction: CUDA_VISIBLE_DEVICES="" for the whole
# chain (card belongs to zcode-o1-run's GRPO chain; comms/gpu.md).
# Verdict math (McNemar etc.) is NOT here — readout agent-side after.
set -u
cd /home/m0hawk/Documents/Sepalith
export CUDA_VISIBLE_DEVICES=""
export LLAMA_CONVERT=/home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453/convert_hf_to_gguf.py
RUNS=/mnt/h/sepalith/runs
LOG=$RUNS/tu2_verdict_chain.log
REPO=scholzmx/sepalith-lora
mkdir -p "$RUNS"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

pull() { # <stem> <hub-run-name> -> $RUNS/tu2_<stem>/final_lora/
  .venv-sft/bin/python - "$1" "$2" "$REPO" <<'PY' >> "$LOG" 2>&1
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
stem, run, repo = sys.argv[1:4]
dst = Path(f"/mnt/h/sepalith/runs/tu2_{stem}/final_lora")
dst.mkdir(parents=True, exist_ok=True)
for fn in ("adapter_config.json", "adapter_model.safetensors"):
    p = hf_hub_download(repo, f"{run}/final_lora/{fn}")
    shutil.copyfile(p, dst / fn)
print(f"[pull] {stem} <- {run}/final_lora OK")
PY
}

export_arm() { # <stem> (adapter already at $RUNS/tu2_<stem>/final_lora)
  log "EXPORT start tu2_$1"
  if MERGE_VIA_PEFT=1 .venv-sft/bin/python experiments/training/export_gguf.py \
       scholzmx/sepalith-base-qwen35-2b-text "$RUNS/tu2_$1/final_lora" "tu2_$1" \
       >> "$RUNS/tu2_$1/export.log" 2>&1; then
    rm -rf "/tmp/merged_tu2_$1"
    log "EXPORT done -> experiments/models/tu2_$1-Q8_0.gguf"
  else
    log "EXPORT FAIL tu2_$1 (see $RUNS/tu2_$1/export.log)"
    return 1
  fi
}

battery() { # <stem> — CPU servers, serialized with every other battery
  local GGUF=$PWD/experiments/models/tu2_$1-Q8_0.gguf
  if [ ! -f "$GGUF" ]; then log "BATTERY SKIP tu2_$1 (no gguf)"; return 1; fi
  (
    flock 9
    log "BATTERY start tu2_$1 (flock held)"
    # scenarios: shared 307-row five-family slice, banked 150/family cap (255 scored)
    .venv/bin/python experiments/eval/eval_scenarios.py --model "$GGUF" \
      --threads 8 --port 18090 \
      >> "$RUNS/tu2_$1/battery.log" 2>&1 && log "scen OK tu2_$1" || log "scen FAIL tu2_$1"
    # noopFP guardrail (pre-registered: must not be worse than a624's)
    .venv/bin/python experiments/eval/eval_noop_fp.py --model "$GGUF" --spawn-cpu \
      --port 18095 \
      >> "$RUNS/tu2_$1/battery.log" 2>&1 && log "noop OK tu2_$1" || log "noop FAIL tu2_$1"
    # decode t/s row (same rig across arms = comparable; load documented)
    experiments/bin/llama/llama-b10453/llama-bench -m "$GGUF" -t 8 -p 512 -n 128 \
      >> "$RUNS/tu2_$1/bench.log" 2>&1 && log "bench OK tu2_$1" || log "bench FAIL tu2_$1"
    log "BATTERY done tu2_$1 (flock released)"
  ) 9>/tmp/b_battery.lock
}

log "=== TU2 VERDICT CHAIN START (CPU-only) pid $$ ==="

# ---- 1) the three landed arms: pull + export -----------------------------
for spec in a624:tu2_a_raw_624 b624:tu2_b_solve_gated_624 c113:tu2_c_teacher_target; do
  stem=${spec%%:*}; run=${spec##*:}
  pull "$stem" "$run" && export_arm "$stem" || log "arm $stem pull/export FAILED"
done

# ---- 2) supplemental b113: poll for the adapter (job fired ~08:3x) -------
got=""
for i in $(seq 1 45); do
  if pull b113 tu2_b_solve_gated_113; then got=1; break; fi
  [ $i -eq 1 ] && log "b113 not on hub yet — polling (max 45 min)"
  sleep 60
done
if [ -n "$got" ]; then
  export_arm b113 || log "arm b113 export FAILED"
else
  log "b113 adapter NEVER landed after 45 min — chain proceeds with 3 arms (b113 battery skipped)"
fi

# ---- 3) battery per arm (a624 first: the noopFP guardrail baseline) ------
for stem in a624 b624 c113 b113; do
  [ -f "experiments/models/tu2_${stem}-Q8_0.gguf" ] && battery "$stem"
done

# ---- 4) mirror GGUFs + results next to the adapters ----------------------
for stem in a624 b624 c113 b113; do
  [ -f "experiments/models/tu2_${stem}-Q8_0.gguf" ] || continue
  cp -f "experiments/models/tu2_${stem}-Q8_0.gguf" "$RUNS/tu2_${stem}/" 2>>"$LOG"
  for f in experiments/eval/results_scenarios_tu2_${stem}.jsonl \
           experiments/eval/results_noop_fp_tu2_${stem}.jsonl \
           experiments/eval/llama-server-scenarios-tu2_${stem}.log; do
    [ -f "$f" ] && cp -f "$f" "$RUNS/tu2_${stem}/" 2>>"$LOG"
  done
  log "mirror tu2_$stem done"
done

log "=== TU2 VERDICT CHAIN END ==="
