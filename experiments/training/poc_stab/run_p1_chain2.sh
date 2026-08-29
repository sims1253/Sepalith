#!/usr/bin/env bash
# P1 arm chain v2 (zcode-stabtok): resumes the three arms that hold a
# step-200 checkpoint after the quick_eval OOM (see board 2026-08-29 17:0x),
# runs nesterov fresh, then the stress-scorer pass.
# Fixes vs v1: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + mem-frac
# 0.65 (the step-400 quick_eval materializes a ~4GB logits tensor that
# fragmented-reserved memory could not satisfy under the 17.5GB cap), and
# per-arm latest.pt cleanup after success (disk: /tmp 98% full).
set -u
ROOT=/home/m0hawk/Documents/Sepalith
RESULTS=$ROOT/experiments/training/poc_stab/RESULTS.md
LAD=/tmp/poc_twin/ladder
cd "$ROOT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGD=$ROOT/experiments/training/poc_twin/ladder/logs
BOARD=$ROOT/comms/board.md
TS() { date "+%Y-%m-%dT%H:%M%z"; }
STEPS=480
DOSE=0.3
MEMFRAC=0.65
OUT=/tmp/p1_chain.out

heartbeat_loop() {
  while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 1800
    local latest
    latest=$(ls -t $LOGD/p1_*.jsonl 2>/dev/null | head -1)
    local state
    state=$(tail -1 "$latest" 2>/dev/null | cut -c1-100)
    echo "[ $(TS) ] zcode-stabtok HEARTBEAT p1-chain2 pid $CHAIN_PID ${latest##*/} :: ${state:-starting}" >> "$BOARD"
  done
}

run_arm() {
  local arm=$1 tag="p1_$1" resume="$2"
  local extra=()
  [ -n "$resume" ] && extra=(--resume "$resume")
  echo "[ $(TS) ] CHAIN2 ARM $arm start (resume=${resume:-none})" >> "$OUT"
  nice -n 5 uv run --project "$ROOT" python \
    experiments/training/poc_stab/train_p1.py \
    --arm "$arm" --dose $DOSE --steps $STEPS --tag "$tag" \
    --mem-frac $MEMFRAC "${extra[@]}" > "/tmp/p1_${arm}.out" 2>&1
  local rc=$?
  echo "[ $(TS) ] CHAIN2 ARM $arm exit $rc" >> "$OUT"
  if [ $rc -eq 0 ] && [ -f "$LAD/ckpt_${tag}/final.pt" ]; then
    rm -f "$LAD/ckpt_${tag}/latest.pt"   # keep final.pt only (disk)
  else
    echo "[ $(TS) ] zcode-stabtok NOTE p1-$arm chain2 rc=$rc (see /tmp/p1_${arm}.out)" >> "$BOARD"
    tail -4 "/tmp/p1_${arm}.out" >> "$BOARD"
    FAILED="$FAILED $arm"
  fi
}

FAILED=""
CHAIN_PID=$$
heartbeat_loop &
HB_PID=$!

run_arm control "$LAD/ckpt_p1_control/latest.pt"
run_arm split   "$LAD/ckpt_p1_split/latest.pt"
run_arm polar   "$LAD/ckpt_p1_polar/latest.pt"
run_arm nesterov ""

kill "$HB_PID" 2>/dev/null
echo "[ $(TS) ] CHAIN2 scorer pass" >> "$OUT"
for arm in split polar nesterov; do
  uv run --project "$ROOT" python \
    experiments/training/poc_stab/stress_metrics.py \
    "$LOGD/p1_control.jsonl" "$LOGD/p1_${arm}.jsonl" \
    --clip 1.0 --skip-first 100 >> "$RESULTS" 2>&1 \
    || echo "scorer failed for $arm" >> "$RESULTS"
  echo "" >> "$RESULTS"
done
echo "[ $(TS) ] zcode-stabtok NOTE p1-chain2 DONE (failed:${FAILED:-none}). Verdict pending per adoption rules; GPU claim release follows." >> "$BOARD"
echo "[ $(TS) ] CHAIN2 DONE failed:${FAILED:-none}" >> "$OUT"
