#!/usr/bin/env bash
# P1 arm chain (zcode-stabtok): 4 sequential arms on the 206M ladder
# instrument, pre-registered design per docs/research/2026-08-29-papers-recon-
# poc-plan.md. The GPU claim for this chain is posted by the launching agent
# in comms/gpu.md; this script appends board HEARTBEATs every 30 min and a
# final scorer pass (stress_metrics, CPU) into poc_stab/RESULTS.md.
#
# Usage: nohup setsid bash run_p1_chain.sh > /tmp/p1_chain.out 2>&1 &
set -u
ROOT=/home/m0hawk/Documents/Sepalith
RESULTS=$ROOT/experiments/training/poc_stab/RESULTS.md
cd "$ROOT" || exit 1
LOGD=$ROOT/experiments/training/poc_twin/ladder/logs
BOARD=$ROOT/comms/board.md
TS() { date "+%Y-%m-%dT%H:%M%z"; }
STEPS=480
DOSE=0.3
MEMFRAC=0.55
ARMS="control split polar nesterov"

heartbeat_loop() {
  while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 1800
    local latest
    latest=$(ls -t $LOGD/p1_*.jsonl 2>/dev/null | head -1)
    local state
    state=$(tail -1 "$latest" 2>/dev/null | cut -c1-100)
    echo "[ $(TS) ] zcode-stabtok HEARTBEAT p1-chain pid $CHAIN_PID ${latest##*/} :: ${state:-starting}" >> "$BOARD"
  done
}

run_arm() {
  local arm=$1 tag="p1_$1"
  echo "[ $(TS) ] ARM $arm start" >> /tmp/p1_chain.out
  nice -n 5 uv run --project "$ROOT" python \
    experiments/training/poc_stab/train_p1.py \
    --arm "$arm" --dose $DOSE --steps $STEPS --tag "$tag" \
    --mem-frac $MEMFRAC > "/tmp/p1_${arm}.out" 2>&1
  local rc=$?
  echo "[ $(TS) ] ARM $arm exit $rc" >> /tmp/p1_chain.out
  if [ $rc -ne 0 ]; then
    echo "[ $(TS) ] zcode-stabtok NOTE p1 arm $arm FAILED (rc=$rc, see /tmp/p1_${arm}.out tail below)" >> "$BOARD"
    tail -5 "/tmp/p1_${arm}.out" >> "$BOARD"
    FAILED="$FAILED $arm"
  fi
}

FAILED=""
CHAIN_PID=$$
heartbeat_loop &
HB_PID=$!

for arm in $ARMS; do
  run_arm "$arm"
done

kill "$HB_PID" 2>/dev/null
echo "[ $(TS) ] scorer pass" >> /tmp/p1_chain.out
for arm in split polar nesterov; do
  uv run --project "$ROOT" python \
    experiments/training/poc_stab/stress_metrics.py \
    "$LOGD/p1_control.jsonl" "$LOGD/p1_${arm}.jsonl" \
    --clip 1.0 --skip-first 100 >> "$RESULTS" 2>&1 \
    || echo "scorer failed for $arm" >> "$RESULTS"
  echo "" >> "$RESULTS"
done
echo "[ $(TS) ] zcode-stabtok NOTE p1-chain DONE (failed:${FAILED:-none}). Verdict pending per adoption rules; GPU claim release follows." >> "$BOARD"
echo "[ $(TS) ] CHAIN DONE failed:${FAILED:-none}" >> /tmp/p1_chain.out
