#!/bin/bash
# FIM dose ladder orchestrator: 4 arms x 350M tokens, SEQUENTIAL (one GPU job),
# resumable per arm. Per arm: train -> GGUF convert -> serve 18107 (parallel 4,
# -ngl 99) -> served span-F1 eval -> teardown (tracked PID only) -> next arm.
# After all arms: regurgitation canary + paired BPB eval.
set -u
cd /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
SERVER=/tmp/llamacpp-cuda-build/bin/llama-server
PORT=18107
STEPS=668
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN() {  # $1 = dose, $2 = tag
  local DOSE=$1 TAG=$2
  local CKPT=/tmp/poc_twin/ladder/ckpt_${TAG}/final.pt
  if [ -f "$CKPT" ]; then echo "[orch] $TAG already trained"; return 0; fi
  local RESUME=""
  if [ -f /tmp/poc_twin/ladder/ckpt_${TAG}/latest.pt ]; then
    RESUME="--resume /tmp/poc_twin/ladder/ckpt_${TAG}/latest.pt"
  fi
  echo "[orch] training $TAG dose=$DOSE (resume='${RESUME:-none}')" >&2
  nice -n 10 $PY -u train_ladder.py --dose "$DOSE" --tag "$TAG" \
    --steps $STEPS --compile $RESUME \
    > logs/${TAG}_stdout.log 2>&1
  local RC=$?
  if [ $RC -ne 0 ] || [ ! -f "$CKPT" ]; then
    echo "[orch] $TAG TRAIN FAILED rc=$RC" >&2; return $RC
  fi
  echo "[orch] $TAG train done" >&2
}

EVALARM() {  # $1 = tag
  local TAG=$1
  local GGUF=/tmp/poc_twin/ladder/${TAG}.gguf
  local CKPT=/tmp/poc_twin/ladder/ckpt_${TAG}/final.pt
  if [ -f logs/fim_eval_${TAG}.jsonl ] && [ -f logs/fim_eval_summary.json ]; then
    if grep -q "\"${TAG}\"" logs/fim_eval_summary.json; then
      echo "[orch] $TAG served eval already done"; return 0
    fi
  fi
  $PY convert_gguf.py --ckpt "$CKPT" --out "$GGUF" > logs/${TAG}_convert.log 2>&1 \
    || { echo "[orch] $TAG CONVERT FAILED" >&2; return 1; }
  $SERVER -m "$GGUF" --port $PORT --host 127.0.0.1 -c 16384 --parallel 4 \
    -ub 512 -t 8 -ngl 99 > logs/${TAG}_server.log 2>&1 &
  local SPID=$!
  echo "[orch] $TAG server pid=$SPID (port $PORT)" >&2
  for i in $(seq 1 120); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/health"; then break; fi
    sleep 2
    if ! kill -0 $SPID 2>/dev/null; then
      echo "[orch] $TAG SERVER DIED during warmup" >&2; return 1; fi
  done
  $PY -u eval_fim_served.py --port $PORT --arm "$TAG" \
    --out logs/fim_eval_${TAG}.jsonl --summary logs/fim_eval_summary.json \
    > logs/${TAG}_fimeval.log 2>&1
  kill $SPID 2>/dev/null   # our own tracked child only
  wait $SPID 2>/dev/null
  echo "[orch] $TAG served eval done, server torn down" >&2
}

for DOSE_TAG in "0.0 ladder_fim0" "0.10 ladder_fim10" "0.20 ladder_fim20" "0.35 ladder_fim35"; do
  set -- $DOSE_TAG
  TRAIN "$1" "$2" || exit 1
  EVALARM "$2" || echo "[orch] WARN: $2 eval failed (training continues)" >&2
done

# canary (all arms, indexes built once) + paired BPB
nice -n 10 $PY -u regurgitation_ladder.py \
  --arms ladder_fim0:/tmp/poc_twin/ladder/ckpt_ladder_fim0/final.pt \
         ladder_fim10:/tmp/poc_twin/ladder/ckpt_ladder_fim10/final.pt \
         ladder_fim20:/tmp/poc_twin/ladder/ckpt_ladder_fim20/final.pt \
         ladder_fim35:/tmp/poc_twin/ladder/ckpt_ladder_fim35/final.pt \
  --out logs/canary.json > logs/canary_stdout.log 2>&1

$PY -u bpb_eval.py \
  --arms ladder_fim0:/tmp/poc_twin/ladder/ckpt_ladder_fim0/final.pt \
         ladder_fim10:/tmp/poc_twin/ladder/ckpt_ladder_fim10/final.pt \
         ladder_fim20:/tmp/poc_twin/ladder/ckpt_ladder_fim20/final.pt \
         ladder_fim35:/tmp/poc_twin/ladder/ckpt_ladder_fim35/final.pt \
         muon_final_anchor:/home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/checkpoints/muon_final.pt \
  --out logs/bpb_eval.json > logs/bpb_eval_stdout.log 2>&1

echo "[orch] LADDER COMPLETE $(date)" >&2
