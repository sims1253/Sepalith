#!/bin/bash
# FIM-Replica arm (queue §2, 2026-09-05, zcode-gpushorts): the masked-loss
# @35% twin of the banked full-loss ladder_fim35 — the untested leg of the
# adopted 20-35% FIM-dose verdict's pre-registered masked-vs-unmasked >=2x
# line-F1 gate. NOTE: the queue row's "unmasked-FIM probe2-replica" label is
# inverted vs repo state — ladder_fim35 IS the full-loss/unmasked arm (code
# verified); this masked twin is the only missing side of the gate.
# SAME recipe as the banked arm: 668 steps x 524,288 tok, dose 0.35,
# seed 1273, MASK_SEED 90210 nested slots (identical tokens+order), Muon
# .01 / embed 4e-3 / WSD / QK-Clip tau=100 / torch.compile. ONLY the loss
# discipline on FIM docs differs: CE on span+<|end|> only (design-A2 §5.2).
# Then: GGUF convert -> serve 18107 (parallel 4, -ngl 99) -> served span-F1
# -> bpb_eval appended to the ladder family file (banked arms compare).
set -u
cd /home/m0hawk/Documents/Sepalith/experiments/training/poc_twin/ladder
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python3
SERVER=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server
PORT=18107
TAG=ladder_fim35m
CKPT=/tmp/poc_twin/ladder/ckpt_${TAG}/final.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train (resumable; taskset keeps cores 0-15 for the CPU bench batch)
if [ ! -f "$CKPT" ]; then
  RESUME=""
  if [ -f /tmp/poc_twin/ladder/ckpt_${TAG}/latest.pt ]; then
    RESUME="--resume /tmp/poc_twin/ladder/ckpt_${TAG}/latest.pt"
  fi
  echo "[fimrep] training $TAG (masked loss, dose 0.35) $(date -Is)" >&2
  taskset -c 16-23 nice -n 5 $PY -u train_ladder.py --dose 0.35 --seed 1273 \
    --steps 668 --compile --loss-mask --tag $TAG $RESUME \
    > logs/${TAG}_stdout.log 2>&1 || { echo "[fimrep] TRAIN FAILED" >&2; exit 1; }
fi
echo "[fimrep] train done $(date -Is)" >&2
mkdir -p /mnt/h/sepalith/runs/ladder_fim35m
rsync -a /tmp/poc_twin/ladder/ckpt_${TAG}/final.pt /mnt/h/sepalith/runs/ladder_fim35m/ 2>/dev/null

# served eval (probe2-style, same 223 rows as every banked arm)
if ! grep -q "\"${TAG}\"" logs/fim_eval_summary.json 2>/dev/null; then
  $PY convert_gguf.py --ckpt "$CKPT" --out /tmp/poc_twin/ladder/${TAG}.gguf \
    > logs/${TAG}_convert.log 2>&1 || { echo "[fimrep] CONVERT FAILED" >&2; exit 1; }
  taskset -c 16-23 $SERVER -m /tmp/poc_twin/ladder/${TAG}.gguf --port $PORT \
    --host 127.0.0.1 -c 16384 --parallel 4 -ub 512 -t 8 -ngl 99 \
    > logs/${TAG}_server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    curl -s -o /dev/null -m 2 -X POST "http://127.0.0.1:${PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"x","max_tokens":1,"temperature":0}' && break
    sleep 2
    kill -0 $SPID 2>/dev/null || { echo "[fimrep] SERVER DIED" >&2; exit 1; }
  done
  taskset -c 16-23 $PY -u eval_fim_served.py --port $PORT --arm $TAG \
    --out logs/fim_eval_${TAG}.jsonl --summary logs/fim_eval_summary.json \
    > logs/${TAG}_fimeval.log 2>&1
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
fi
echo "[fimrep] served eval done $(date -Is)" >&2

# paired BPB + TF stop-accuracy (appends to the ladder family file)
taskset -c 16-23 $PY -u bpb_eval.py \
  --arms ${TAG}:$CKPT --out logs/bpb_eval.json > logs/${TAG}_bpb.log 2>&1
echo "[fimrep] ALL DONE $(date -Is)" >&2
