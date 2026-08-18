#!/bin/zsh
# Keystroke sim: fresh server per ctx size so 'cold' is genuinely cold.
BIN=bin/llama/llama-b10453/llama-server
MODEL=$1; PORT=$2; THREADS=${3:-8}; OUT=$4
: > $OUT
for CTX in 2048 4096 8192; do
  $BIN -m $MODEL --port $PORT -t $THREADS -c $((CTX + 1024)) -ngl 0 --host 127.0.0.1 > /tmp/llama-server-sim.log 2>&1 &
  SRV=$!
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 1; done
  python3 eval/keystroke_sim.py --port $PORT --ctx $CTX >> $OUT
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
done
