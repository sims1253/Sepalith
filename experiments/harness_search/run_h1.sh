#!/bin/bash
# H1 bake-off pipeline — sequential phases on ONE CPU llama-server (port 18310,
# the plan's serving convention + --cache-reuse 1024; see H1_RESULTS.md for the
# deviation note). Resumable per phase: methods.py checkpoints per iteration;
# rerunning a phase continues its state.json. Kill by tracked PID only.
set -u
cd /home/m0hawk/Documents/Sepalith
exec 9>/tmp/h1_pipeline.lock; flock -n 9 || { echo "pipeline already running"; exit 1; }

PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python
RIG=/home/m0hawk/Documents/Sepalith/experiments/harness_search
RES=$RIG/results
SERVER_PID_FILE=$RES/server.pid
mkdir -p "$RES"

# ---- server (tracked PID; CPU-only binary, no CUDA context) ----------------
if ! ss -tln 2>/dev/null | grep -q ":18310 "; then
  nohup experiments/bin/llama/llama-b10453/llama-server \
    -m experiments/models/sft_v7_minicpm5-Q8_0.gguf \
    --port 18310 --host 127.0.0.1 -t 8 --parallel 1 -c 8192 -ngl 0 \
    --cache-reuse 1024 >> "$RES/llama-server-h1-18310.log" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
  echo "$(date '+%F %T') server started pid $(cat "$SERVER_PID_FILE")"
fi

# ---- phases -----------------------------------------------------------------
echo "$(date '+%F %T') phase: baseline"
nice -n 15 "$PY" "$RIG/methods.py" --arm baseline --results "$RES" \
  --no-server > "$RES/baseline.log" 2>&1 || { echo baseline FAILED; exit 2; }

for ARM in hill population gepa; do
  echo "$(date '+%F %T') phase: $ARM"
  nice -n 15 "$PY" "$RIG/methods.py" --arm "$ARM" --iters 13 \
    --results "$RES" --no-server > "$RES/$ARM.log" 2>&1 \
    || { echo "$ARM FAILED"; exit 3; }
done

echo "$(date '+%F %T') phase: verdict battery"
nice -n 15 "$PY" "$RIG/verdict_battery.py" --results "$RES" \
  --out verdict.json > "$RES/verdict.log" 2>&1 || { echo verdict FAILED; exit 4; }

date '+%F %T' > "$RES/PIPELINE_DONE"
echo "$(date '+%F %T') PIPELINE DONE"
