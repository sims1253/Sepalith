# Root-run only. This packet preparation does not start a server or touch
# CUDA. Run each model sequentially under the existing external watchdog and
# fresh ext4 RUN directory. The only arm difference is -m MODEL.
set -euo pipefail

PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET="$PLAN/docs/campaign/work/lead/r2-final-quant-root-run-v1"
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
SERVER=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server
PANEL="$PACKET/quant-panel.jsonl"
MANIFEST="$PACKET/quant-panel.manifest.json"
RUN=${RUN:?set a fresh ext4 output directory}
PORT=${PORT:-18414}
F16_MODEL=/mnt/e/sepalith/campaign-20260915/intermediate-f16/SFT11-task-global-b-500-quant/model-F16.gguf
F16_SHA=fe1c38a2b53519fb15eeb4ac36efdd7b6f60ad5a58451475b51308a93cb8a8ee
Q8_MODEL=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf
Q8_SHA=d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db

test -x "$SERVER" -a -x "$PYTHON" -a -f "$PANEL" -a -f "$MANIFEST"
test -f "$F16_MODEL" -a -f "$Q8_MODEL"
test -d "$RUN" && test -z "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)"
mkdir -p "$RUN"

common=(--host 127.0.0.1 --port "$PORT" --temp 0 --seed 0
        -t 6 -tb 6 --threads-http 2 --parallel 1 -c 4096 -b 256 -ub 256
        -ngl 99 -ngld 99 -lv 4)
SERVER_PID=
stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM -- "$SERVER_PID" 2>/dev/null || true
    for _ in {1..5}; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap stop_server EXIT
trap 'stop_server; exit 143' INT TERM

start_server() {
  local model="$1" log="$2"
  stop_server
  env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_GRAPH_OPT=0 \
    "$SERVER" -m "$model" "${common[@]}" >"$RUN/$log-server.log" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$RUN/$log-server.pid"
  for _ in {1..60}; do
    if curl --max-time 2 --fail --silent "http://127.0.0.1:$PORT/health" >"$RUN/$log-health.json"; then
      curl --max-time 2 --fail --silent "http://127.0.0.1:$PORT/props" >"$RUN/$log-props.json"
      return 0
    fi
    sleep 1
  done
  return 1
}

probe() {
  local arm="$1" model="$2" sha="$3"
  local code=0
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PYTHON" -B \
    "$PACKET/quant_pair_probe.py" --arm "$arm" --model-path "$model" \
    --model-sha256 "$sha" --url "http://127.0.0.1:$PORT" \
    --panel "$PANEL" --manifest "$MANIFEST" --cap 192 --context 4096 \
    --reps 1 --deadline-ms 5000 --out "$RUN/$arm.json" || code=$?
  printf '%s\n' "$code" >"$RUN/$arm-client-exit.txt"
  # Exit 1 is a measured protocol result; preserve it and continue to the
  # other quantization. Exit >1 is an invalid input/infrastructure failure.
  if (( code > 1 )); then return "$code"; fi
}

start_server "$F16_MODEL" f16
probe f16 "$F16_MODEL" "$F16_SHA" || exit $?
stop_server
start_server "$Q8_MODEL" q8
probe q8 "$Q8_MODEL" "$Q8_SHA" || exit $?
stop_server
