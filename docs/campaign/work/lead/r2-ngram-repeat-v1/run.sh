#!/usr/bin/env bash
set -euo pipefail

# Root-only future command. Run from an ext4 run directory under an external
# watchdog; this packet preparation did not launch a server or touch CUDA.
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
SERVER=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server
TARGET_Q8=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf
TRAINED_DSPARK_Q8=${TRAINED_DSPARK_Q8:?bind a root-header-verified step_8 conversion output}
PROBE="$(dirname "$0")/paired_panel_probe.py"
FIXTURE="$(dirname "$0")/train-serving-panel.jsonl"
FIXTURE_MANIFEST="$(dirname "$0")/train-serving-panel.manifest.json"
RUN=${RUN:?set a fresh ext4 benchmark output directory}
PORT=${PORT:-18411}
RELEASED_DSPARK_MODEL=${RELEASED_DSPARK_MODEL:-}
RUN_RELEASED=${RUN_RELEASED:-0}

test -x "$SERVER"
test -f "$TARGET_Q8"
test -f "$PROBE"
test -f "$FIXTURE"
test -f "$FIXTURE_MANIFEST"
test -d "$RUN" && test -z "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)"
if [[ "$RUN_RELEASED" == 1 ]]; then
  test -n "$RELEASED_DSPARK_MODEL"
  test -f "$RELEASED_DSPARK_MODEL"
fi
mkdir -p "$RUN"

common=(--host 127.0.0.1 --port "$PORT" -t 6 -tb 6 --threads-http 2
        --parallel 1 -c 4096 -b 256 -ub 256 -ngl 99 -ngld 99 -lv 4)
SERVER_PID=
stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..5}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server process survived cleanup" >&2
      return 1
    fi
    SERVER_PID=
  fi
}
cleanup() { stop_server || true; }
on_signal() { cleanup; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

start_server() {
  local name="$1"
  shift
  stop_server
  setsid env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_GRAPH_OPT=0 \
    "$SERVER" -m "$TARGET_Q8" "${common[@]}" "$@" \
    >"$RUN/${name}-server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" >"$RUN/${name}-server.pid"
  for _ in {1..60}; do
    if curl --max-time 2 --fail --silent --show-error "http://127.0.0.1:$PORT/health" \
        >"$RUN/${name}-health.json" 2>/dev/null; then
      curl --max-time 2 --fail --silent --show-error "http://127.0.0.1:$PORT/props" \
        >"$RUN/${name}-props.json"
      return 0
    fi
    sleep 1
  done
  echo "server did not become healthy: $name" >&2
  return 1
}

run_panel_probe() {
  local arm="$1"
  local out="$2"
  local code=0
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PYTHON" -B "$PROBE" \
    --panel "$FIXTURE" --manifest "$FIXTURE_MANIFEST" \
    --url "http://127.0.0.1:$PORT" --cap 192 --context 4096 \
    --reps 1 --deadline-ms 5000 --out "$out" || code=$?
  printf '%s\n' "$code" > "$RUN/$arm-client-exit.txt"
  # Exit1 is measured protocol failure; exit2 is invalid inputs/infrastructure.
  if (( code > 1 )); then return "$code"; fi
}

# Reverse arm order relative to the first panel.
start_server model-free-ngram --spec-type ngram-mod --spec-draft-n-max 64 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
run_panel_probe model-free-ngram "$RUN/model-free-ngram.json"
stop_server
start_server ordinary-baseline
run_panel_probe ordinary-baseline "$RUN/ordinary-baseline.json"
stop_server
