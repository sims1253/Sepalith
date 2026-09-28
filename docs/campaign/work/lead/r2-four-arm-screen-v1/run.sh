#!/usr/bin/env bash
set -euo pipefail

# Root-only future command. Run from an ext4 run directory under an external
# watchdog; this packet preparation did not launch a server or touch CUDA.
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
SERVER=/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-cuda-b10453/llama-server
TARGET_Q8=/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf
TRAINED_DSPARK_Q8=${TRAINED_DSPARK_Q8:?bind a root-header-verified step_8 conversion output}
PROBE="$PLAN/docs/campaign/work/serving-readiness/runtime_native_probe.py"
DSPARK_PROBE="$PLAN/docs/campaign/work/lead/r2-trained-draft-serving-preparation-v1/dspark_probe.py"
ANALYZER="$PLAN/docs/campaign/work/lead/r2-trained-draft-serving-preparation-v1/analyze_pair.py"
FIXTURE="$PLAN/docs/campaign/work/serving-readiness/native-probe-train-fixture.jsonl"
FIXTURE_MANIFEST="$PLAN/docs/campaign/work/serving-readiness/native-probe-train-fixture.manifest.json"
RUN=${RUN:?set a fresh ext4 benchmark output directory}
PORT=${PORT:-18411}
RELEASED_DSPARK_MODEL=${RELEASED_DSPARK_MODEL:-}
RUN_RELEASED=${RUN_RELEASED:-0}

test -x "$SERVER"
test -f "$TARGET_Q8"
test -f "$PROBE"
test -f "$DSPARK_PROBE"
test -f "$ANALYZER"
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

run_baseline_probe() {
  local arm="$1"
  local out="$2"
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
    "$PYTHON" -B "$PROBE" \
    --fixture "$FIXTURE" --manifest "$FIXTURE_MANIFEST" \
    --url "http://127.0.0.1:$PORT" --arm baseline \
    --cap 192 --context 4096 --reps 1 --user-deadline-ms 5000 \
    --diagnostic-timeout-ms 60000 --out "$out"
}

run_extra_probe() {
  local label="$1"
  local out="$2"
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
    "$PYTHON" -B "$DSPARK_PROBE" --probe "$PROBE" \
    --fixture "$FIXTURE" --manifest "$FIXTURE_MANIFEST" \
    --url "http://127.0.0.1:$PORT" --arm "$label" \
    --cap 192 --context 4096 --reps 1 --user-deadline-ms 5000 \
    --diagnostic-timeout-ms 60000 --out "$out"
}

# ordinary-baseline
start_server ordinary-baseline
run_baseline_probe ordinary-baseline "$RUN/ordinary-baseline.json"
stop_server

# model-free-ngram; this arm uses the packet adapter so its final counters are
# retained when b10453 exposes them.
start_server model-free-ngram \
  --spec-type ngram-mod --spec-draft-n-max 64 \
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
run_extra_probe model-free-ngram "$RUN/model-free-ngram.json"
stop_server

if [[ "$RUN_RELEASED" == 1 ]]; then
  # released-dspark is a conditional comparator. Root must provide a concrete
  # file plus an independent GGUF header/hash receipt before this branch.
  start_server released-dspark -md "$RELEASED_DSPARK_MODEL" \
    --spec-type draft-dspark --spec-draft-n-max 7
  run_extra_probe released-dspark "$RUN/released-dspark.json"
  stop_server
fi

# trained-targetmatched-dspark
start_server trained-dspark -md "$TRAINED_DSPARK_Q8" \
  --spec-type draft-dspark --spec-draft-n-max 7
run_extra_probe trained-dspark "$RUN/trained-dspark.json"
stop_server

candidate_args=(--candidate "model-free-ngram=$RUN/model-free-ngram.json"
                --candidate "trained-dspark=$RUN/trained-dspark.json")
if [[ "$RUN_RELEASED" == 1 ]]; then
  candidate_args+=(--candidate "released-dspark=$RUN/released-dspark.json")
fi
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  "$PYTHON" -B "$ANALYZER" \
  --baseline "$RUN/ordinary-baseline.json" "${candidate_args[@]}" \
  --out "$RUN/pair-analysis.json"

printf '%s\n' "pair benchmark complete: $RUN/pair-analysis.json"
