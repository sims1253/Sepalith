#!/usr/bin/env bash
# Root-only managed launcher template. This file is never invoked by the
# preparation worker. It starts one already-admitted theta0 server when root
# explicitly runs it on m0pad, with a fresh evidence directory and owned
# process-group cleanup.
set -euo pipefail

ROOT=${ROOT:-/home/m0hawk/.local/share/sepalith-campaign-20260915}
CANDIDATE=${CANDIDATE:?set CANDIDATE=Q8_0 or Q6_K}
PORT=${PORT:-18404}
RUN_ID=${RUN_ID:?set a fresh RUN_ID; an existing run is never reused}
RUN_DIR="$ROOT/runs/$RUN_ID"
SERVER="$ROOT/build-b10453-vulkan-avx2/bin/llama-server"

case "$CANDIDATE" in
  Q8_0)
    MODEL="$ROOT/models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf"
    EXPECTED_MODEL_BYTES=2679710496
    EXPECTED_MODEL_SHA=22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559
    ;;
  Q6_K)
    MODEL="$ROOT/models/SFT-primary-step1000-quant-candidates-c/model-Q6_K.gguf"
    EXPECTED_MODEL_BYTES=2199741216
    EXPECTED_MODEL_SHA=7ea2ddfd45dce35d5016af1d1ba14408d29764df5a90f3a0dfe6f886ecec3578
    ;;
  *) echo "unsupported CANDIDATE=$CANDIDATE" >&2; exit 2 ;;
esac
EXPECTED_SERVER_SHA=92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing existing run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"

reason=server_exit
child=
started_epoch=$(date -u +%s)
write_launch() {
  cat > "$RUN_DIR/launch.json" <<EOF
{
  "schema": "sepalith.run01.theta0-notebook-launch.v1",
  "task": "RUN-01",
  "owner": "root",
  "candidate": "$CANDIDATE",
  "run_id": "$RUN_ID",
  "root": "$ROOT",
  "server": "$SERVER",
  "server_sha256": "$EXPECTED_SERVER_SHA",
  "model": "$MODEL",
  "model_sha256": "$EXPECTED_MODEL_SHA",
  "profile": {"port": $PORT, "threads": 6, "threads_batch": 6, "threads_http": 2, "context": 4096, "batch": 256, "ubatch": 256, "parallel": 1, "ngl": 99, "flash_attention": "on", "verbosity": 4},
  "argv": ["$SERVER", "-m", "$MODEL", "--host", "127.0.0.1", "--port", "$PORT", "-t", "6", "-tb", "6", "--threads-http", "2", "--parallel", "1", "-c", "4096", "-b", "256", "-ub", "256", "-ngl", "99", "-fa", "on", "-lv", "4"],
  "timeout_seconds": 1100,
  "probe_owner": "root; run theta0_notebook_probe.py against this already-running endpoint",
  "quality_claims": "none"
}
EOF
  chmod 600 "$RUN_DIR/launch.json"
}
write_launch

server_sha=$(sha256sum "$SERVER" | awk '{print $1}')
[[ "$server_sha" == "$EXPECTED_SERVER_SHA" ]] || { echo "server hash mismatch" >&2; reason=server_hash_mismatch; exit 3; }
model_bytes=$(stat -c '%s' "$MODEL")
[[ "$model_bytes" == "$EXPECTED_MODEL_BYTES" ]] || { echo "model byte count mismatch" >&2; reason=model_size_mismatch; exit 3; }
model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
[[ "$model_sha" == "$EXPECTED_MODEL_SHA" ]] || { echo "model hash mismatch" >&2; reason=model_hash_mismatch; exit 3; }

on_signal() {
  reason=interrupted
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
  fi
}
trap on_signal INT TERM

env CUDA_VISIBLE_DEVICES= GGML_CUDA_VISIBLE_DEVICES=-1 \
  setsid timeout --signal=TERM --kill-after=10s 1100s \
  "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -t 6 -tb 6 --threads-http 2 --parallel 1 -c 4096 -b 256 -ub 256 \
  -ngl 99 -fa on -lv 4 > "$RUN_DIR/server.log" 2>&1 &
child=$!
printf '{"child_pid":%s,"started_utc":"%s"}\n' "$child" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/process.json"
chmod 600 "$RUN_DIR/server.log" "$RUN_DIR/process.json"

set +e
wait "$child"
rc=$?
set -e
if [[ "$reason" == server_exit && "$rc" -eq 124 ]]; then reason=hard_timeout; fi
if [[ "$reason" == server_exit && "$rc" -eq 143 ]]; then reason=server_timeout_signal; fi
cat > "$RUN_DIR/terminal.json" <<EOF
{
  "schema": "sepalith.run01.theta0-notebook-terminal.v1",
  "ended_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "reason": "$reason",
  "child_exit_code": $rc,
  "child_pid": $child,
  "server_sha256": "$server_sha",
  "model_sha256": "$model_sha",
  "probe_followup": "not run by this launcher; root must run three-case probe with a fresh output path",
  "quality_claims": "none"
}
EOF
chmod 600 "$RUN_DIR/terminal.json"
exit "$rc"
