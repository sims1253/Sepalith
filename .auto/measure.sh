#!/bin/bash
# Autoresearch measure script: run the frozen PRM03 session bench once.
# Emits METRIC name=value lines. Config under test: configs/current.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${SEPALITH_PY:-/home/m0hawk/sandbox/Sepalith/.venv/bin/python}"

# fast pre-checks (<2s): syntax + config validity
"$PY" -m py_compile experiments/serving/prefill/bench.py experiments/serving/prefill/build_sessions.py
"$PY" - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("experiments/serving/prefill/configs/current.json").read_text())
assert cfg["model"] and isinstance(cfg["server_args"], list)
EOF

# dispatch: configs may select a probe instead of the session bench
MODE=$(/home/m0hawk/sandbox/Sepalith/.venv/bin/python - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("experiments/serving/prefill/configs/current.json").read_text())
print(cfg.get("bench", "sessions"))
EOF
)
case "$MODE" in
  probe_interrupt) exec "$PY" experiments/serving/prefill/probe_interrupt.py ;;
  probe_parallel) exec "$PY" experiments/serving/prefill/probe_parallel.py ;;
  probe_slots) exec "$PY" experiments/serving/prefill/probe_slots.py ;;
  probe_tokenize) exec "$PY" experiments/serving/prefill/probe_tokenize.py ;;
esac

# host-load guard: warn (do not fail) when the box is busy
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
echo "host_load1=${LOAD1}"

CFG_PASSES=$("$PY" - <<'EOF2'
import json, pathlib
print(json.loads(pathlib.Path("experiments/serving/prefill/configs/current.json").read_text()).get("passes", 2))
EOF2
)
exec "$PY" experiments/serving/prefill/bench.py \
  --config experiments/serving/prefill/configs/current.json \
  --sessions "${SESSIONS:-experiments/serving/prefill/sessions.jsonl}" \
  --manifest "${MANIFEST:-experiments/serving/prefill/manifest.json}" \
  --passes "$CFG_PASSES" \
  --port "${PORT:-18071}" \
  ${EXTRA:+$EXTRA}
