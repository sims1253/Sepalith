#!/bin/bash
# Autoresearch measure script: run the frozen PRM03 session bench once.
# Emits METRIC name=value lines. Config under test: configs/current.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/m0hawk/sandbox/Sepalith/.venv/bin/python

# fast pre-checks (<2s): syntax + config validity
"$PY" -m py_compile experiments/serving/prefill/bench.py experiments/serving/prefill/build_sessions.py
"$PY" - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("experiments/serving/prefill/configs/current.json").read_text())
assert cfg["model"] and isinstance(cfg["server_args"], list)
EOF

# host-load guard: warn (do not fail) when the box is busy
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
echo "host_load1=${LOAD1}"

exec "$PY" experiments/serving/prefill/bench.py \
  --config experiments/serving/prefill/configs/current.json \
  --passes "${PASSES:-2}" \
  --port "${PORT:-18071}"
