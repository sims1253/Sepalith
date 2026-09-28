#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET=$ROOT/docs/campaign/work/lead/r2-semantic763-context-admission-v1
DATA=/mnt/e/sepalith/campaign-20260915/data-work/Semantic763-context-admission-v1
PROD=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 0,2 "$PY" "$PACKET/test_outputs.py" "$DATA" > "$PACKET/test-results.json"
cd "$PACKET"
"$PROD/node_modules/.bin/tsc" --noEmit --typeRoots "$PROD/node_modules/@types"
"$PROD/node_modules/.bin/esbuild" render_prediction_inputs.ts --bundle --platform=node --format=esm --outfile=/tmp/semantic763-render-check.mjs >/dev/null
printf '%s\n' 'PASS semantic763 context admission preparation'
