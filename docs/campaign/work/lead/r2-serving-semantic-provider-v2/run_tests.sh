#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET=$ROOT/docs/campaign/work/lead/r2-serving-semantic-provider-v2
PROD=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
NODE=node
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/build_actual_train_fixture.py"
taskset -c 0,2 "$NODE" --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/test_provider.ts" "$PACKET/actual-train-fixture.json" "$PACKET/test-results.json"
cd "$PACKET"
"$PROD/node_modules/.bin/tsc" --noEmit --typeRoots "$PROD/node_modules/@types"
"$PROD/node_modules/.bin/esbuild" source/extension.ts --bundle --platform=node --format=cjs --external:vscode --outfile=/tmp/sepalith-semantic-provider-extension.cjs >/dev/null
printf '%s\n' 'PASS semantic provider production integration candidate'
