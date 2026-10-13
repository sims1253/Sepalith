#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET=$ROOT/docs/campaign/work/lead/r2-semantic-serving-geometry-parity-v1
TSC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/.bin/tsc
TYPES=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/@types
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
cd "$ROOT"
node --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/check_current_serving.ts" "$PACKET/current-baseline.json"
node --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/check_parity.ts" "$PACKET/parity-output.json"
"$TSC" --noEmit --strict --skipLibCheck --target ES2022 --module commonjs --moduleResolution node --allowImportingTsExtensions --types node --typeRoots "$TYPES" "$PACKET/check_current_serving.ts" "$PACKET/check_parity.ts" "$PACKET/check_actual_full_parity.ts" "$PACKET/source/context_select.ts"
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/verify_materializer_geometry.py" "$PACKET/parity-output.json" "$PACKET/materializer-crosscheck.json"
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/audit_actual_train_row.py"
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/build_actual_full_fixture.py"
node --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/check_actual_full_parity.ts" "$PACKET/actual-full-fixture.json" "$PACKET/actual-full-parity.json"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$PACKET/verify_materializer_geometry.py" "$PACKET/audit_actual_train_row.py" "$PACKET/build_actual_full_fixture.py"
printf '%s\n' 'PASS all local checks'
