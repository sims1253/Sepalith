#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET=$ROOT/docs/campaign/work/lead/r2-sourcewalk-semantic-materialization-v3
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
TSC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/.bin/tsc
TYPES=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/node_modules/@types
cd "$ROOT"
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/test_materialize.py" -v
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/build_actual_parity_fixture.py"
node --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/check_actual_parity.ts" "$PACKET/actual-parity-fixture.json" "$PACKET/actual-parity-result.json"
"$TSC" --noEmit --strict --skipLibCheck --target ES2022 --module commonjs --moduleResolution node --allowImportingTsExtensions --types node --typeRoots "$TYPES" "$PACKET/check_actual_parity.ts"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$PACKET/materialize_semantic.py" "$PACKET/campaign_selection.py" "$PACKET/semantic_context_geometry_v2.py" "$PACKET/build_actual_parity_fixture.py"
printf '%s\n' 'PASS all v3 preparation checks'
