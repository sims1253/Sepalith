#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PACKET=$ROOT/docs/campaign/work/lead/r2-full-document-policy-v1
PROD=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
NODE=node
PYTHONNOUSERSITE=1 taskset -c 0,2 "$NODE" --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/render_candidates.ts" "$ROOT/docs/campaign/work/lead/r2-full-document-context-audit-v1/inputs.json" "$PACKET/candidate-prompts.json"
PYTHONNOUSERSITE=1 taskset -c 0,2 "$PY" "$PACKET/build_policy_fixtures.py"
taskset -c 0,2 "$NODE" --no-warnings=ExperimentalWarning --experimental-strip-types "$PACKET/test_policy.ts" "$PACKET/policy-fixtures.json" "$PACKET/policy-results.json"
cd "$PACKET"
"$PROD/node_modules/.bin/tsc" --noEmit --typeRoots "$PROD/node_modules/@types"
"$PROD/node_modules/.bin/esbuild" source/extension.ts --bundle --platform=node --format=cjs --external:vscode --outfile=/tmp/sepalith-full-document-policy-v1-extension.cjs >/dev/null
cmp policy-results.json notebook/policy-results.json
printf '%s\n' 'PASS full-document prediction policy preparation'
