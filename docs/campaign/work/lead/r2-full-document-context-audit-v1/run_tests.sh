#!/usr/bin/env bash
set -euo pipefail
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-full-document-context-audit-v1
PROD=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith
PY=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
PYTHONNOUSITE=1 taskset -c 0,2 "$PY" "$P/build_inputs.py"
taskset -c 0,2 node --no-warnings=ExperimentalWarning --experimental-strip-types "$P/render_full_documents.ts" "$P/inputs.json" "$P/rendered.json"
PYTHONNOUSITE=1 taskset -c 0,2 "$PY" "$P/measure_tokens.py"
taskset -c 0,2 node --no-warnings=ExperimentalWarning --experimental-strip-types "$P/render_provider_comparison.ts" "$P/inputs.json" "$P/provider-comparison-prompts.json"
PYTHONNOUSITE=1 taskset -c 0,2 "$PY" "$P/compare_tokens.py"
taskset -c 0,2 node --no-warnings=ExperimentalWarning --experimental-strip-types "$P/test_full_document.ts" "$P/inputs.json" "$P/rendered.json"
cd "$P" && "$PROD/node_modules/.bin/tsc" --noEmit --typeRoots "$PROD/node_modules/@types"
python3 "$P/summarize.py" >/dev/null
