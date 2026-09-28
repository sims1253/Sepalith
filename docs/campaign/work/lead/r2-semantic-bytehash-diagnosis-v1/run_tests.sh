#!/usr/bin/env bash
set -euo pipefail
P=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-semantic-bytehash-diagnosis-v1
PYTHONNOUSERSITE=1 taskset -c 0,2 python3 -B -m unittest discover -s "$P/tests" -v
PYTHONNOUSERSITE=1 taskset -c 0,2 python3 -m py_compile "$P/source/analyze_semantics.py" "$P/reproduce_exact.py"
taskset -c 0,2 python3 "$P/reproduce_exact.py"
PYTHONNOUSERSITE=1 taskset -c 0,2 python3 "$P/run_exact_pipeline.py"
