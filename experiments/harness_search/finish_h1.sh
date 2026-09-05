#!/bin/bash
# H1 finisher — run AFTER the bake-off pipeline completes (results/PIPELINE_DONE
# exists). Appends the glm-judged intent leg to verdict.json, prints the full
# budget accounting + the pre-registered verdict analysis, and leaves
# H1_RESULTS.md sections 4-6 to fill from this output.
set -u
cd /home/m0hawk/Documents/Sepalith
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python
RIG=/home/m0hawk/Documents/Sepalith/experiments/harness_search
RES=$RIG/results

if [ ! -f "$RES/PIPELINE_DONE" ]; then
  echo "pipeline not done yet (no $RES/PIPELINE_DONE); GEPA+verdict still running:"
  bash "$RIG/status.sh"
  exit 1
fi

echo "== intent leg (glm-5.3 judge; needs ZAI key from ~/.zshrc) =="
nice -n 15 "$PY" "$RIG/verdict_battery.py" --results "$RES" \
  --ports 18310,18311 --intent-only 2>&1 | tail -8

echo "== summary + budget accounting + verdict analysis =="
"$PY" "$RIG/summarize.py" --results "$RES"
