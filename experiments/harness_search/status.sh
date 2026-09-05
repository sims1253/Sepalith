#!/bin/bash
# One-line H1 pipeline status (safe to run anytime; read-only).
R=/home/m0hawk/Documents/Sepalith/experiments/harness_search/results
echo "$(date '+%F %T') phase: $(tail -1 $R/pipeline.log 2>/dev/null)"
for A in hill population gepa; do
  [ -f $R/$A/state.json ] && /home/m0hawk/Documents/Sepalith/.venv/bin/python -c "
import json
s=json.load(open('$R/$A/state.json'))
ev=s['evaluated']
print(f'  $A: iters {s[\"iterations_done\"]}/13, cands {len(ev)}, best {s[\"best_entry\"][\"exact_pass\"]:.4f} ({s[\"best_entry\"][\"fp\"]})')
"
done
for P in 18310 18311; do
  printf "  srv %s: %s reqs\n" $P "$(grep -c launch_slot $R/llama-server-h1-$P.log 2>/dev/null)"
done
[ -f $R/verdict.json ] && echo "  VERDICT: $R/verdict.json exists"
[ -f $R/PIPELINE_DONE ] && echo "  PIPELINE DONE"
