#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo 'usage: run_lane.sh LANE CORE CONTEXT RESERVE INPUT_ROOT OUTPUT_ROOT' >&2; exit 2; fi
lane="$1"; core="$2"; context="$3"; reserve="$4"; inputs="$5"; outputs="$6"; packet="$(cd "$(dirname "$0")" && pwd)"
[[ "$lane" == 0 || "$lane" == 1 ]] || { echo 'lane must be 0 or 1' >&2; exit 2; }
mapfile -t shards < <(python3 - "$inputs/manifest.json" "$lane" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); lane=int(sys.argv[2])
assert m['status'] in ('complete_target_free_review_only','complete_target_free') and m['training_admission'] is False
for i,x in enumerate(m['shards']):
 if i%2==lane and x.get('provider_rows',x.get('rows',0))>0:print(f"{int(x['shard']):04d}")
PY
)
for shard in "${shards[@]}"; do bash "$packet/run_shard.sh" "$shard" "$core" "$context" "$reserve" "$inputs" "$outputs"; done
