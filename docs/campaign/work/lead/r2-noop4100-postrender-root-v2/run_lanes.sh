#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: run_fallback32_lanes.sh FALLBACK_ROOT OUTPUT_ROOT' >&2; exit 2; }
F=$1; OUT=$2; R=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-noop4100-postrender-root-v2/run_shard.sh
[[ ! -e "$OUT" ]] || { echo 'fresh output required' >&2; exit 3; }; mkdir -p "$OUT"
python3 - "$F/manifest.json" <<'PY' > "$OUT/launch.tsv"
import json,sys
m=json.load(open(sys.argv[1]));assert m['schema']=='sepalith.dat10.noop4100.provider-run-plan.v1' and m['phase']=='render32' and m['provider_denominator']==4100 and m['upstream_holds']==127 and m['execution_authorized'] is False
for x in m['entries']:
 if x['rows']:print(f"{x['shard']:04d}\t{x['core']}")
PY
run_lane(){ local core=$1; while IFS=$'\t' read -r shard assigned; do [[ "$assigned" != "$core" ]] || bash "$R" "$shard" "$core" 32768 2048 "$F" "$OUT"; done < "$OUT/launch.tsv"; }
run_lane 4 & p0=$!; run_lane 6 & p1=$!; rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; exit "$rc"
