#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: run64_lanes.sh PREPARED_ROOT OUTPUT_ROOT' >&2; exit 2; }
prepared="$1"; output="$2"
runner=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-noop4100-postrender-root-v2/run_shard.sh
[[ ! -e "$output" ]] || { echo 'fresh render64 output required' >&2; exit 3; }
mkdir -p "$output"
python3 - "$prepared/manifest.json" <<'PY' > "$output/launch.tsv"
import json,sys
m=json.load(open(sys.argv[1]));assert m['schema']=='sepalith.dat10.noop4100.context64-queue.v1' and m['status']=='prepared_inputs_only_not_rendered'
assert m['rows']==38 and m['retained_holds']==67 and m['prior_supported']==3995 and m['closure']=='3995+38+67=4100'
for x in m['entries']:
 if x['rows']: print(f"{x['shard']:04d}\t{x['core']}")
PY
run_lane(){ local core="$1"; while IFS=$'\t' read -r shard assigned; do [[ "$assigned" != "$core" ]] || bash "$runner" "$shard" "$core" 65536 2048 "$prepared/inputs" "$output"; done < "$output/launch.tsv"; }
export -f run_lane; export runner prepared output
timeout --signal=TERM --kill-after=30s 3600 bash -c 'run_lane "$1"' _ 4 & p0=$!
timeout --signal=TERM --kill-after=30s 3600 bash -c 'run_lane "$1"' _ 6 & p1=$!
rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; exit "$rc"
