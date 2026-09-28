#!/usr/bin/env bash
set -euo pipefail

# This is a root-launched command.  It must run only after
# verify_render16.sh and prepare_fallback_9534.sh succeed.  It uses the same
# parse-unavailable rich provider closure as the fresh 16K retry.
if [[ $# -ne 1 ]]; then
  echo "usage: run_fallback32_lanes.sh FALLBACK_INPUT_ROOT" >&2
  exit 2
fi
FALLBACK_INPUTS="$1"
PACKET="/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-provider-parse-unavailable-preparation-v1/semantic9534-retry"
RICH_SHARD="$PACKET/run_shard_rich.sh"
OUTPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-32k-semantic9534-postrender-root-v1"
OLD_16K="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-v1"
[[ "$OUTPUT_ROOT" != "$OLD_16K" ]] || { echo "refusing preserved output root" >&2; exit 3; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "fresh 32K output root required: $OUTPUT_ROOT" >&2; exit 3; }
[[ -x "$RICH_SHARD" ]] || { echo "missing reviewed rich provider: $RICH_SHARD" >&2; exit 3; }
mkdir -p "$OUTPUT_ROOT"

mapfile -t LANE0 < <(python3 - "$FALLBACK_INPUTS/manifest.json" 0 <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8')); lane=int(sys.argv[2])
assert m['schema']=='sepalith.dat10.semantic9534.fallback32_inputs.v1'
assert m['status']=='complete_target_free_review_only' and m['training_admission'] is False
for i,x in enumerate(m['shards']):
    if i%2==lane and int(x.get('provider_rows',x.get('rows',0)))>0:
        print(f"{int(x['shard']):04d}")
PY
)
mapfile -t LANE1 < <(python3 - "$FALLBACK_INPUTS/manifest.json" 1 <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8')); lane=int(sys.argv[2])
assert m['schema']=='sepalith.dat10.semantic9534.fallback32_inputs.v1'
assert m['status']=='complete_target_free_review_only' and m['training_admission'] is False
for i,x in enumerate(m['shards']):
    if i%2==lane and int(x.get('provider_rows',x.get('rows',0)))>0:
        print(f"{int(x['shard']):04d}")
PY
)

run_lane() {
  local core="$1"; shift
  local failures=()
  for shard in "$@"; do
    if ! bash "$RICH_SHARD" "$shard" "$core" 32768 2048 "$FALLBACK_INPUTS" "$OUTPUT_ROOT"; then
      failures+=("$shard")
    fi
  done
  if ((${#failures[@]})); then
    printf 'failed 32K shards on core %s: %s\n' "$core" "${failures[*]}" >&2
    return 1
  fi
}

run_lane 8 "${LANE0[@]}" & lane0=$!
run_lane 10 "${LANE1[@]}" & lane1=$!
if wait "$lane0"; then rc0=0; else rc0=$?; fi
if wait "$lane1"; then rc1=0; else rc1=$?; fi
((rc0==0 && rc1==0))
