#!/usr/bin/env bash
set -euo pipefail
PACKET="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-inputs-v1"
OUTPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1"
OLD_OUTPUT_ROOT="/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-v1"
[[ "$OUTPUT_ROOT" != "$OLD_OUTPUT_ROOT" ]] || { echo 'refusing to use preserved failed output root' >&2; exit 3; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "fresh retry root required: $OUTPUT_ROOT" >&2; exit 3; }
mkdir -p "$OUTPUT_ROOT"

# Each lane continues after a failed shard so every terminal record contains
# rich phase/code/signal diagnostics. A nonzero aggregate still fails the
# command; no failed shard is converted into a hold or silently skipped.
run_lane() {
  local core="$1"; shift
  local failures=()
  for shard in "$@"; do
    if ! bash "$PACKET/run_shard_rich.sh" "$shard" "$core" 16384 2048 "$INPUT_ROOT" "$OUTPUT_ROOT"; then
      failures+=("$shard")
    fi
  done
  if ((${#failures[@]})); then
    printf 'failed shards on core %s: %s\n' "$core" "${failures[*]}" >&2
    return 1
  fi
}

run_lane 8 0027 0029 0031 0033 0035 0037 0039 & lane0=$!
run_lane 10 0028 0030 0032 0034 0036 0038 0040 & lane1=$!
if wait "$lane0"; then rc0=0; else rc0=$?; fi
if wait "$lane1"; then rc1=0; else rc1=$?; fi
((rc0==0 && rc1==0))
