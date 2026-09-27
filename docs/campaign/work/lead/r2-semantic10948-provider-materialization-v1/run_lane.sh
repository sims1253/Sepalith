#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo 'usage: run_lane.sh LANE CORE CONTEXT RESERVE INPUT_ROOT OUTPUT_ROOT' >&2; exit 2; fi
lane="$1";core="$2";context="$3";reserve="$4";inputs="$5";outputs="$6";packet="$(cd "$(dirname "$0")" && pwd)"
case "$lane" in
  0) shards=(0012 0014 0016 0018 0020 0022 0024 0026) ;;
  1) shards=(0013 0015 0017 0019 0021 0023 0025) ;;
  *) echo 'lane must be 0 or 1' >&2; exit 2 ;;
esac
for shard in "${shards[@]}"; do bash "$packet/run_shard.sh" "$shard" "$core" "$context" "$reserve" "$inputs" "$outputs"; done
