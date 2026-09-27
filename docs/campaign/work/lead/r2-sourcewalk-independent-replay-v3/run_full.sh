#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo 'usage: run_full.sh FRESH_OUTPUT_DIR' >&2; exit 64; fi
OUT=$1
SOURCE=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-sourcewalk-independent-replay-v3/full_replay.py
EXPECTED_SOURCE=63d165ae1488c1f38174f7e24f68a51e4c80c2d48e5aec44a0446fa062a61f7f
[[ $(sha256sum "$SOURCE" | cut -d' ' -f1) == "$EXPECTED_SOURCE" ]] || { echo source_hash_mismatch >&2; exit 65; }
mkdir -p "$OUT"
exec 9>"$OUT/.owner.lock"
flock -n 9 || { echo output_owned_by_another_process >&2; exit 66; }
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/controller-exit-code.txt"; exit "$rc"' EXIT
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
taskset -c 0,2 nice -n 10 ionice -c 3 python3 "$SOURCE" build-index --output "$OUT" --shards all
taskset -c 0,2 nice -n 10 ionice -c 3 python3 "$SOURCE" replay --output "$OUT" --shards all
taskset -c 0,2 nice -n 10 ionice -c 3 python3 "$SOURCE" merge --output "$OUT"
