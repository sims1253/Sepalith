#!/usr/bin/env bash
# Run mine_edit_pairs.py as N parallel nice'd shards over cloned repos.
# Resumable: completed repos (in spool/_progress.jsonl) are skipped on restart.
set -u
REPOS_DIR=/mnt/h/sepalith/git
SPOOL=/mnt/h/sepalith/datasets/edit_pairs_v1/spool
PY=/home/m0hawk/Documents/Sepalith/.venv/bin/python
SHARDS=${1:-6}
PER_REPO=${2:-30}
SINCE=${3:-2026-05-01}
MAXC=${4:-120}

mkdir -p "$SPOOL" /mnt/h/sepalith/logs
exec 9>/mnt/h/sepalith/logs/mining.lock
flock -n 9 || { echo "another mining run is active; exiting"; exit 0; }
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  nice -n 15 "$PY" /home/m0hawk/Documents/Sepalith/experiments/data-mining/mine_edit_pairs.py \
    --repos-dir "$REPOS_DIR" --spool "$SPOOL" \
    --per-repo "$PER_REPO" --since "$SINCE" --max-commits "$MAXC" \
    --shard "$i" --shards "$SHARDS" \
    > "/mnt/h/sepalith/logs/mine_shard$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shards: ${pids[*]}"
wait "${pids[@]}"
echo "all shards finished at $(date)"
