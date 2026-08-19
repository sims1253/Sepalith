#!/usr/bin/env bash
# Continuously mine newly-cloned repos in waves until cloning is done and
# every cloned repo is in the progress file. Safe to restart at any time.
set -u
SPOOL=/mnt/h/sepalith/datasets/edit_pairs_v1/spool
GITDIR=/mnt/h/sepalith/git
CLONE_PID=${1:-0}   # pid of the clone driver to watch (0 = unknown)

while :; do
  done_n=$(wc -l < "$SPOOL/_progress.jsonl" 2>/dev/null || echo 0)
  dirs_n=$(ls "$GITDIR" | wc -l)
  echo "$(date +%H:%M:%S) cloned=$dirs_n mined=$done_n examples=$(cat $SPOOL/*__*.jsonl 2>/dev/null | wc -l)"
  clone_running=0
  [ "$CLONE_PID" != "0" ] && kill -0 "$CLONE_PID" 2>/dev/null && clone_running=1
  if [ "$clone_running" = "0" ] && [ "$done_n" -ge "$dirs_n" ]; then
    echo "$(date +%H:%M:%S) all cloned repos mined; done"
    break
  fi
  if [ "$done_n" -lt "$dirs_n" ]; then
    bash /home/m0hawk/Documents/Sepalith/experiments/data-mining/run_mining.sh 8 30 2026-05-01 250 \
      >> /mnt/h/sepalith/logs/mine_waves_run.log 2>&1
  else
    sleep 90
  fi
done
