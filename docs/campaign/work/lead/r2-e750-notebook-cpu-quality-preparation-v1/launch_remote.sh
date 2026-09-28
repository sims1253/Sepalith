#!/usr/bin/env bash
set -euo pipefail
run_id=$1
admission=$2
base=$HOME/.local/share/sepalith-e750-notebook-cpu-quality-v1
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]]
test -f "$admission"
test ! -e "$base/runs/$run_id"
test ! -e "$base/logs/$run_id.runner-identity.json"
! tmux has-session -t "e750-cap-$run_id" 2>/dev/null
tmux new-session -d -s "e750-cap-$run_id" "$base/packet-final/remote_entry.sh" "$run_id" "$admission"
printf 'session=%s run=%s\n' "e750-cap-$run_id" "$base/runs/$run_id"
