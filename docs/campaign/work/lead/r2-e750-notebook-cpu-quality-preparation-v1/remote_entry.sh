#!/usr/bin/env bash
set -euo pipefail
run_id=$1
admission=$2
base=$HOME/.local/share/sepalith-e750-notebook-cpu-quality-v1
run=$base/runs/$run_id
log=$base/logs/$run_id.log
identity=$base/logs/$run_id.runner-identity.json
test ! -e "$run"
test ! -e "$identity"
pid=$$
start_tick=$(awk '{print $22}' /proc/$$/stat)
printf '{"pid":%s,"start_tick":"%s","run_id":"%s"}\n' "$pid" "$start_tick" "$run_id" > "$identity"
exec /usr/bin/taskset --cpu-list 0,2 /usr/bin/python3 -B "$base/packet-final/run_suite.py" --packet "$base/packet-final/packet.json" --admission "$admission" --run-root "$run" >>"$log" 2>&1
